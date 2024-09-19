
import os
import csv
import time
import fnmatch
from pydoc import locate
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchnet.meter as tnt

import ai8x
import distiller
import parsecmd
import parse_qat_yaml

from distiller import apputils
from distiller.data_loggers import PythonLogger

from train import (create_model, update_old_model_params, create_optimizer, create_activation_stats_collectors,
                   OVERALL_LOSS_KEY, OBJECTIVE_LOSS_KEY)

from brainmepnas import AccuracyMetrics

# import matplotlib
# import shap
#
# import sample
#
# import nnplot
#
# from distiller.quantization.range_linear import PostTrainLinearQuantizer
# from distiller.data_loggers.collector import (QuantCalibrationStatsCollector,
#                                               RecordsActivationStatsCollector,
#                                               SummaryActivationStatsCollector, collectors_context)
# from torchmetrics.detection.map import MAP as MeanAveragePrecision
# from utils import kd_relationbased, object_detection_utils, parse_obj_detection_yaml

# Logger handle
msglogger = None


def load_model():

    script_dir = os.path.dirname(__file__)
    global msglogger

    supported_models = []
    supported_sources = []
    model_names = []
    dataset_names = []

    # Dynamically load models
    for _, _, files in sorted(os.walk('models')):
        for name in sorted(files):
            if fnmatch.fnmatch(name, '*.py'):
                fn = 'models.' + name[:-3]
                m = locate(fn)
                try:
                    for i in m.models:
                        i['module'] = fn
                    supported_models += m.models
                    model_names += [item['name'] for item in m.models]
                except AttributeError:
                    # Skip files that don't have 'models' or 'models.name'
                    pass

    # Dynamically load datasets
    for _, _, files in sorted(os.walk('datasets')):
        for name in sorted(files):
            if fnmatch.fnmatch(name, '*.py'):
                ds = locate('datasets.' + name[:-3])
                try:
                    supported_sources += ds.datasets
                    dataset_names += [item['name'] for item in ds.datasets]
                except AttributeError:
                    # Skip files that don't have 'datasets' or 'datasets.name'
                    pass

    args = parsecmd.get_parser(model_names, dataset_names).parse_args()
    ai8x.set_device(args.device, args.act_mode_8bit, args.avg_pool_rounding)

    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name,
                                         args.output_dir)

    args.optimizer = 'SGD'
    args.lr = 0.1
    args.device = 'cpu'
    args.gpus = -1

    selected_source = next((item for item in supported_sources if item['name'] == args.dataset))
    args.labels = selected_source['output']
    args.num_classes = len(args.labels)

    dimensions = selected_source['input']
    if len(dimensions) == 2:
        dimensions += (1, )
    args.dimensions = dimensions

    args.datasets_fn = selected_source['loader']
    args.collate_fn = selected_source.get('collate')  # .get returns None if key does not exist

    model = create_model(supported_models, dimensions, args)

    pylogger = PythonLogger(msglogger, log_1d=True)
    all_loggers = [pylogger]

    qat_policy = parse_qat_yaml.parse(args.qat_policy) \
        if args.qat_policy.lower() != "none" else None

    if args.load_model_path:
        update_old_model_params(args.load_model_path, model)
        if qat_policy is not None:
            checkpoint = torch.load(args.load_model_path,
                                    map_location=lambda storage, loc: storage)
            # pylint: disable=unsubscriptable-object
            if checkpoint.get('epoch', None) >= qat_policy['start_epoch']:
                ai8x.fuse_bn_layers(model)
                if args.name:
                    args.name = f'{args.name}_qat'
                else:
                    args.name = 'qat'
            # pylint: enable=unsubscriptable-object
        model = apputils.load_lean_checkpoint(model, args.load_model_path,
                                              model_device=args.device)
        ai8x.update_model(model)

    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = create_optimizer(model, args)
    msglogger.info('Optimizer Type: %s', type(optimizer))
    msglogger.info('Optimizer Args: %s', optimizer.defaults)

    activations_collectors = create_activation_stats_collectors(model, *args.activation_stats)

    train_loader, val_loader, test_loader, _ = apputils.get_data_loaders(
        args.datasets_fn, (os.path.expanduser(args.data), args), args.batch_size,
        args.workers, args.validation_split, args.deterministic,
        args.effective_train_size, args.effective_valid_size, args.effective_test_size,
        test_only=args.evaluate, collate_fn=args.collate_fn, cpu=args.device == 'cpu', rand_data=False)
    assert args.evaluate or train_loader is not None and val_loader is not None, \
        "No training and/or validation data in train mode"
    assert not args.evaluate or test_loader is not None, "No test data in eval mode"

    # print('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d'
    #       % (len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler)))
    msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    compression_scheduler = distiller.CompressionScheduler(model)
    model.to(args.device)

    # evaluate_model(model, criterion, test_loader, pylogger, activations_collectors,
    #                args, compression_scheduler)

    return model, criterion, test_loader, pylogger, args


def custom_evaluate(model, criterion, data_loader, loggers, args, epoch=-1):

    msglogger.info('--- test ---------------------')

    # predicted_labels_array = np.array([])
    # true_labels_array = np.array([])

    predicted_labels_array = []
    true_labels_array = []

    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, min(args.num_classes, 5)))

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size

    if args.display_confusion:
        confusion = tnt.ConfusionMeter(args.num_classes)
    total_steps = (total_samples + batch_size - 1) // batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    model.eval()

    end = time.time()

    with (torch.no_grad()):
        m = model.module if isinstance(model, nn.DataParallel) else model

        for validation_step, (inputs, target) in enumerate(data_loader):
            inputs, target = inputs.to(args.device), target.to(args.device)
            output = model(inputs)

            if args.act_mode_8bit:
                output /= 128.
                for key in model.__dict__['_modules'].keys():
                    if (hasattr(model.__dict__['_modules'][key], 'wide')
                            and model.__dict__['_modules'][key].wide):
                        output /= 256.
                if args.regression:
                    target /= 128.

            loss = criterion(output, target)

            # measure accuracy and record loss
            losses[OBJECTIVE_LOSS_KEY].add(loss.item())

            if len(output.data.shape) <= 2 or args.regression:
                classerr.add(output.data, target)
            else:
                classerr.add(output.data.permute(0, 2, 3, 1).flatten(start_dim=0,
                                                                     end_dim=2),
                             target.flatten())
            if args.display_confusion:
                confusion.add(output.data, target)

            predicted_labels = output.data.cpu().numpy()
            predicted_labels = np.argmax(predicted_labels, 1)
            true_labels = target.cpu().numpy()

            # predicted_labels_array = np.append(predicted_labels_array, predicted_labels)
            # true_labels_array = np.append(true_labels_array, true_labels)

            predicted_labels_array.append(predicted_labels)
            true_labels_array.append(true_labels)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = validation_step + 1

            if steps_completed % args.print_freq == 0 or steps_completed == total_steps:
                if not args.regression:
                    stats = (
                        '',
                        OrderedDict([('Loss', losses[OBJECTIVE_LOSS_KEY].mean),
                                     ('Top1', classerr.value(1))])
                    )
                    if args.num_classes > 5:
                        stats[1]['Top5'] = classerr.value(5)

                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)

    msglogger.info('==> Top1: %.3f    Loss: %.3f\n',
                   classerr.value()[0], losses[OBJECTIVE_LOSS_KEY].mean)

    msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))

    return predicted_labels_array, true_labels_array


# ----------------------------------------------------------------------------------------------------
# functions copied from train.py
# ----------------------------------------------------------------------------------------------------


# def create_model(supported_models, dimensions, args, mode='default'):
#     """Create the model"""
#     if mode == 'default':
#         module = next(item for item in supported_models if item['name'] == args.cnn)
#     elif mode == 'kd_teacher':
#         module = next(item for item in supported_models if item['name'] == args.kd_teacher)
#
#     # Override distiller's input shape detection. This is not a very clean way to do it since
#     # we're replacing a protected member.
#     distiller.utils._validate_input_shape = (  # pylint: disable=protected-access
#         lambda _a, _b: (1, ) + dimensions[:module['dim'] + 1]
#     )
#     if mode == 'default':
#         Model = locate(module['module'] + '.' + args.cnn)
#         if not Model:
#             raise RuntimeError("Model " + args.cnn + " not found\n")
#     elif mode == 'kd_teacher':
#         Model = locate(module['module'] + '.' + args.kd_teacher)
#         if not Model:
#             raise RuntimeError("Model " + args.kd_teacher + " not found\n")
#
#     if args.dr and ('dr' not in module or not module['dr']):
#         raise ValueError("Dimensionality reduction is not supported for this model")
#
#     # Set model parameters
#     if args.act_mode_8bit:
#         weight_bits = 8
#         bias_bits = 8
#         quantize_activation = True
#     else:
#         weight_bits = None
#         bias_bits = None
#         quantize_activation = False
#
#     model_args = {}
#     model_args["pretrained"] = False
#     model_args["num_classes"] = args.num_classes
#     model_args["num_channels"] = dimensions[0]
#     model_args["dimensions"] = (dimensions[1], dimensions[2])
#     model_args["bias"] = args.use_bias
#     model_args["weight_bits"] = weight_bits
#     model_args["bias_bits"] = bias_bits
#     model_args["quantize_activation"] = quantize_activation
#
#     if args.dr:
#         model_args["dimensionality"] = args.dr
#
#     if args.backbone_checkpoint:
#         model_args["backbone_checkpoint"] = args.backbone_checkpoint
#
#     if args.obj_detection:
#         model_args["device"] = args.device
#
#     if module['dim'] > 1 and module['min_input'] > dimensions[2]:
#         model_args["padding"] = (module['min_input'] - dimensions[2] + 1) // 2
#
#     model = Model(**model_args).to(args.device)
#
#     return model
#
#
# def create_optimizer(model, args):
#     """Create the optimizer"""
#     if args.optimizer.lower() == 'sgd':
#         optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
#                                     weight_decay=args.weight_decay)
#     elif args.optimizer.lower() == 'adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
#                                      weight_decay=args.weight_decay)
#     else:
#         msglogger.info('Unknown optimizer type: %s. SGD is set as optimizer!!!', args.optimizer)
#         optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
#                                     weight_decay=args.weight_decay)
#     return optimizer
#
#
# def update_old_model_params(model_path, model_new):
#     """Adds missing model parameters added with default values.
#     This is mainly due to the saved checkpoints from previous versions of the repo.
#     New model is saved to `model_path` and the old model copied into the same file_path with
#     `__obsolete__` prefix."""
#     is_model_old = False
#     model_old = torch.load(model_path,
#                            map_location=lambda storage, loc: storage)
#     # Fix up any instances of DataParallel
#     old_dict = model_old['state_dict'].copy()
#     for k in old_dict:
#         if k.startswith('module.'):
#             model_old['state_dict'][k[7:]] = old_dict[k]
#     for new_key, new_val in model_new.state_dict().items():
#         if new_key not in model_old['state_dict'] and 'bn' not in new_key:
#             is_model_old = True
#             model_old['state_dict'][new_key] = new_val
#             if 'compression_sched' in model_old:
#                 if 'masks_dict' in model_old['compression_sched']:
#                     model_old['compression_sched']['masks_dict'][new_key] = None
#
#     if is_model_old:
#         dir_path, file_name = os.path.split(model_path)
#         new_file_name = '__obsolete__' + file_name
#         old_model_path = os.path.join(dir_path, new_file_name)
#         os.rename(model_path, old_model_path)
#         torch.save(model_old, model_path)
#         msglogger.info('Model `%s` is old. Missing parameters added with default values!',
#                        model_path)
#
#
# def evaluate_model(model, criterion, test_loader, loggers, activations_collectors, args,
#                    scheduler=None):
#     """
#     This sample application can be invoked to evaluate the accuracy of your model on
#     the test dataset.
#     You can optionally quantize the model to 8-bit integer before evaluation.
#     For example:
#     python3 compress_classifier.py --arch resnet20_cifar \
#              ../data.cifar10 -p=50 --resume-from=checkpoint.pth.tar --evaluate
#     """
#
#     if not isinstance(loggers, list):
#         loggers = [loggers]
#
#     if args.quantize_eval:
#         model.cpu()
#         # if args.ai84:
#         #     quantizer = PostTrainLinearQuantizerAI84.from_args(model, args)
#         # else:
#         quantizer = PostTrainLinearQuantizer.from_args(model, args)
#         quantizer.prepare_model()
#         model.to(args.device)
#
#     top1, _, _, mAP = test(test_loader, model, criterion, loggers, activations_collectors,
#                            args=args)
#
#     if args.shap > 0:
#         matplotlib.use('TkAgg')
#         print("Generating plot...")
#         images, _ = iter(test_loader).next()
#         background = images[:100]
#         test_images = images[100:100 + args.shap]
#
#         # pylint: disable=protected-access
#         shap.explainers._deep.deep_pytorch.op_handler['Clamp'] = \
#             shap.explainers._deep.deep_pytorch.passthrough
#         shap.explainers._deep.deep_pytorch.op_handler['Empty'] = \
#             shap.explainers._deep.deep_pytorch.passthrough
#         shap.explainers._deep.deep_pytorch.op_handler['Floor'] = \
#             shap.explainers._deep.deep_pytorch.passthrough
#         shap.explainers._deep.deep_pytorch.op_handler['Quantize'] = \
#             shap.explainers._deep.deep_pytorch.passthrough
#         shap.explainers._deep.deep_pytorch.op_handler['Scaler'] = \
#             shap.explainers._deep.deep_pytorch.passthrough
#         # pylint: enable=protected-access
#         e = shap.DeepExplainer(model.to(args.device), background.to(args.device))
#         shap_values = e.shap_values(test_images.to(args.device))
#         shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
#         test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
#         # Plot the feature attributions
#         shap.image_plot(shap_numpy, -test_numpy)
#
#     if args.quantize_eval:
#         checkpoint_name = 'quantized'
#
#         if args.obj_detection:
#             extras = {'quantized_mAP': mAP}
#         else:
#             extras = {'quantized_top1': top1}
#         apputils.save_checkpoint(0, args.cnn, model, optimizer=None, scheduler=scheduler,
#                                  name='_'.join([args.name, checkpoint_name])
#                                  if args.name else checkpoint_name,
#                                  dir=msglogger.logdir, extras=extras)
#
#
# class missingdict(dict):
#     """This is a little trick to prevent KeyError"""
#
#     def __missing__(self, key):
#         return None  # note, does *not* set self[key] - we don't want defaultdict's behavior
#
#
# def create_activation_stats_collectors(model, *phases):
#     """Create objects that collect activation statistics.
#
#     This is a utility function that creates two collectors:
#     1. Fine-grade sparsity levels of the activations
#     2. L1-magnitude of each of the activation channels
#
#     Args:
#         model - the model on which we want to collect statistics
#         phases - the statistics collection phases: train, valid, and/or test
#
#     WARNING! Enabling activation statsitics collection will significantly slow down training!
#     """
#     distiller.utils.assign_layer_fq_names(model)
#
#     genCollectors = lambda: missingdict({  # noqa E731 pylint: disable=unnecessary-lambda-assignment
#         "sparsity":      SummaryActivationStatsCollector(model, "sparsity",
#                                                          lambda t:
#                                                          100 * distiller.utils.sparsity(t)),
#         "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
#                                                          distiller.utils.activation_channels_l1),
#         "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
#                                                          distiller.utils.activation_channels_apoz),
#         "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
#                                                          distiller.utils.
#                                                          activation_channels_means),
#         "records":       RecordsActivationStatsCollector(model, classes=[nn.Conv2d])
#     })
#
#     return {k: (genCollectors() if k in phases else missingdict())
#             for k in ('train', 'valid', 'test')}
#
#
# def save_collectors_data(collectors, directory):
#     """Utility function that saves all activation statistics to Excel workbooks
#     """
#     for name, collector in collectors.items():
#         workbook = os.path.join(directory, name)
#         msglogger.info("Generating %s", workbook)
#         collector.save(workbook)
#
#
# def test(test_loader, model, criterion, loggers, activations_collectors, args):
#     """Model Test"""
#     msglogger.info('--- test ---------------------')
#     if activations_collectors is None:
#         activations_collectors = create_activation_stats_collectors(model, None)
#     with collectors_context(activations_collectors["test"]) as collectors:
#         top1, top5, vloss, mAP = _validate(test_loader, model, criterion, loggers, args)
#         distiller.log_activation_statistics(-1, "test", loggers, collector=collectors['sparsity'])
#
#         if args.kernel_stats:
#             print("==> Kernel Stats")
#             with torch.no_grad():
#                 global weight_min, weight_max, weight_count  # pylint: disable=global-statement
#                 global weight_sum, weight_stddev, weight_mean  # pylint: disable=global-statement
#                 weight_min = torch.tensor(float('inf'), device=args.device)
#                 weight_max = torch.tensor(float('-inf'), device=args.device)
#                 weight_count = torch.tensor(0, dtype=torch.int, device=args.device)
#                 weight_sum = torch.tensor(0.0, device=args.device)
#                 weight_stddev = torch.tensor(0.0, device=args.device)
#
#                 def traverse_pass1(m):
#                     """
#                     Traverse model to build weight stats
#                     """
#                     global weight_min, weight_max  # pylint: disable=global-statement
#                     global weight_count, weight_sum  # pylint: disable=global-statement
#                     if isinstance(m, nn.Conv2d):
#                         weight_min = torch.min(torch.min(m.weight), weight_min)
#                         weight_max = torch.max(torch.max(m.weight), weight_max)
#                         weight_count += len(m.weight.flatten())
#                         weight_sum += m.weight.flatten().sum()
#                         if hasattr(m, 'bias') and m.bias is not None:
#                             weight_min = torch.min(torch.min(m.bias), weight_min)
#                             weight_max = torch.max(torch.max(m.bias), weight_max)
#                             weight_count += len(m.bias.flatten())
#                             weight_sum += m.bias.flatten().sum()
#
#                 def traverse_pass2(m):
#                     """
#                     Traverse model to build weight stats
#                     """
#                     global weight_stddev  # pylint: disable=global-statement
#                     if isinstance(m, nn.Conv2d):
#                         weight_stddev += ((m.weight.flatten() - weight_mean) ** 2).sum()
#                         if hasattr(m, 'bias') and m.bias is not None:
#                             weight_stddev += ((m.bias.flatten() - weight_mean) ** 2).sum()
#
#                 model.apply(traverse_pass1)
#
#                 weight_mean = weight_sum / weight_count
#
#                 model.apply(traverse_pass2)
#
#                 weight_stddev = torch.sqrt(weight_stddev / weight_count)
#
#                 print(f"Total 2D kernel weights: {weight_count} --> min: {weight_min}, "
#                       f"max: {weight_max}, stddev: {weight_stddev}")
#
#         save_collectors_data(collectors, msglogger.logdir)
#     return top1, top5, vloss, mAP
#
#
# OVERALL_LOSS_KEY = 'Overall Loss'
# OBJECTIVE_LOSS_KEY = 'Objective Loss'
#
#
# def earlyexit_validate_loss(output, target, _criterion, args):
#     """
#     We need to go through each sample in the batch itself - in other words, we are
#     not doing batch processing for exit criteria - we do this as though it were batchsize of 1
#     but with a grouping of samples equal to the batch size.
#     Note that final group might not be a full batch - so determine actual size.
#     """
#     this_batch_size = target.size()[0]
#     earlyexit_validate_criterion = nn.CrossEntropyLoss(reduce=False).to(args.device)
#
#     for exitnum in range(args.num_exits):
#         # calculate losses at each sample separately in the minibatch.
#         args.loss_exits[exitnum] = earlyexit_validate_criterion(  # pylint: disable=not-callable
#             output[exitnum], target
#         )
#         # for batch_size > 1, we need to reduce this down to an average over the batch
#         args.losses_exits[exitnum].add(torch.mean(args.loss_exits[exitnum]).cpu())
#
#     for batch_index in range(this_batch_size):
#         earlyexit_taken = False
#         # take the exit using CrossEntropyLoss as confidence measure (lower is more confident)
#         for exitnum in range(args.num_exits - 1):
#             if args.loss_exits[exitnum][batch_index] < args.earlyexit_thresholds[exitnum]:
#                 # take the results from early exit since lower than threshold
#                 args.exiterrors[exitnum].add(
#                     torch.tensor(
#                         np.array(output[exitnum].data[batch_index].cpu(), ndmin=2),
#                         dtype=torch.float
#                     ),
#                     torch.full([1], target[batch_index], dtype=torch.long))
#                 args.exit_taken[exitnum] += 1
#                 earlyexit_taken = True
#                 break  # since exit was taken, do not affect the stats of subsequent exits
#         # this sample does not exit early and therefore continues until final exit
#         if not earlyexit_taken:
#             exitnum = args.num_exits - 1
#             args.exiterrors[exitnum].add(
#                 torch.tensor(
#                     np.array(output[exitnum].data[batch_index].cpu(), ndmin=2),
#                     dtype=torch.float
#                 ),
#                 torch.full([1], target[batch_index], dtype=torch.long))
#             args.exit_taken[exitnum] += 1
#
#
# def earlyexit_validate_stats(args):
#     """Print some interesting summary stats for number of data points that could exit early"""
#     top1k_stats = [0] * args.num_exits
#     top5k_stats = [0] * args.num_exits
#     losses_exits_stats = [0] * args.num_exits
#     sum_exit_stats = 0
#     for exitnum in range(args.num_exits):
#         if args.exit_taken[exitnum]:
#             sum_exit_stats += args.exit_taken[exitnum]
#             msglogger.info("Exit %d: %d", exitnum, args.exit_taken[exitnum])
#             if not args.regression:
#                 top1k_stats[exitnum] += args.exiterrors[exitnum].value(1)
#                 top5k_stats[exitnum] += args.exiterrors[exitnum].value(
#                     min(args.num_classes, 5)
#                 )
#             else:
#                 top1k_stats[exitnum] += args.exiterrors[exitnum].value()
#             losses_exits_stats[exitnum] += args.losses_exits[exitnum].mean
#     for exitnum in range(args.num_exits):
#         if args.exit_taken[exitnum]:
#             msglogger.info("Percent Early Exit %d: %.3f", exitnum,
#                            (args.exit_taken[exitnum]*100.0) / sum_exit_stats)
#     total_top1 = 0
#     total_top5 = 0
#     for exitnum in range(args.num_exits):
#         total_top1 += (top1k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
#         if not args.regression:
#             total_top5 += (top5k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
#             msglogger.info("Accuracy Stats for exit %d: top1 = %.3f, top5 = %.3f", exitnum,
#                            top1k_stats[exitnum], top5k_stats[exitnum])
#         else:
#             msglogger.info("Accuracy Stats for exit %d: top1 = %.3f", exitnum,
#                            top1k_stats[exitnum])
#     msglogger.info("Totals for entire network with early exits: top1 = %.3f, top5 = %.3f",
#                    total_top1, total_top5)
#     return total_top1, total_top5, losses_exits_stats
#
#
# def _validate(data_loader, model, criterion, loggers, args, epoch=-1, tflogger=None):
#     """Execute the validation/test loop."""
#     losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
#                           (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
#
#     if args.obj_detection:
#         map_calculator = MeanAveragePrecision(
#             # box_format='xyxy',  # Enable in torchmetrics > 0.6
#             # iou_type='bbox',  # Enable in torchmetrics > 0.6
#             class_metrics=False,
#             # iou_thresholds=[0.5],  # Enable in torchmetrics > 0.6
#         ).to(args.device)
#         mAP = 0.00
#     if not args.regression:
#         classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, min(args.num_classes, 5)))
#     else:
#         classerr = tnt.MSEMeter()
#
#     def save_tensor(t, f, regression=True):
#         """ Save tensor `t` to file handle `f` in CSV format """
#         if t.dim() > 1:
#             if not regression:
#                 t = nn.functional.softmax(t, dim=1)
#             np.savetxt(f, t.reshape(t.shape[0], t.shape[1], -1).cpu().numpy().mean(axis=2),
#                        delimiter=",")
#         else:
#             if regression:
#                 np.savetxt(f, t.cpu().numpy(), delimiter=",")
#             else:
#                 for i in t:
#                     f.write(f'{args.labels[i.int()]}\n')
#
#     if args.csv_prefix is not None:
#         with open(f'{args.csv_prefix}_ytrue.csv', mode='w', encoding='utf-8') as f_ytrue:
#             f_ytrue.write('truth\n')
#         with open(f'{args.csv_prefix}_ypred.csv', mode='w', encoding='utf-8') as f_ypred:
#             f_ypred.write(','.join(args.labels) + '\n')
#         with open(f'{args.csv_prefix}_x.csv', mode='w', encoding='utf-8') as f_x:
#             for i in range(args.dimensions[0]-1):
#                 f_x.write(f'x_{i}_mean,')
#             f_x.write(f'x_{args.dimensions[0]-1}_mean\n')
#
#     if args.earlyexit_thresholds:
#         # for Early Exit, we have a list of errors and losses for each of the exits.
#         args.exiterrors = []
#         args.losses_exits = []
#         for exitnum in range(args.num_exits):
#             if not args.regression:
#                 args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True,
#                                                            topk=(1, min(args.num_classes, 5))))
#             else:
#                 args.exiterrors.append(tnt.MSEMeter())
#             args.losses_exits.append(tnt.AverageValueMeter())
#         args.exit_taken = [0] * args.num_exits
#
#     batch_time = tnt.AverageValueMeter()
#     total_samples = len(data_loader.sampler)
#     batch_size = data_loader.batch_size
#     if args.display_confusion:
#         confusion = tnt.ConfusionMeter(args.num_classes)
#     total_steps = (total_samples + batch_size - 1) // batch_size
#     msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)
#
#     # Switch to evaluation mode
#     model.eval()
#
#     end = time.time()
#     class_probs = []
#     class_preds = []
#     sample_saved = False  # Track if --save-sample has been done for this validation step
#
#     # Get object detection params
#     obj_detection_params = parse_obj_detection_yaml.parse(args.obj_detection_params) \
#         if args.obj_detection_params else None
#
#     mAP = 0.0
#     have_mAP = False
#     with torch.no_grad():
#         m = model.module if isinstance(model, nn.DataParallel) else model
#
#         for validation_step, (inputs, target) in enumerate(data_loader):
#             if args.obj_detection:
#                 if not object_detection_utils.check_target_exists(target):
#                     print(f'No target in batch. Ep: {epoch}, validation_step: {validation_step}')
#                     continue
#
#                 boxes_list = [elem[0].to(args.device) for elem in target]
#                 labels_list = [elem[1].to(args.device) for elem in target]
#                 filtered_all_images_boxes = None
#
#                 # Adjust ground truth index as mAP calculator uses 0-indexed class labels
#                 labels_list_for_map = [elem[1].to(args.device) - 1 for elem in target]
#
#                 inputs = inputs.to(args.device)
#
#                 target = (boxes_list, labels_list)
#
#                 # compute output from model
#                 output_boxes, output_conf = model(inputs)
#
#                 # correct output for accurate loss calculation
#                 if args.act_mode_8bit:
#                     output_boxes /= 128.
#                     output_conf /= 128.
#
#                     if (hasattr(m, 'are_locations_wide') and m.are_locations_wide):
#                         output_boxes /= 128.
#
#                     if (hasattr(m, 'are_scores_wide') and m.are_scores_wide):
#                         output_conf /= 128.
#
#                 output = (output_boxes, output_conf)
#
#                 if boxes_list:
#                     # .module is added to model for access in multi GPU environments
#                     # as https://github.com/pytorch/pytorch/issues/16885 has not been merged yet
#                     det_boxes_batch, det_labels_batch, det_scores_batch = \
#                         m.detect_objects(output_boxes, output_conf,
#                                          min_score=obj_detection_params['nms']['min_score'],
#                                          max_overlap=obj_detection_params['nms']['max_overlap'],
#                                          top_k=obj_detection_params['nms']['top_k'])
#
#                     # Filter images with only background box
#                     filtered_list = list(
#                         filter(lambda elem: not (len(elem[1]) == 1 and elem[1][0] == 0),
#                                zip(det_boxes_batch, det_labels_batch, det_scores_batch))
#                     )
#
#                     # Update mAP Calculator
#                     if filtered_list:
#                         filtered_all_images_boxes, filtered_all_images_labels, \
#                             filtered_all_images_scores = zip(*filtered_list)
#
#                         # mAP calculator uses 0-indexed class labels
#                         filtered_all_images_labels = [e - 1 for e in filtered_all_images_labels]
#
#                         # Prepare truths
#                         boxes = torch.cat(boxes_list)
#                         labels = torch.cat(labels_list_for_map)
#
#                         gt = [{'boxes': boxes, 'labels': labels}]
#
#                         # Prepare predictions
#                         pred_boxes = torch.cat(filtered_all_images_boxes)
#                         pred_scores = torch.cat(filtered_all_images_scores)
#                         pred_labels = torch.cat(filtered_all_images_labels)
#
#                         preds = [
#                             {'boxes': pred_boxes, 'scores': pred_scores, 'labels': pred_labels}
#                         ]
#
#                         # Update mAP calculator
#                         map_calculator.update(preds=preds, target=gt)
#                         have_mAP = True
#             else:
#                 inputs, target = inputs.to(args.device), target.to(args.device)
#                 # compute output from model
#                 if args.kd_relationbased:
#                     output = args.kd_policy.forward(inputs)
#                 else:
#                     output = model(inputs)
#                 if args.out_fold_ratio != 1:
#                     output = ai8x.unfold_batch(output, args.out_fold_ratio)
#
#                 # correct output for accurate loss calculation
#                 if args.act_mode_8bit:
#                     output /= 128.
#                     for key in model.__dict__['_modules'].keys():
#                         if (hasattr(model.__dict__['_modules'][key], 'wide')
#                                 and model.__dict__['_modules'][key].wide):
#                             output /= 256.
#                     if args.regression:
#                         target /= 128.
#
#             if args.generate_sample is not None and args.act_mode_8bit and not sample_saved:
#                 sample.generate(args.generate_sample, inputs, target, output,
#                                 args.dataset, False, args.slice_sample)
#                 sample_saved = True
#
#             if args.csv_prefix is not None:
#                 save_tensor(inputs, f_x)
#                 save_tensor(output, f_ypred, regression=args.regression)
#                 save_tensor(target, f_ytrue, regression=args.regression)
#
#             if not args.earlyexit_thresholds:
#                 # compute loss
#                 loss = criterion(output, target)
#                 if args.kd_relationbased:
#                     agg_loss = args.kd_policy.before_backward_pass(None, None, None, None,
#                                                                    loss, None)
#                     losses[OVERALL_LOSS_KEY].add(agg_loss.overall_loss.item())
#                 # measure accuracy and record loss
#                 losses[OBJECTIVE_LOSS_KEY].add(loss.item())
#
#                 if not args.obj_detection and not args.kd_relationbased:
#                     if len(output.data.shape) <= 2 or args.regression:
#                         classerr.add(output.data, target)
#                     else:
#                         classerr.add(output.data.permute(0, 2, 3, 1).flatten(start_dim=0,
#                                                                              end_dim=2),
#                                      target.flatten())
#                     if args.display_confusion:
#                         confusion.add(output.data, target)
#             else:
#                 earlyexit_validate_loss(output, target, criterion, args)
#
#             # measure elapsed time
#             batch_time.add(time.time() - end)
#             end = time.time()
#
#             steps_completed = validation_step + 1
#             if steps_completed % args.print_freq == 0 or steps_completed == total_steps:
#                 if args.display_prcurves and tflogger is not None:
#                     # TODO PR Curve generation for Object Detection case is NOT implemented yet
#                     class_probs_batch = [nn.functional.softmax(el, dim=0) for el in output]
#                     _, class_preds_batch = torch.max(output, 1)
#                     class_probs.append(class_probs_batch)
#                     class_preds.append(class_preds_batch)
#
#                 if not args.earlyexit_thresholds:
#                     if args.kd_relationbased:
#                         stats = (
#                             '',
#                             OrderedDict([('Loss', losses[OBJECTIVE_LOSS_KEY].mean),
#                                          ('Overall Loss', losses[OVERALL_LOSS_KEY].mean)])
#                         )
#                     elif args.obj_detection:
#                         # Only run compute() if there is at least one new update()
#                         if have_mAP:
#                             # Remove [0] in new torchmetrics
#                             mAP = map_calculator.compute()['map_50'][0]
#                             have_mAP = False
#                         stats = (
#                             '',
#                             OrderedDict([('Loss', losses[OBJECTIVE_LOSS_KEY].mean),
#                                          ('mAP', mAP)])
#                         )
#                     else:
#                         if not args.regression:
#                             stats = (
#                                 '',
#                                 OrderedDict([('Loss', losses[OBJECTIVE_LOSS_KEY].mean),
#                                             ('Top1', classerr.value(1))])
#                             )
#                             if args.num_classes > 5:
#                                 stats[1]['Top5'] = classerr.value(5)
#                         else:
#                             stats = (
#                                 '',
#                                 OrderedDict([('Loss', losses[OBJECTIVE_LOSS_KEY].mean),
#                                             ('MSE', classerr.value())])
#                             )
#                 else:
#                     stats_dict = OrderedDict()
#                     stats_dict['Test'] = validation_step
#                     for exitnum in range(args.num_exits):
#                         la_string = 'LossAvg' + str(exitnum)
#                         stats_dict[la_string] = args.losses_exits[exitnum].mean
#                         # Because of the nature of ClassErrorMeter, if an exit is never taken
#                         # during the batch, then accessing the value(k) will cause a divide by
#                         # zero. So we'll build the OrderedDict accordingly and we will not print
#                         # for an exit error when that exit is never taken.
#                         if args.exit_taken[exitnum]:
#                             if not args.regression:
#                                 t1 = 'Top1_exit' + str(exitnum)
#                                 stats_dict[t1] = args.exiterrors[exitnum].value(1)
#                                 if args.num_classes > 5:
#                                     t5 = 'Top5_exit' + str(exitnum)
#                                     stats_dict[t5] = args.exiterrors[exitnum].value(5)
#                             else:
#                                 t1 = 'MSE_exit' + str(exitnum)
#                                 stats_dict[t1] = args.exiterrors[exitnum].value()
#                     stats = ('Performance/Validation/', stats_dict)
#
#                 distiller.log_training_progress(stats, None, epoch, steps_completed,
#                                                 total_steps, args.print_freq, loggers)
#
#                 if args.display_prcurves and tflogger is not None:
#                     test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
#                     test_preds = torch.cat(class_preds)
#                     for i in range(args.num_classes):
#                         tb_preds = test_preds == i
#                         tb_probs = test_probs[:, i]
#                         tflogger.tblogger.writer.add_pr_curve(str(args.labels[i]), tb_preds,
#                                                               tb_probs, global_step=epoch)
#
#                 if args.display_embedding and tflogger is not None \
#                    and steps_completed == total_steps:
#                     def select_n_random(data, labels, features, n=100):
#                         """Selects n random datapoints, their corresponding labels and features"""
#                         assert len(data) == len(labels) == len(features)
#
#                         perm = torch.randperm(len(data))
#                         return data[perm][:n], labels[perm][:n], features[perm][:n]
#
#                     # Select up to 100 random images and their target indices
#                     images, labels, features = select_n_random(inputs, target, output,
#                                                                n=min(100, len(inputs)))
#
#                     # Get the class labels for each image
#                     class_labels = [args.labels[lab] for lab in labels]
#
#                     tflogger.tblogger.writer.add_embedding(
#                         features,
#                         metadata=class_labels,
#                         label_img=args.visualize_fn(images, args),
#                         global_step=epoch,
#                         tag='verification/embedding'
#                     )
#
#     if args.csv_prefix is not None:
#         f_ytrue.close()
#         f_ypred.close()
#         f_x.close()
#
#     if not args.earlyexit_thresholds:
#
#         if args.kd_relationbased:
#
#             msglogger.info('==> Overall Loss: %.3f\n',
#                            losses[OVERALL_LOSS_KEY].mean)
#
#             return 0, 0, losses[OVERALL_LOSS_KEY].mean, 0
#
#         if args.obj_detection:
#
#             msglogger.info('==> mAP: %.5f    Loss: %.3f\n',
#                            mAP,
#                            losses[OBJECTIVE_LOSS_KEY].mean)
#
#             return 0, 0, losses[OBJECTIVE_LOSS_KEY].mean, mAP
#
#         if not args.regression:
#             if args.num_classes > 5:
#                 msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
#                                classerr.value()[0], classerr.value()[1],
#                                losses[OBJECTIVE_LOSS_KEY].mean)
#             else:
#                 msglogger.info('==> Top1: %.3f    Loss: %.3f\n',
#                                classerr.value()[0], losses[OBJECTIVE_LOSS_KEY].mean)
#         else:
#             msglogger.info('==> MSE: %.5f    Loss: %.3f\n',
#                            classerr.value(), losses[OBJECTIVE_LOSS_KEY].mean)
#             return classerr.value(), .0, losses[OBJECTIVE_LOSS_KEY].mean, 0
#
#         if args.display_confusion:
#             msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
#             if tflogger is not None:
#                 cf = nnplot.confusion_matrix(confusion.value(), args.labels)
#                 tflogger.tblogger.writer.add_image('Validation/ConfusionMatrix', cf, epoch,
#                                                    dataformats='HWC')
#         if not args.regression:
#             return classerr.value(1), classerr.value(min(args.num_classes, 5)), \
#                 losses[OBJECTIVE_LOSS_KEY].mean, 0
#     # else:
#     total_top1, total_top5, losses_exits_stats = earlyexit_validate_stats(args)
#     return total_top1, total_top5, losses_exits_stats[args.num_exits-1], 0

# ----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    model, criterion, test_loader, pylogger, args = load_model()

    predicted_array, target_array = custom_evaluate(model, criterion, test_loader, pylogger, args)

    predicted_array = np.hstack(np.array(predicted_array))
    target_array = np.hstack(np.array(target_array))

    am_before = AccuracyMetrics(target_array, predicted_array, 3.97, 2, threshold="max_f_score")

    # test_array = predicted_array[0::4]

    num_channels = 4
    predicted_array = predicted_array.reshape(predicted_array.shape[0] // num_channels, num_channels)
    target_array = target_array.reshape(target_array.shape[0] // num_channels, num_channels)

    predicted_array = predicted_array.mean(axis=1)
    target_array = target_array.mean(axis=1)

    predicted_array = np.where(predicted_array >= 0.5, 1, 0)

    am_after = AccuracyMetrics(target_array, predicted_array, 3.97, 2, threshold="max_f_score")

    am_before = am_before.as_dict()
    am_after = am_after.as_dict()

    with open(os.path.join(msglogger.logdir, 'acc_metrics_separate_ch.csv'), 'w', newline='') as csvfile:
        fieldnames = list(am_before.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(am_before)

    with open(os.path.join(msglogger.logdir, 'acc_metrics_concat_ch.csv'), 'w', newline='') as csvfile:
        fieldnames = list(am_after.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(am_after)

    try:
        pass
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: %s', os.path.realpath(msglogger.log_filename))
