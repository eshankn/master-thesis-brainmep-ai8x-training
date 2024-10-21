
import os
import csv
import json
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

from train import (create_model, update_old_model_params, create_optimizer, OVERALL_LOSS_KEY, OBJECTIVE_LOSS_KEY)

from brainmepnas import AccuracyMetrics

# Logger handle
msglogger = None

CROSS_VAL_PATH = './logs/cross_validation/epidenet_v3.2_768samples_qat1'


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

    if not args.act_mode_8bit:
        checkpoint_filename = 'best.pth.tar'
    else:
        checkpoint_filename = 'qat_best-q.pth.tar'

    for fold_idx, folder in enumerate(sorted(os.listdir(CROSS_VAL_PATH))):
        if len(folder.split('_')) < 5:
            continue

        samples = folder.split('_')[3]
        leaveout = folder.split('_')[5][-1]

        args.dataset = 'chbmit_singlech_' + samples + '_patient_5_leave_out_seizure_' + leaveout
        args.load_model_path = os.path.join(CROSS_VAL_PATH, folder, checkpoint_filename)

        selected_source = next((item for item in supported_sources if item['name'] == args.dataset))
        args.labels = selected_source['output']
        args.num_classes = len(args.labels)

        msglogger.info('')
        msglogger.info('------------------------------------------------------------------------------------------')
        msglogger.info('----------------------------------------------------------------------')
        msglogger.info('Fold: %d', fold_idx)
        msglogger.info('Dataset: %s', args.dataset)
        msglogger.info('----------------------------------------------------------------------')
        msglogger.info('')

        dimensions = selected_source['input']
        if len(dimensions) == 2:
            dimensions += (1, )
        args.dimensions = dimensions

        args.datasets_fn = selected_source['loader']
        args.collate_fn = selected_source.get('collate')  # .get returns None if key does not exist

        model = create_model(supported_models, dimensions, args)

        pylogger = PythonLogger(msglogger, log_1d=True)
        pylogger = [pylogger]

        # qat_policy = parse_qat_yaml.parse(args.qat_policy) \
        #     if args.qat_policy.lower() != "none" else None
        #
        # if args.load_model_path:
        #     update_old_model_params(args.load_model_path, model)
        #     if qat_policy is not None:
        #         checkpoint = torch.load(args.load_model_path,
        #                                 map_location=lambda storage, loc: storage)
        #         # pylint: disable=unsubscriptable-object
        #         if checkpoint.get('epoch', None) >= qat_policy['start_epoch']:
        #             ai8x.fuse_bn_layers(model)
        #             if args.name:
        #                 args.name = f'{args.name}_qat'
        #             else:
        #                 args.name = 'qat'
        #         # pylint: enable=unsubscriptable-object
        #     model = apputils.load_lean_checkpoint(model, args.load_model_path,
        #                                           model_device=args.device)
        #     ai8x.update_model(model)

        if args.load_model_path:
            update_old_model_params(args.load_model_path, model)
            if args.act_mode_8bit:
                # pylint: disable=unsubscriptable-object
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

        train_loader, val_loader, test_loader, _ = apputils.get_data_loaders(
            args.datasets_fn, (os.path.expanduser(args.data), args), args.batch_size,
            args.workers, args.validation_split, args.deterministic,
            args.effective_train_size, args.effective_valid_size, args.effective_test_size,
            test_only=args.evaluate, collate_fn=args.collate_fn, cpu=args.device == 'cpu', rand_data=False,
            custom_shuffle_split=args.custom_shuffle_split)
        assert args.evaluate or train_loader is not None and val_loader is not None, \
            "No training and/or validation data in train mode"
        assert not args.evaluate or test_loader is not None, "No test data in eval mode"

        msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                       len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

        model.to(args.device)

        yield model, criterion, train_loader, test_loader, pylogger, args


def custom_evaluate(model, criterion, data_loader, loggers, args, epoch=-1):

    msglogger.info('--- test ---------------------')

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
            # predicted_labels = np.argmax(predicted_labels, 1)

            predicted_labels = (predicted_labels + 1) / 2
            predicted_labels = predicted_labels[:, 1]

            true_labels = target.cpu().numpy()

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


if __name__ == '__main__':

    cross_val_metrics_dict = {"sample_accuracy": OrderedDict(), "sample_precision": OrderedDict(),
                              "sample_sensitivity": OrderedDict(), "sample_specificity": OrderedDict(),
                              "sample_f_score": OrderedDict(), "sample_balanced_accuracy": OrderedDict(),
                              "sample_roc_auc": OrderedDict(), "sample_prc_auc": OrderedDict(),
                              "event_sensitivity": OrderedDict(), "event_precision": OrderedDict(),
                              "event_f_score": OrderedDict(), "event_false_detections_per_hour": OrderedDict()}

    sample_accuracy_arr = []
    sample_precision_arr = []
    sample_sensitivity_arr = []
    sample_specificity_arr = []
    sample_f_score_arr = []
    sample_balanced_accuracy_arr = []
    sample_roc_auc_arr = []
    sample_prc_auc_arr = []

    event_sensitivity_arr = []
    event_precision_arr = []
    event_f_score_arr = []
    event_false_detections_per_hour_arr = []

    for fold, (model, criterion, train_loader, test_loader, pylogger, args) in enumerate(load_model()):
        train_predicted_arr, train_target_arr = custom_evaluate(model, criterion, train_loader, pylogger, args)
        test_predicted_arr, test_target_arr = custom_evaluate(model, criterion, test_loader, pylogger, args)

        train_predicted_arr = np.hstack(np.array(train_predicted_arr))
        train_target_arr = np.hstack(np.array(train_target_arr))
        test_predicted_arr = np.hstack(np.array(test_predicted_arr))
        test_target_arr = np.hstack(np.array(test_target_arr))

        am_train = AccuracyMetrics(train_target_arr, train_predicted_arr, 3.97, 2, threshold="max_sample_f_score")
        threshold = am_train.threshold

        am_test_sep = AccuracyMetrics(test_target_arr, test_predicted_arr, 3.97, 2, threshold=threshold)

        num_channels = 4
        test_predicted_arr = test_predicted_arr.reshape(test_predicted_arr.shape[0] // num_channels, num_channels)
        test_target_arr = test_target_arr.reshape(test_target_arr.shape[0] // num_channels, num_channels)

        test_predicted_arr = test_predicted_arr.mean(axis=1)
        test_target_arr = test_target_arr.mean(axis=1)

        # test_predicted_arr = np.where(test_predicted_arr >= 0.5, 1, 0)

        am_test_concat = AccuracyMetrics(test_target_arr, test_predicted_arr, 3.97, 2, threshold=threshold)

        sample_accuracy_arr.append(am_test_concat.sample_accuracy)
        sample_precision_arr.append(am_test_concat.sample_precision)
        sample_sensitivity_arr.append(am_test_concat.sample_sensitivity)
        sample_specificity_arr.append(am_test_concat.sample_specificity)
        sample_f_score_arr.append(am_test_concat.sample_f_score)
        sample_balanced_accuracy_arr.append(am_test_concat.sample_balanced_accuracy)
        sample_roc_auc_arr.append(am_test_concat.sample_roc_auc)
        sample_prc_auc_arr.append(am_test_concat.sample_prc_auc)

        event_sensitivity_arr.append(am_test_concat.event_sensitivity)
        event_precision_arr.append(am_test_concat.event_precision)
        event_f_score_arr.append(am_test_concat.event_f_score)
        event_false_detections_per_hour_arr.append(am_test_concat.event_false_detections_per_hour)

        cross_val_metrics_dict['sample_accuracy'][args.dataset] = am_test_concat.sample_accuracy
        cross_val_metrics_dict['sample_precision'][args.dataset] = am_test_concat.sample_precision
        cross_val_metrics_dict['sample_sensitivity'][args.dataset] = am_test_concat.sample_sensitivity
        cross_val_metrics_dict['sample_specificity'][args.dataset] = am_test_concat.sample_specificity
        cross_val_metrics_dict['sample_f_score'][args.dataset] = am_test_concat.sample_f_score
        cross_val_metrics_dict['sample_balanced_accuracy'][args.dataset] = am_test_concat.sample_balanced_accuracy
        cross_val_metrics_dict['sample_roc_auc'][args.dataset] = am_test_concat.sample_roc_auc
        cross_val_metrics_dict['sample_prc_auc'][args.dataset] = am_test_concat.sample_prc_auc

        cross_val_metrics_dict['event_sensitivity'][args.dataset] = am_test_concat.event_sensitivity
        cross_val_metrics_dict['event_precision'][args.dataset] = am_test_concat.event_precision
        cross_val_metrics_dict['event_f_score'][args.dataset] = am_test_concat.event_f_score
        cross_val_metrics_dict['event_false_detections_per_hour'][args.dataset] = (
            am_test_concat.event_false_detections_per_hour)

        am_test_concat = am_test_concat.as_dict()

        with open(os.path.join(msglogger.logdir, os.path.basename(msglogger.logdir) + '-fold_' + str(fold) + '.csv'),
                  'w', newline='') as csvfile:
            fieldnames = list(am_test_concat.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(am_test_concat)

    cross_val_metrics_dict['sample_accuracy']['sample_accuracy_mean'] = np.array(sample_accuracy_arr).mean()
    cross_val_metrics_dict['sample_precision']['sample_precision_mean'] = np.array(sample_precision_arr).mean()
    cross_val_metrics_dict['sample_sensitivity']['sample_sensitivity_mean'] = np.array(sample_sensitivity_arr).mean()
    cross_val_metrics_dict['sample_specificity']['sample_specificity_mean'] = np.array(sample_specificity_arr).mean()
    cross_val_metrics_dict['sample_f_score']['sample_f_score_mean'] = np.array(sample_f_score_arr).mean()
    cross_val_metrics_dict['sample_balanced_accuracy']['sample_balanced_accuracy_mean'] = (
        np.array(sample_balanced_accuracy_arr).mean())
    cross_val_metrics_dict['sample_roc_auc']['sample_roc_auc_mean'] = np.array(sample_roc_auc_arr).mean()
    cross_val_metrics_dict['sample_prc_auc']['sample_prc_auc_mean'] = np.array(sample_prc_auc_arr).mean()

    cross_val_metrics_dict['event_sensitivity']['event_sensitivity_mean'] = np.array(event_sensitivity_arr).mean()
    cross_val_metrics_dict['event_precision']['event_precision_mean'] = np.array(event_precision_arr).mean()
    cross_val_metrics_dict['event_f_score']['event_f_score_mean'] = np.array(event_f_score_arr).mean()
    cross_val_metrics_dict['event_false_detections_per_hour']['event_false_detections_per_hour_mean'] = (
        np.array(event_false_detections_per_hour_arr).mean())

    try:
        pass
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    finally:
        if msglogger is not None:
            msglogger.info('------------------------------------------------------------------------------------------')
            msglogger.info('')
            msglogger.info('Log file for this run: %s', os.path.realpath(msglogger.log_filename))

            with open(os.path.join(msglogger.logdir, "cross_val_metrics.json"), "w") as file:
                json.dump(cross_val_metrics_dict, file, indent=4)
