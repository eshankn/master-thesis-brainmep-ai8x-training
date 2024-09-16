#!/bin/sh
python train.py --evaluate --model energy_profiling \
  --dataset chbmit_singlech_1021samples_patient_5_leave_out_seizure_1 \
  --exp-load-weights-from ../ai8x-synthesis/trained_custom/checkpoints/energy_profiling/input_data_size/ep_demo_1021_qat_best-q.pth.tar \
  --save-sample 9 \
  --confusion -8 --device MAX78000 "$@"
exec $SHELL
