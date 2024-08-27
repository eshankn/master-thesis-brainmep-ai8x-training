#!/bin/sh
python train.py --evaluate --model epidenet_b \
  --dataset chbmit_singlech_1016samples_patient_5_leave_out_seizure_1 \
  --exp-load-weights-from ../ai8x-synthesis/trained_custom/checkpoints/energy_profiling/input_data_size/ep_v1.22_1016_epidenet_b_v3.2_qat_best-q.pth.tar \
  --save-sample 9 \
  --confusion -8 --device MAX78000 "$@"
exec $SHELL
