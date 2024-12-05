#!/bin/sh
python train.py --evaluate --model epidenet_b \
  --dataset chbmit_singlech_768samples_patient_5_leave_out_seizure_1 \
  --exp-load-weights-from ./logs/cross_validation/epidenet_v3.1_768samples/2024.10.21-204649-epidenet_b_v3.1_768samples_patient5_leaveout1_lr1e-4_adam_multisteplr150_qat_train/qat_best-q.pth.tar \
  --save-sample 9 \
  --confusion -8 --device MAX78000 "$@"
exec $SHELL
