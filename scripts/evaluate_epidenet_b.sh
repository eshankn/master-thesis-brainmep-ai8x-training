#!/bin/sh
python train.py --evaluate --model epidenet_b --dataset chbmit_singlech_768samples_patient_5_leave_out_seizure_1 \
  --exp-load-weights-from ../ai8x-synthesis/trained_custom/checkpoints/epidenet_b_v3.2_768_qat_best-q.pth.tar \
  --save-sample 9 \
  --confusion -8 --device MAX78000 "$@"
exec $SHELL
