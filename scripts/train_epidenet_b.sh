#!/bin/sh
python train.py --device MAX78000 \
  --model epidenet_b --dataset chbmit_singlech_1016samples_patient_5_leave_out_seizure_1 \
  --epochs 15 --batch-size 256 --lr 0.001 --deterministic \
  --confusion --pr-curves "$@"
exec $SHELL
