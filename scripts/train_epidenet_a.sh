#!/bin/sh
python train.py --device MAX78000 \
  --model epidenet_a --dataset chbmit_patient_5_leave_out_seizure_1 \
  --epochs 100 --batch-size 256 --lr 0.001 --deterministic \
  --confusion --pr-curves