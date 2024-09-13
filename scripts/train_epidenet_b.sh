#!/bin/sh
python train.py --device MAX78000 \
  --model epidenet_b --dataset chbmit_singlech_1016samples_patient_5_leave_out_seizure_1 \
  --epochs 1000 --batch-size 256 --lr 1e-4 --deterministic --optimizer adam \
  --confusion --pr-curves --param-hist \
  --qat-policy policies/qat_policy_epidenet_b.yaml \
  --show-train-accuracy full \
  --track-vloss "$@"
exec $SHELL
