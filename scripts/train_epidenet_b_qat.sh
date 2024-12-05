#!/bin/sh
python train.py --device MAX78000 \
  --model epidenet_b --dataset chbmit_singlech_768samples_patient_10_leave_out_seizure_6 \
  --epochs 1000 --batch-size 256 --lr 1e-4 --deterministic --optimizer adam \
  --confusion --pr-curves --param-hist \
  --qat-policy policies/qat_policy_epidenet_b.yaml \
  --compress policies/schedule_epidenet_b_multisteplr.yaml \
  --show-train-accuracy full --validation-split 0.2 --enable-tensorboard \
  --resume-from logs/cross_validation_others/patient10/2024.10.28-141856-epidenet_b_v3.2_768samples_patient10_leaveout6_lr1e-4_adam_train/best.pth.tar \
  --track-vloss --custom-shuffle-split "$@"
exec $SHELL
