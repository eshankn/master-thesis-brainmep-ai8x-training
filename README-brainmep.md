# BrainMep - AI8x
This is a fork of the ai8x-training repository for the purposes of the BrainMep Project.

## Requirements
```
pip install git+https://github.com/jonathanlarochelle/brainmep-nas@features/dataset
pip install mne
```

## New datasets
All new dataset scripts and objects are placed in the datasets folder.

### CHB-MIT
Raw data is processed using process_chbmit.py:
```
python datasets/process_chbmit.py
```
The processed data will then be placed in data/chbmit.
Different datasets are created from the CHB-MIT data in datasets/chbmit.py.
For each patient are available S datasets (for S seizures), where one seizure is left out in each dataset.
For example, to train with data from patient 5 and leaving out seizure 1 for testing, set the argument as 
```
python train.py --dataset ChbMit_patient_5_leave_out_seizure_1 [...]
```

## New models
[...]
