# ------------------------------------------------------------------------------------------
# Author        : Eshank Jayant Nazare
# File          : chbmit_singlech_b.py
# Project       : BrainMEP
# Modified      : 27.11.2024
# Description   : Dataset function definitions for training on patient 5 in the CHB-MIT
#                 dataset
# ------------------------------------------------------------------------------------------

# -*- coding: utf-8 -*-

# import built-in modules
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as transforms
import numpy as np

# import custom modules
from brainmepnas import Dataset as BrainMepNasDataset

# import sys
# sys.path.insert(0, '../')
import ai8x


class ChbMitSingleChannelPatientSpecific(TorchDataset):

    def __init__(self, root_dir, target_folder, d_type=None, transform=None,
                 patient: str = "2", leave_out_seizure: int = 1):
        """
        Parameters
        ----------
        root_dir : str
            Path to data directory. CHB-MIT should be in
            [root_dir]/chbmit_singlech
        d_type : str
            "train" or "test"
        transform : torchvision.transforms
            Transforms to apply to data.
        patient : str
            Target patient.
        leave_out_seizure: int
            Seizure to leave out for test set. If d_type="train", all seizures
            from the patient except leave_out_seizure are included. If
            d_type="test", only leave_out_seizure is included.
        """
        # Load and concatenate all data
        dataset = BrainMepNasDataset(root_dir + target_folder)
        if d_type == "train":
            nb_records = dataset.nb_records_per_patient[patient]
            patient_records = [i for i in range(nb_records)
                               if i != leave_out_seizure]
            x, y = dataset.get_data({patient: patient_records},
                                    set="train", shuffle=True,
                                    shuffle_seed=42)
        elif d_type == "test":
            x, y = dataset.get_data({patient: [leave_out_seizure]},
                                    set="test", shuffle=False)
        else:
            raise ValueError("d_type must be either 'train' or 'test'")

        x = x.astype(np.float32)
        #x = np.swapaxes(x, 1, 2)
        y = y.astype(np.int_)

        # Normalize data to [0, 1]
        # TODO: This is not robust scaling, but it works for now.
        x_max = x.max()
        x_min = x.min()
        x_normalized = (x - x_min) / (x_max - x_min)
        # x_transformed = transform(x_normalized)
        self.transform = transform
        self.x = x_normalized
        self.y = y
        self._len = len(x)

    def __getitem__(self, index):
        element = self.x[index]

        return self.transform(element), int(self.y[index])

    def __len__(self):
        return self._len


def ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train=True, load_test=True,
                                                    dataset_folder: str = "/chbmit_singlech_1024samples",
                                                    patient: str = "5", leave_out_seizure: int = 1):
    (data_dir, args) = data
    transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args),
        ])

    if load_train:
        train_dataset = ChbMitSingleChannelPatientSpecific(root_dir=data_dir, target_folder=dataset_folder,
                                                           d_type="train", patient=patient,
                                                           transform=transform, leave_out_seizure=leave_out_seizure)
    else:
        train_dataset = None

    if load_test:
        test_dataset = ChbMitSingleChannelPatientSpecific(root_dir=data_dir, target_folder=dataset_folder,
                                                          d_type="test", patient=patient,
                                                          transform=transform, leave_out_seizure=leave_out_seizure)
    else:
        test_dataset = None

    return train_dataset, test_dataset


def chbmit_singlech_16samples_patient_5_leave_out_seizure_1_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_16samples",
                                                           patient="5", leave_out_seizure=1)


def chbmit_singlech_32samples_patient_5_leave_out_seizure_1_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_32samples",
                                                           patient="5", leave_out_seizure=1)


def chbmit_singlech_64samples_patient_5_leave_out_seizure_1_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_64samples",
                                                           patient="5", leave_out_seizure=1)


def chbmit_singlech_128samples_patient_5_leave_out_seizure_1_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_128samples",
                                                           patient="5", leave_out_seizure=1)


def chbmit_singlech_256samples_patient_5_leave_out_seizure_1_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_256samples",
                                                           patient="5", leave_out_seizure=1)


# --------------------------------------------------------------------------------------------------------------
#                                               512 samples
# --------------------------------------------------------------------------------------------------------------

def chbmit_singlech_512samples_patient_5_leave_out_seizure_0_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_512samples",
                                                           patient="5", leave_out_seizure=0)


def chbmit_singlech_512samples_patient_5_leave_out_seizure_1_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_512samples",
                                                           patient="5", leave_out_seizure=1)


def chbmit_singlech_512samples_patient_5_leave_out_seizure_2_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_512samples",
                                                           patient="5", leave_out_seizure=2)


def chbmit_singlech_512samples_patient_5_leave_out_seizure_3_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_512samples",
                                                           patient="5", leave_out_seizure=3)


def chbmit_singlech_512samples_patient_5_leave_out_seizure_4_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_512samples",
                                                           patient="5", leave_out_seizure=4)


# --------------------------------------------------------------------------------------------------------------
#                                               768 samples
# --------------------------------------------------------------------------------------------------------------

def chbmit_singlech_768samples_patient_5_leave_out_seizure_0_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_768samples",
                                                           patient="5", leave_out_seizure=0)


def chbmit_singlech_768samples_patient_5_leave_out_seizure_1_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_768samples",
                                                           patient="5", leave_out_seizure=1)


def chbmit_singlech_768samples_patient_5_leave_out_seizure_2_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_768samples",
                                                           patient="5", leave_out_seizure=2)


def chbmit_singlech_768samples_patient_5_leave_out_seizure_3_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_768samples",
                                                           patient="5", leave_out_seizure=3)


def chbmit_singlech_768samples_patient_5_leave_out_seizure_4_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_768samples",
                                                           patient="5", leave_out_seizure=4)


# --------------------------------------------------------------------------------------------------------------
#                                               1016 samples
# --------------------------------------------------------------------------------------------------------------

def chbmit_singlech_1016samples_patient_5_leave_out_seizure_0_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_1016samples",
                                                           patient="5", leave_out_seizure=0)


def chbmit_singlech_1016samples_patient_5_leave_out_seizure_1_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_1016samples",
                                                           patient="5", leave_out_seizure=1)


def chbmit_singlech_1016samples_patient_5_leave_out_seizure_2_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_1016samples",
                                                           patient="5", leave_out_seizure=2)


def chbmit_singlech_1016samples_patient_5_leave_out_seizure_3_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_1016samples",
                                                           patient="5", leave_out_seizure=3)


def chbmit_singlech_1016samples_patient_5_leave_out_seizure_4_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_1016samples",
                                                           patient="5", leave_out_seizure=4)


# --------------------------------------------------------------------------------------------------------------

def chbmit_singlech_1024samples_patient_5_leave_out_seizure_1_get_datasets(data, load_train=True, load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                                           dataset_folder="/chbmit_singlech_1024samples",
                                                           patient="5", leave_out_seizure=1)


datasets = [
    {
        "name": "chbmit_singlech_16samples_patient_5_leave_out_seizure_1",
        "input": (1, 16),
        "output": (0, 1),
        "loader": chbmit_singlech_16samples_patient_5_leave_out_seizure_1_get_datasets,
    },
    {
        "name": "chbmit_singlech_32samples_patient_5_leave_out_seizure_1",
        "input": (1, 32),
        "output": (0, 1),
        "loader": chbmit_singlech_32samples_patient_5_leave_out_seizure_1_get_datasets,
    },
    {
        "name": "chbmit_singlech_64samples_patient_5_leave_out_seizure_1",
        "input": (1, 64),
        "output": (0, 1),
        "loader": chbmit_singlech_64samples_patient_5_leave_out_seizure_1_get_datasets,
    },
    {
        "name": "chbmit_singlech_128samples_patient_5_leave_out_seizure_1",
        "input": (1, 128),
        "output": (0, 1),
        "loader": chbmit_singlech_128samples_patient_5_leave_out_seizure_1_get_datasets,
    },
    {
        "name": "chbmit_singlech_256samples_patient_5_leave_out_seizure_1",
        "input": (1, 256),
        "output": (0, 1),
        "loader": chbmit_singlech_256samples_patient_5_leave_out_seizure_1_get_datasets,
    },
    # ---------------------------------------- 512 samples ----------------------------------------
    {
        "name": "chbmit_singlech_512samples_patient_5_leave_out_seizure_0",
        "input": (1, 512),
        "output": (0, 1),
        "loader": chbmit_singlech_512samples_patient_5_leave_out_seizure_0_get_datasets,
    },
    {
        "name": "chbmit_singlech_512samples_patient_5_leave_out_seizure_1",
        "input": (1, 512),
        "output": (0, 1),
        "loader": chbmit_singlech_512samples_patient_5_leave_out_seizure_1_get_datasets,
    },
    {
        "name": "chbmit_singlech_512samples_patient_5_leave_out_seizure_2",
        "input": (1, 512),
        "output": (0, 1),
        "loader": chbmit_singlech_512samples_patient_5_leave_out_seizure_2_get_datasets,
    },
    {
        "name": "chbmit_singlech_512samples_patient_5_leave_out_seizure_3",
        "input": (1, 512),
        "output": (0, 1),
        "loader": chbmit_singlech_512samples_patient_5_leave_out_seizure_3_get_datasets,
    },
    {
        "name": "chbmit_singlech_512samples_patient_5_leave_out_seizure_4",
        "input": (1, 512),
        "output": (0, 1),
        "loader": chbmit_singlech_512samples_patient_5_leave_out_seizure_4_get_datasets,
    },
    # ---------------------------------------- 768 samples ----------------------------------------
    {
        "name": "chbmit_singlech_768samples_patient_5_leave_out_seizure_0",
        "input": (1, 768),
        "output": (0, 1),
        "loader": chbmit_singlech_768samples_patient_5_leave_out_seizure_0_get_datasets,
    },
    {
        "name": "chbmit_singlech_768samples_patient_5_leave_out_seizure_1",
        "input": (1, 768),
        "output": (0, 1),
        "loader": chbmit_singlech_768samples_patient_5_leave_out_seizure_1_get_datasets,
    },
    {
        "name": "chbmit_singlech_768samples_patient_5_leave_out_seizure_2",
        "input": (1, 768),
        "output": (0, 1),
        "loader": chbmit_singlech_768samples_patient_5_leave_out_seizure_2_get_datasets,
    },
    {
        "name": "chbmit_singlech_768samples_patient_5_leave_out_seizure_3",
        "input": (1, 768),
        "output": (0, 1),
        "loader": chbmit_singlech_768samples_patient_5_leave_out_seizure_3_get_datasets,
    },
    {
        "name": "chbmit_singlech_768samples_patient_5_leave_out_seizure_4",
        "input": (1, 768),
        "output": (0, 1),
        "loader": chbmit_singlech_768samples_patient_5_leave_out_seizure_4_get_datasets,
    },
    # ---------------------------------------- 1016 samples ----------------------------------------
    {
        "name": "chbmit_singlech_1016samples_patient_5_leave_out_seizure_0",
        "input": (1, 1016),
        "output": (0, 1),
        "loader": chbmit_singlech_1016samples_patient_5_leave_out_seizure_0_get_datasets,
    },
    {
        "name": "chbmit_singlech_1016samples_patient_5_leave_out_seizure_1",
        "input": (1, 1016),
        "output": (0, 1),
        "loader": chbmit_singlech_1016samples_patient_5_leave_out_seizure_1_get_datasets,
    },
    {
        "name": "chbmit_singlech_1016samples_patient_5_leave_out_seizure_2",
        "input": (1, 1016),
        "output": (0, 1),
        "loader": chbmit_singlech_1016samples_patient_5_leave_out_seizure_2_get_datasets,
    },
    {
        "name": "chbmit_singlech_1016samples_patient_5_leave_out_seizure_3",
        "input": (1, 1016),
        "output": (0, 1),
        "loader": chbmit_singlech_1016samples_patient_5_leave_out_seizure_3_get_datasets,
    },
    {
        "name": "chbmit_singlech_1016samples_patient_5_leave_out_seizure_4",
        "input": (1, 1016),
        "output": (0, 1),
        "loader": chbmit_singlech_1016samples_patient_5_leave_out_seizure_4_get_datasets,
    },
    # ------------------------------------------------------------------------------------------
    {
        "name": "chbmit_singlech_1024samples_patient_5_leave_out_seizure_1",
        "input": (1, 1024),
        "output": (0, 1),
        "loader": chbmit_singlech_1024samples_patient_5_leave_out_seizure_1_get_datasets,
    }
]

if __name__ == "__main__":
    class ArgsObj:
        act_mode_8bit = False
    train_ds, test_ds = chbmit_singlech_256samples_patient_5_leave_out_seizure_1_get_datasets(("../data", ArgsObj()))
    print("Train dataset")
    print(f"Length: {len(train_ds)}")
    print(f"Element 100: {train_ds[100]}")
    print(f"Dimensions: {train_ds[100][0].shape}")
    #print(f"Dimensions: {train_ds[100][1].shape}")

    print("Test dataset")
    print(f"Length: {len(test_ds)}")
    print(f"Element 100: {test_ds[100]}")
    print(f"Dimensions: {test_ds[100][0].shape}")
    #print(f"Dimensions: {test_ds[100][1].shape}")
