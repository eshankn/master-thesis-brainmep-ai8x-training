# -*- coding: utf-8 -*-

# import built-in module

# import third-party modules
from torch.utils.data import Dataset as TorchDataset
from brainmepnas import Dataset as BrainMepNasDataset
import torchvision.transforms as transforms
import numpy as np

# import your own module
import ai8x


class ChbMitSingleChannelPatientSpecific(TorchDataset):

    def __init__(self, root_dir, d_type=None, transform=None,
                 patient: str = "5", leave_out_seizure: int = 1):
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
        dataset = BrainMepNasDataset(root_dir + "/chbmit_singlech")
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


def ChbMitSingleChannelPatientSpecific_get_datasets(data,
                                                    load_train=True,
                                                    load_test=True,
                                       patient: str = "5",
                                       leave_out_seizure: int = 1):
    (data_dir, args) = data
    transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args),
        ])

    if load_train:
        train_dataset = ChbMitSingleChannelPatientSpecific(root_dir=data_dir,
                                                           d_type="train",
                                              patient=patient,
                                                           transform=transform,
                                              leave_out_seizure=leave_out_seizure)
    else:
        train_dataset = None

    if load_test:
        test_dataset = ChbMitSingleChannelPatientSpecific(root_dir=data_dir,
                                                          d_type="test",
                                             patient=patient,
                                                          transform=transform,
                                             leave_out_seizure=leave_out_seizure)
    else:
        test_dataset = None

    return train_dataset, test_dataset


def chbmit_singlech_patient_5_leave_out_seizure_1_get_datasets(data, load_train=True,
                                                      load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                              patient="5", leave_out_seizure=1)


def chbmit_singlech_patient_5_leave_out_seizure_2_get_datasets(data, load_train=True,
                                                      load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                              patient="5", leave_out_seizure=2)


def chbmit_singlech_patient_5_leave_out_seizure_3_get_datasets(data, load_train=True,
                                                      load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                              patient="5", leave_out_seizure=3)


def chbmit_singlech_patient_5_leave_out_seizure_4_get_datasets(data, load_train=True,
                                                      load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                              patient="5", leave_out_seizure=4)


def chbmit_singlech_patient_5_leave_out_seizure_5_get_datasets(data, load_train=True,
                                                      load_test=True):
    return ChbMitSingleChannelPatientSpecific_get_datasets(data, load_train, load_test,
                                              patient="5", leave_out_seizure=5)


datasets = [
    {
        "name": "chbmit_singlech_patient_5_leave_out_seizure_1",
        "input": (1, 1024),
        "output": (0, 1),
        "loader": chbmit_singlech_patient_5_leave_out_seizure_1_get_datasets,
    },
    {
        "name": "chbmit_singlech_patient_5_leave_out_seizure_2",
        "input": (1, 1024),
        "output": (0, 1),
        "loader": chbmit_singlech_patient_5_leave_out_seizure_2_get_datasets,
    },
    {
        "name": "chbmit_singlech_patient_5_leave_out_seizure_3",
        "input": (1, 1024),
        "output": (0, 1),
        "loader": chbmit_singlech_patient_5_leave_out_seizure_3_get_datasets,
    },
    {
        "name": "chbmit_singlech_patient_5_leave_out_seizure_4",
        "input": (1, 1024),
        "output": (0, 1),
        "loader": chbmit_singlech_patient_5_leave_out_seizure_4_get_datasets,
    },
    {
        "name": "chbmit_singlech_patient_5_leave_out_seizure_5",
        "input": (1, 1024),
        "output": (0, 1),
        "loader": chbmit_singlech_patient_5_leave_out_seizure_5_get_datasets,
    },
]

if __name__ == "__main__":
    class ArgsObj:
        act_mode_8bit = False
    train_ds, test_ds = chbmit_singlech_patient_5_leave_out_seizure_1_get_datasets(("../data", ArgsObj()))
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
