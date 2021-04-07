# Datasets and DatasetModules.

import pandas as pd
import numpy as np

import torch

from copy import deepcopy

from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader


class EnergyDemandDataset(Dataset):
    """
    Historical Energy Demand Data.

    Dataset yields from underlying .csv file
    """
    def __init__(self, filepath, n_historical_timepoints):
        self.filepath = filepath
        self.n_historical_timepoints = n_historical_timepoints
        data = pd.read_csv(filepath)
        data['date_time'] = pd.to_datetime(data['date_time'])
        self.data = data.set_index('date_time')
        self.dates_map = self.data.index.date.unique()

        # restrict dates_map to only inlcude dates w/ enough history
        # TODO make sure that this is a valid exclusion
        self.dates_map = self.dates_map[n_historical_timepoints:]

    def __getitem__(self, idx):
        date = self.dates_map[idx]
        time_series = self.data.loc[date-self.n_historical_timepoints:date] # TODO implement this indexing here
        return torch.Tensor(time_series)

    def __len__(self):
        return self.dates_map.shape[0]


class ESDDataModule(pl.LightningDataModule):
    """
    DataModule for ESD dataset. Follows simple pl API.
    """
    def __init__(self, filepath, num_workers=8, **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.filepath = filepath

    def get_datasets(self):
        # first get dataset
        dataset = EnergyDemandDataset(filepath=self.filepath, n_historical_timepoints=self.n_historical_timepoints)
        val_size = int(0.20 * len(dataset))
        test_size = 0.5 * val_size

        # TODO: make this random!
        # split dates:
        test_ds = deepcopy(dataset)
        test_ds.dates_map = test_ds.dates_map.iloc[:test_size]

        valid_ds = deepcopy(dataset)
        valid_ds.dates_map = valid_ds.dates_map.iloc[test_size:valid_size]

        train_ds = dataset
        train_ds.dates_map = train_ds.dates_map.iloc[valid_size:]

        return train_ds, valid_ds, test_ds

    def setup(self, stage=None):
        train_ds, valid_ds, test_ds = self.get_datasets()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        raise NotImplementedError('Not Implemented.')
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)