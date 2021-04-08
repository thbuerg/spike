# Datasets and DatasetModules.

import pandas as pd

import torch

from copy import deepcopy

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class EnergyDemandDataset(Dataset):
    """
    Historical Energy Demand Data.

    Dataset yields from underlying .csv file
    """
    def __init__(self, filepath, n_historical_dates):
        self.filepath = filepath
        self.n_historical_dates = n_historical_dates
        data = pd.read_csv(filepath)
        data['date_time'] = pd.to_datetime(data['date_time'])
        self.data = data.set_index('date_time')[['TT_10', 'GS_10', 'FF_10']]
        self.data.head()
        self.labels = data.set_index('date_time')[['daily_max']]
        self.dates_map = pd.unique(self.data.index.date)

        # restrict dates_map to only inlcude dates w/ enough history
        self.dates_map = self.dates_map[self.n_historical_dates+1:]

    def __getitem__(self, idx):
        date = self.dates_map[idx]
        start_date = (date - pd.tseries.offsets.DateOffset(days=self.n_historical_dates)).date()
        time_series = self.data.loc[start_date:date].values
        label = self.labels.loc[str(self.dates_map[idx])].values.ravel()
        return torch.Tensor(time_series), torch.Tensor(label)

    def __len__(self):
        return self.dates_map.shape[0]


class ESDDataModule(pl.LightningDataModule):
    """
    DataModule for ESD dataset. Follows simple pl API.
    """
    def __init__(self, filepath, batch_size, seq_len, num_workers=8, **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.filepath = filepath

    def get_datasets(self):
        # first get dataset
        dataset = EnergyDemandDataset(filepath=self.filepath, n_historical_dates=self.seq_len//96)
        val_size = int(0.20 * len(dataset))
        test_size = int(0.5 * val_size)

        # TODO: make this random!
        # split dates:
        test_ds = deepcopy(dataset)
        test_ds.dates_map = test_ds.dates_map[:test_size]

        valid_ds = deepcopy(dataset)
        valid_ds.dates_map = valid_ds.dates_map[test_size:val_size]

        train_ds = dataset
        train_ds.dates_map = train_ds.dates_map[val_size:]

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