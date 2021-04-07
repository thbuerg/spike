# Datasets and DatasetModules.

import pandas as pd
import numpy as np

import torch

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
