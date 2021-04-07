import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data


class BasicESD(pl.LightningModule):
    """
    Autoregressive model to predict spikes in enegry demand data.
    """
    def __init__(self,
                 n_features,
                 hidden_size,
                 seq_len,
                 n_targets,
                 batch_size,
                 num_layers,
                 dropout,
                 loss_fn,
                 loss_fn_kwargs={},
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={'lr': 1e-4},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs
                 ):
        super(BasicESD, self).__init__()
        self.loss_fn = loss_fn(**loss_fn_kwargs)
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, n_targets),
            nn.Softmax()
        )
        self.netlist = [self.lstm, self.predictor]

        self.save_hyperparameters()

    def configure_optimizers(self):
        params = []
        for net in self.netlist:
            for p in [net.parameters()]:
                params.extend(p)

        if self.schedule is not None:
            optimizer = self.optimizer(params, **self.optimizer_kwargs)
            if self.schedule == torch.optim.lr_scheduler.ReduceLROnPlateau:
                lr_scheduler = self.schedule(optimizer, **self.schedule_kwargs)
                scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}
                return [optimizer], [scheduler]
            else:
                return [optimizer], [self.schedule(optimizer, **self.schedule_kwargs)]  # not working in 1.X ptl version with ReduceLROnPlateau
        else:
            return self.optimizer(params, **self.optimizer_kwargs)

    @auto_move_data
    def forward(self, inputs):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(inputs)
        y_pred = self.predictor(lstm_out[:,-1])
        return y_pred

    def training_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self(data)
        loss = self.loss_fn(predictions, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {'train_loss': loss}

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self(data)
        loss = self.loss_fn(predictions, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        return result

    def training_step(self, *args, **kwargs):