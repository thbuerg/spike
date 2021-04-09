import torch
import torch.nn as nn

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

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.schedule = schedule
        self.schedule_kwargs = schedule_kwargs
        self.loss_fn = loss_fn(**loss_fn_kwargs)

        # networks
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.date_net = nn.Sequential(
            nn.Linear(96*3, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size+64, n_targets),
            nn.Softmax()
        )
        self.netlist = [self.lstm, self.date_net, self.predictor]

        # save hps
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
        ts_data, date_data = inputs
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(ts_data)
        date_fts = self.date_net(date_data)
        predictions = self.predictor(torch.cat([lstm_out[:,-1], date_fts], dim=-1))
        return predictions

    def training_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self(data)
        loss = self.loss_fn(predictions, targets)
        acc1, acc5 = self.accuracy(predictions, targets)
        logs = {'train_acc@1': acc1,
                'train_acc@5': acc5,
                'train_loss': loss}
        for k, v in logs.items():
            self.log(k, v, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self(data)
        loss = self.loss_fn(predictions, targets)
        acc1, acc5 = self.accuracy(predictions, targets)

        # gather results and log
        logs = {'val_acc@1': acc1,
                'val_acc@5': acc5,
                'val_loss': loss}
        for k, v in logs.items():
            self.log(k, v, on_step=False, on_epoch=False, logger=False, prog_bar=False)
        return logs

    def validation_epoch_end(self, outputs):
        losses = {}
        # aggregate the per-batch-metrics:
        for l in ["val_loss", 'val_acc@1', 'val_acc@5']:
            losses[l] = torch.stack([output[l] for output in outputs]).mean()
        for key, value in losses.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self(data)
        loss = self.loss_fn(predictions, targets)
        acc1, acc5 = self.accuracy(predictions, targets)

        # gather results and log
        logs = {'test_acc@1': acc1,
                'test_acc@5': acc5,
                'test_loss': loss}
        for k, v in logs.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss}

    @torch.no_grad()
    def accuracy(self, predictions, targets, k=(1, 5)):
        """
        calculates topk accuracy. Implementation from:
        https://github.com/DonkeyShot21/essential-BYOL/blob/main/byol/model.py
        :param predictions:
        :param targets:
        :param k:
        :return:
        """
        predictions = predictions.topk(max(k), 1, True, True)[1].t() # indices of top k
        correct = predictions.eq(torch.argmax(targets, dim=-1).view(1, -1).expand_as(predictions))

        res = []
        for k_i in k:
            correct_k = correct[:k_i].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / targets.size(0)))
        return res