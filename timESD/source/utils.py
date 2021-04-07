import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger


def set_up_neptune(project_name='debug', experiment_name='debug', params={}, tags=[], close_after_fit=False, **kwargs):
    """
    Set up a neptune logger from file.
    :param keyfile:
    :param project_name:
    :param experiment_name:
    :param params:
    :param tags:
    :param close_after_fit:
    :param kwargs:
    :return:
    """
    if not "NEPTUNE_API_TOKEN" in os.environ:
        raise EnvironmentError('Please set environment variable `NEPTUNE_API_TOKEN`.')

    neptune_logger = NeptuneLogger(api_key=os.environ["NEPTUNE_API_TOKEN"],
                                    project_name=project_name,
                                    experiment_name=experiment_name,
                                    params=params,
                                    tags=tags,
                                    close_after_fit=close_after_fit)

    return neptune_logger


def get_neptune_params(FLAGS, callbacks=[]):
    """
    :param FLAGS:
    :param callbacks:
    :return:
    """
    neptune_params = {
        "project_name": FLAGS.setup.project_name,
        "experiment_name": FLAGS.setup.experiment_name,
        "tags": FLAGS.setup.tags.rstrip(',').split(','),
        "params": {**FLAGS.experiment, **FLAGS.trainer},
        "callbacks": [type(cb).__name__ for cb in callbacks],
        "close_after_fit": False
    }
    return neptune_params


def get_default_callbacks(monitor='val_loss', mode='min', early_stop=True):
    """
    Instantate the default callbacks: EarlyStopping and Checkpointing.
    :param monitor:
    :param mode:
    :return:
    """
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor=monitor, verbose=True,
                                                                        save_last=True, save_top_k=3,
                                                                        save_weights_only=False, mode=mode,
                                                                        period=1, prefix='')
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=False)
    if early_stop:
        early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10,
                                                               verbose=True, mode='min', strict=False)
        return [checkpoint_callback, early_stop, lr_monitor]
    else:
        return [checkpoint_callback, lr_monitor]