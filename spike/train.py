import hydra
from omegaconf import DictConfig, OmegaConf

from spike.source.model import BasicNet
from spike.source.data import ESDDataModule
from spike.source.utils import set_up_neptune, get_neptune_params, get_default_callbacks

import torch
import pytorch_lightning as pl
pl.seed_everything(23)


def train(FLAGS):
    print(OmegaConf.to_yaml(FLAGS))
    # ------------
    # LR FINDER:
    # ------------
    datamodule = ESDDataModule(**FLAGS.experiment)
    datamodule.prepare_data()
    datamodule.setup("fit")
    model = BasicNet(**FLAGS.experiment, loss_fn=torch.nn.BCELoss)
    trainer = pl.Trainer(**FLAGS.trainer)
    lr = trainer.tuner.lr_find(model, datamodule, num_training=500).suggestion()
    if lr > 0.1:
        lr = FLAGS.optimizer_kwargs["lr"]
        print(f"LR to high -> Corrected to {lr}")
    if lr < 0.00001:
        lr = FLAGS.optimizer_kwargs["lr"]
        print(f"LR to low -> Corrected to {lr}")
    print(f"Best Learning Rate: {lr}")
    FLAGS.optimizer_kwargs["lr"] = lr

    # ------------
    # data
    # ------------
    datamodule = ESDDataModule(**FLAGS.experiment)
    datamodule.prepare_data()
    datamodule.setup("fit")


    # ------------
    # model
    # ------------
    model = BasicNet(**FLAGS.experiment, loss_fn=torch.nn.BCELoss)

    # ------------
    # training
    # ------------
    callbacks = get_default_callbacks(monitor='val_loss', mode='min', early_stop=False)
    trainer = pl.Trainer(**FLAGS.trainer,
                         callbacks=callbacks,
                         logger=set_up_neptune(**get_neptune_params(FLAGS, callbacks)))
    trainer.fit(model, datamodule)

    # ------------
    # testing
    # ------------
    result = trainer.test()
    print(result)


@hydra.main(config_path='./source/config/', config_name="spike.yml")
def main(FLAGS: DictConfig):
    OmegaConf.set_struct(FLAGS, False)
    return train(FLAGS)


if __name__ == '__main__':
    main()
