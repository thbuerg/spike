import hydra
from omegaconf import DictConfig, OmegaConf

from timESD.source.model import BasicESD
from timESD.source.data import ESDDataModule
from timESD.source.utils import set_up_neptune, get_neptune_params, get_default_callbacks


import torch
import pytorch_lightning as pl
pl.seed_everything(23)
print('hi')


def train(FLAGS):
    print(OmegaConf.to_yaml(FLAGS))
    # ------------
    # data
    # ------------
    datamodule = ESDDataModule(**FLAGS.experiment)
    datamodule.prepare_data()
    datamodule.setup("fit")

    # ------------
    # model
    # ------------
    model = BasicESD(**FLAGS.experiment, loss_fn=torch.nn.BCELoss)

    # ------------
    # LR FINDER:
    # ------------
    model = BasicESD(**FLAGS.experiment, loss_fn=torch.nn.BCELoss)
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


@hydra.main(config_path='./source/config/', config_name="timesd.yml")
def main(FLAGS: DictConfig):
    OmegaConf.set_struct(FLAGS, False)
    return train(FLAGS)


if __name__ == '__main__':
    main()
