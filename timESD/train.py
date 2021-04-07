import hydra
from omegaconf import DictConfig, OmegaConf

from timESD.source.model import BasicESD
from timESD.source.data import ESDDataModule
from timESD.source.utils import set_up_neptune, get_neptune_params, get_default_callbacks


import torch
import pytorch_lightning as pl
pl.seed_everything(23)


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
