import hydra
from omegaconf import DictConfig, OmegaConf

from timESD.source.model import BasicESD
from timESD.source.data import ESDDataModule

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
    model = BasicESD(**FLAGS.experiment)

    # ------------
    # training
    # ------------
    # TODO add callbacks
    callbacks = ''

    # TODO add logger!
    trainer = pl.Trainer(**FLAGS.trainer, callbacks=callbacks)
    trainer.fit(model, datamodule)

    # ------------
    # testing
    # ------------
    result = trainer.test()
    print(result)


@hydra.main(config_path='./source/config/', config_name="timesd")
def main(FLAGS: DictConfig):
    OmegaConf.set_struct(FLAGS, False)
    FLAGS.setup.config_path = config_path
    return train(FLAGS)


if __name__ == '__main__':
    cli_main()
