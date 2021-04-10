import os

from pytorch_lightning import Trainer, seed_everything
from spike.source.model import BasicNet
from spike.source.data import ESDDataModule
from spike.source.preprocessing import main
from omegaconf import OmegaConf


def test_setup():
    seed_everything(23)
    filepath = './data/'
    os.makedirs(filepath, exist_ok=True)

    # preprocessing
    conf = OmegaConf.create({"experiment": {"filepath": './data',}})
    OmegaConf.set_struct(conf, True)
    main(conf)

    # model and data
    model = BasicNet(batch_size=64)
    datamodule = ESDDataModule(filepath=filepath+'data_normed.csv', batch_size=64)
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)

    # run
    trainer.fit(model, datamodule=datamodule)
    results = trainer.test()

    assert results[0]['val_acc@5'] > 0.1