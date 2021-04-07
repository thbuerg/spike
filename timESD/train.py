from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms



def cli_main():
    pl.seed_everything(23)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    datasetmodule = ESDatasetModule('', train=True, download=True, transform=transforms.ToTensor())

    # ------------
    # model
    # ------------
    model = ESDModule()

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule)

    # ------------
    # testing
    # ------------
    result = trainer.test()
    print(result)


if __name__ == '__main__':
    cli_main()
