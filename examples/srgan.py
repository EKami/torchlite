"""
This script implements a neural network to enhance images:
 - https://arxiv.org/abs/1712.05927

Private dropbox roadmap: https://paper.dropbox.com/doc/Resources-qhkHHNEfwWgF28qMFU8tR

To implement:
 - https://github.com/aitorzip/PyTorch-SRGAN
 - https://github.com/leftthomas/SRGAN (PyTorch)
 - https://github.com/tensorlayer/srgan (Tensorflow)
"""
import os
import argparse
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
import torchlight.data.fetcher as fetcher
from torchlight.data.datasets import ImageDataset
import torchlight.data.files as tfiles
from torchlight.nn.models.srgan import Generator, Discriminator
from torchlight.nn.losses.srgan import GeneratorLoss
from torchlight.nn.learner import Learner
from torchlight.nn.train_callbacks import ModelSaverCallback, ReduceLROnPlateau
from PIL import Image
import torchvision.transforms as transforms
import torchlight.nn.transforms as ttransforms
import torch.nn as nn


def enhance_img(img):
    """
    Method used for inference only
    """
    pass


def get_loaders(args, num_workers=os.cpu_count()):
    # TODO take a look and use this datasets: https://superresolution.tf.fau.de/
    ds_path = Path("/tmp")
    fetcher.WebFetcher.download_dataset("https://s3-eu-west-1.amazonaws.com/torchlight-datasets/DIV2K.zip",
                                        ds_path.absolute(), True)
    ds_path = ds_path / "DIV2K"
    if args.train_hr == "@default" and args.train_lr == "@default":
        train_hr_path = ds_path / "DIV2K_train_HR"
        train_lr_path = ds_path / "DIV2K_train_LR_bicubic" / "X4"
    else:
        train_hr_path = Path(args.train_hr)
        train_lr_path = Path(args.train_lr)
    val_hr_path = ds_path / "DIV2K_valid_HR"
    val_lr_path = ds_path / "DIV2K_valid_LR_bicubic" / "X4"

    # TODO may not be compatible with VGG!!
    # TODO crop here
    x_transformations = transforms.Compose([transforms.RandomCrop((88, 88)),  # TODO change with crop?
                                            transforms.ToTensor(),
                                            ttransforms.FactorNormalize(),
                                            ])

    # TODO the author downsample the 386x386 HR images to 96x96
    # TODO resize here or crop if taken from val folder
    # https://github.com/tensorlayer/srgan/blob/cd9dc3a67275ece28165e52fbed3a81bc56c4e43/main.py#L201
    y_transformations = transforms.Compose([transforms.Resize((96, 96), interpolation=Image.BICUBIC),
                                            transforms.ToTensor(),
                                            ttransforms.FactorNormalize(),
                                            ])

    train_ds = ImagesDataset(tfiles.get_files(train_lr_path.absolute()),
                             y=tfiles.get_files(train_hr_path.absolute()),
                             x_transforms=x_transformations,
                             y_transforms=y_transformations)
    # Use the DIV2K dataset for validation as default
    val_ds = ImagesDataset(tfiles.get_files(val_lr_path.absolute()),
                           y=tfiles.get_files(val_hr_path.absolute()),
                           x_transforms=x_transformations,
                           y_transforms=y_transformations)

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, None


def main(args):
    upscale_factor = 4  # Models with different upscale factors are not compatible together

    train_loader, valid_loader, _ = get_loaders(args)
    saved_model_path = tfiles.create_dir_if_not_exists(args.models_dir) / "srgan_model.pth"
    netG = Generator(upscale_factor)
    netD = Discriminator()

    optimizer_g = optim.Adam(netG.parameters())
    optimizer_d = optim.Adam(netD.parameters())

    generator_epochs = args.gen_epochs
    adversarial_epochs = args.adv_epochs

    loss = GeneratorLoss()  # TODO try with VGG54 as in the paper
    callbacks = [ModelSaverCallback(saved_model_path.absolute(), every_n_epoch=5)]

    #  ---------------------- Train generator a bit ----------------------
    init_callbacks = [ReduceLROnPlateau(optimizer_g)]
    init_loss = nn.MSELoss()
    g_init_learner = Learner(netG)
    g_init_learner.train(optimizer_g, init_loss, None, generator_epochs, train_loader, callbacks=init_callbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--train_hr', default="@default", type=str, help='The path to the HR files for training')
    parser.add_argument('--train_lr', default="@default", type=str, help='The path to the LR files for training')
    parser.add_argument('--gen_epochs', default=100, type=int, help='Number of epochs for the generator training')
    parser.add_argument('--adv_epochs', default=500, type=int, help='Number of epochs for the adversarial training')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--models_dir', default="checkpoint", type=str,
                        help='The path to the saved model. This allow for the training to continue where it stopped')

    main(parser.parse_args())
