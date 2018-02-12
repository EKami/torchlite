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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchlight.data.fetcher as fetcher
import torchlight.data.files as tfiles
from torchlight.nn.models.srgan import Generator, Discriminator
from torchlight.nn.train_callbacks import ModelSaverCallback, ReduceLROnPlateau
from torchlight.data.datasets.srgan import TrainDataset, ValDataset
from torchlight.nn.learners.learner import Learner
from torchlight.nn.learners.cores import ClassifierCore, SRGanCore
from torchlight.nn.losses.srgan import GeneratorLoss


def enhance_img(img):
    """
    Method used for inference only
    """
    pass


def get_loaders(args, num_workers=os.cpu_count()):
    # TODO take a look and use this datasets: https://superresolution.tf.fau.de/
    ds_path = Path("/tmp")
    fetcher.WebFetcher.download_dataset("https://s3-eu-west-1.amazonaws.com/torchlight-datasets/DIV2K_sample.zip",
                                        ds_path.absolute(), True)
    ds_path = ds_path / "DIV2K"
    if args.train_hr == "@default" and args.train_lr == "@default":
        train_hr_path = ds_path / "DIV2K_train_HR"
    else:
        train_hr_path = Path(args.train_hr)
    val_hr_path = ds_path / "DIV2K_valid_HR"

    # TODO normalize the images
    train_ds = TrainDataset(tfiles.get_files(train_hr_path.absolute()),
                            lr_image_filenames=None,  # Use LR images from dir?
                            crop_size=args.crop_size, upscale_factor=args.upscale_factor)

    # Use the DIV2K dataset for validation as default
    val_ds = ValDataset(tfiles.get_files(val_hr_path.absolute()),
                        args.upscale_factor)

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl


def main(args):
    train_loader, valid_loader = get_loaders(args)
    saved_model_name = "srgan_model_upfac-" + str(args.upscale_factor) + "_crop-" + str(args.crop_size) + ".pth"
    saved_model_path = tfiles.create_dir_if_not_exists(args.models_dir) / saved_model_name

    netG = Generator(args.upscale_factor)
    netD = Discriminator()
    optimizer_g = optim.Adam(netG.parameters())
    optimizer_d = optim.Adam(netD.parameters())

    print("---------------------- Generator training ----------------------")
    callbacks = [ReduceLROnPlateau(optimizer_g, loss_step="train")]
    loss = nn.MSELoss()
    learner = Learner(ClassifierCore(netG, optimizer_g, loss))
    learner.train(args.gen_epochs, None, train_loader, None, callbacks)

    print("----------------- Adversarial (SRGAN) training -----------------")
    callbacks = [ModelSaverCallback(saved_model_path.absolute(), every_n_epoch=5),
                 ReduceLROnPlateau(optimizer_g, loss_step="valid")]

    g_loss = GeneratorLoss()
    learner = Learner(SRGanCore(netG, netD, optimizer_g, optimizer_d, g_loss))
    learner.train(args.gen_epochs, None, train_loader, valid_loader, callbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--train_hr', default="@default", type=str, help='The path to the HR files for training')
    parser.add_argument('--train_lr', default="@default", type=str, help='The path to the LR files for training')
    parser.add_argument('--gen_epochs', default=1, type=int, help='Number of epochs for the generator training')
    parser.add_argument('--adv_epochs', default=500, type=int, help='Number of epochs for the adversarial training')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    # Models with different upscale factors and crop sizes are not compatible together
    parser.add_argument('--crop_size', default=84, type=int, help='training images crop size')  # 384
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                        help='super resolution upscale factor')
    parser.add_argument('--models_dir', default="checkpoint", type=str,
                        help='The path to the saved model. This allow for the training to continue where it stopped')

    main(parser.parse_args())
