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
from torch.utils.data import DataLoader
import torchlight.data.fetcher as fetcher
from torchlight.data.datasets import ImagesDataset
import torchlight.data.files as tfiles
from torchlight.nn.models.srgan import Generator, Discriminator


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

    train_ds = ImagesDataset(tfiles.get_files(train_lr_path.absolute()),
                             y=tfiles.get_files(train_hr_path.absolute()))
    # Use the DIV2K dataset for validation as default
    val_ds = ImagesDataset(tfiles.get_files(val_lr_path.absolute()),
                           y=tfiles.get_files(val_hr_path.absolute()))

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, None


def main(args):
    batch_size = 128
    epochs = 20

    train_loader, valid_loader, _ = get_loaders(args)
    saved_model_path = tfiles.create_dir_if_not_exists(args.models_dir) / "srgan_model.pth"
    netG = Generator(args.upscale_factor)
    netD = Discriminator()
    d = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--train_hr', default="@default", type=str, help='The path to the HR files for training')
    parser.add_argument('--train_lr', default="@default", type=str, help='The path to the LR files for training')
    parser.add_argument('--gen_epochs', default=100, type=int, help='Number of epochs for the generator training')
    parser.add_argument('--adv_epochs', default=150, type=int, help='Number of epochs for the adversarial training')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--models_dir', default="checkpoint", type=str,
                        help='The path to the saved model. This allow for the training to continue where it stopped')
    parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                        help='Super resolution upscale factor')

    main(parser.parse_args())
