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
import torchlight.nn.tools.image_tools as image_tools
from torchlight.nn.models.srgan import Generator, Discriminator
from torchlight.nn.train_callbacks import ModelSaverCallback, ReduceLROnPlateau, TensorboardVisualizerCallback
from torchlight.data.datasets.srgan import TrainDataset, ValDataset, EvalDataset
from torchlight.nn.learners.learner import Learner
from torchlight.nn.learners.cores import ClassifierCore, SRGanCore
from torchlight.nn.losses.srgan import PerceptualLoss
from torchlight.nn.metrics.metrics import SSIM, PSNR

cur_path = os.path.dirname(os.path.abspath(__file__))
tensorboard_dir = tfiles.del_dir_if_exists(os.path.join(cur_path, "tensorboard"))
saved_model_dir = tfiles.create_dir_if_not_exists(os.path.join(cur_path, "checkpoints"))


def get_loaders(args, num_workers=os.cpu_count()):
    # TODO take a look and use this datasets: https://superresolution.tf.fau.de/
    ds_path = Path("/tmp")
    fetcher.WebFetcher.download_dataset("https://s3-eu-west-1.amazonaws.com/torchlight-datasets/DIV2K.zip",
                                        ds_path.absolute(), True)
    ds_path = ds_path / "DIV2K"
    if args.hr_dir == "@default" and args.lr_dir == "@default":
        train_hr_path = ds_path / "DIV2K_train_HR"
    else:
        train_hr_path = Path(args.hr_dir)
    val_hr_path = ds_path / "DIV2K_valid_HR"

    train_ds = TrainDataset(tfiles.get_files(train_hr_path.absolute()),
                            lr_image_filenames=None,  # Use LR images from dir?
                            crop_size=args.crop_size, upscale_factor=args.upscale_factor)

    # Use the DIV2K dataset for validation as default
    val_ds = ValDataset(tfiles.get_files(val_hr_path.absolute()),
                        crop_size=args.crop_size, upscale_factor=args.upscale_factor)

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl


def evaluate(args, num_workers=os.cpu_count()):
    """
    Method used for inference only
    """
    if args.images_dir == "@default":
        imgs_path = os.path.join(cur_path, "eval")
    else:
        imgs_path = args.images_dir
    if args.to_dir == "@default":
        to_dir = tfiles.del_dir_if_exists(os.path.join(cur_path, "results"))
    else:
        to_dir = Path(args.to_dir)
    netG = Generator(args.upscale_factor)
    learner = Learner(ClassifierCore(netG, None, None), use_cuda=args.cuda)
    ModelSaverCallback.restore_model([netG], saved_model_dir.absolute(), load_with_cpu=not args.cuda)
    eval_ds = EvalDataset(tfiles.get_files(imgs_path))
    # One batch at a time as the pictures may differ in size
    eval_dl = DataLoader(eval_ds, 1, shuffle=False, num_workers=num_workers)

    predictions = learner.predict(eval_dl)
    for i, pred in enumerate(predictions):
        pred = pred.view(pred.size()[1:])  # Remove batch size == 1
        file_name = eval_ds.get_file_from_index(i)
        image_tools.save_tensor_as_png(pred, (to_dir / file_name).absolute())


def train(args):
    train_loader, valid_loader = get_loaders(args)

    model_saver = ModelSaverCallback(saved_model_dir.absolute(), args.adv_epochs, every_n_epoch=5)

    netG = Generator(args.upscale_factor)
    netD = Discriminator((3, args.crop_size, args.crop_size))
    optimizer_g = optim.Adam(netG.parameters())
    optimizer_d = optim.Adam(netD.parameters())

    # Restore models if they exists
    if args.restore_models == 1:
        model_saver.restore_model([netG, netD], saved_model_dir.absolute())

    if args.gen_epochs > 0:
        print("---------------------- Generator training ----------------------")
        callbacks = [ReduceLROnPlateau(optimizer_g, loss_step="train")]
        loss = nn.MSELoss()
        learner = Learner(ClassifierCore(netG, optimizer_g, loss))
        learner.train(args.gen_epochs, None, train_loader, None, callbacks)

    print("----------------- Adversarial (SRGAN) training -----------------")
    callbacks = [model_saver, ReduceLROnPlateau(optimizer_g, loss_step="valid"),
                 TensorboardVisualizerCallback(tensorboard_dir.absolute())]

    g_loss = PerceptualLoss()
    learner = Learner(SRGanCore(netG, netD, optimizer_g, optimizer_d, g_loss))
    learner.train(args.adv_epochs, [SSIM("validation"), PSNR("validation")], train_loader, valid_loader, callbacks)


def main():
    parser = argparse.ArgumentParser(description='Train/Evaluate Super Resolution Models. While training 2 new '
                                                 'directories will be created at the same level of this file: '
                                                 '("checkpoints") containing the saved models and '
                                                 '("tensorboard") containing the tensorboard logs. ')
    subs = parser.add_subparsers(dest='mode')
    train_parser = subs.add_parser('train', help='Use this script in train mode')
    eval_parser = subs.add_parser('eval', help='Use this script in evaluation mode')

    train_parser.add_argument('--hr_dir', default="@default", type=str, help='The path to the HR files for training')
    train_parser.add_argument('--lr_dir', default="@default", type=str,
                              help='The path to the LR files for training (not used for now)')
    train_parser.add_argument('--gen_epochs', default=0, type=int, help='Number of epochs for the generator training')
    train_parser.add_argument('--adv_epochs', default=2000, type=int,
                              help='Number of epochs for the adversarial training')
    train_parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    train_parser.add_argument('--restore_models', default=0, type=int, choices=[0, 1],
                              help="0: Don't restore the models and erase the existing ones. "
                                   "1: Restore the models from the 'checkpoint' folder")
    # Models with different upscale factors and crop sizes are not compatible together
    train_parser.add_argument('--crop_size', default=384, type=int, help='training images crop size')
    train_parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                              help="Super Resolution upscale factor. "
                                   "/!\ Models trained on different scale factors won't be compatible with each other")

    eval_parser.add_argument('--images_dir', default="@default", type=str, help='The path to the files for SR')
    eval_parser.add_argument('--to_dir', default="@default", type=str,
                             help='The directory where the SR files will be stored')
    eval_parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                             help='Super Resolution upscale factor')
    eval_parser.add_argument('--on_cpu', dest="cuda", action="store_false",
                             help='Run eval on the CPU (defaults to on the GPU)')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        parser.print_usage()


if __name__ == "__main__":
    main()
