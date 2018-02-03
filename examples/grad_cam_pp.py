"""
This script implements the grad-cam++ method to look inside
a CNN and see a heatmap of where the network is looking to do
its predictions. Link to the paper: https://arxiv.org/abs/1710.11063

The presented code uses a pretrained VGG model on a cat/dog classification
task as a demo of using Grad-cam++ in that context.

In

Resources:
 - https://github.com/fastai/fastai/blob/master/courses/dl1/lesson7-CAM.ipynb
 - https://github.com/adityac94/Grad_CAM_plus_plus
"""
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchlight.data.fetcher as fetcher
import torchlight.nn.tools as tools
import torch.nn as nn
from torchlight.shortcuts import ImageClassifierShortcut
from torchlight.nn.learner import Learner
from torchlight.nn.metrics import CategoricalAccuracy


def main():
    batch_size = 128
    epochs = 20
    root_dir = "/tmp/dogscat"
    fetcher.WebFetcher.download_dataset("https://s3-eu-west-1.amazonaws.com/torchlight-datasets/dogscats.zip", root_dir,
                                        True)
    root_dir = Path(root_dir)
    tmp_dir = root_dir / "tmp"
    train_folder = root_dir / "train"
    val_folder = root_dir / "valid"

    # Image augmentation
    transformations = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),
                                          transforms.Resize((224, 224)),
                                          ])
    shortcut = ImageClassifierShortcut.from_paths(train_folder=train_folder.absolute(),
                                                  val_folder=val_folder.absolute(),
                                                  preprocess_dir=tmp_dir.absolute(),
                                                  transforms=transformations,
                                                  batch_size=batch_size)
    net = shortcut.get_resnet_model()
    learner = Learner(net)

    optimizer = optim.RMSprop(net.parameters(), lr=1e-3)
    loss = F.nll_loss
    metrics = [CategoricalAccuracy()]

    learner.train(optimizer, loss, metrics, epochs, shortcut.get_train_loader, None, callbacks=None)


if __name__ == "__main__":
    main()
