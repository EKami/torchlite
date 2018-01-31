"""
This script implements the grad-cam++ method to look inside
a CNN and see a heatmap of where the network is looking to do
its predictions. Link to the paper: https://arxiv.org/abs/1710.11063

The presented code uses a pretrained VGG model on a cat/dog classification
task as a demo of using Grad-cam++ in that context.

https://github.com/fastai/fastai/blob/master/courses/dl1/lesson7-CAM.ipynb
https://github.com/adityac94/Grad_CAM_plus_plus
"""
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import pandas as pd
from skimage import io
import torchlight.data.fetcher as fetcher
import torchlight.nn.tools as tools
import torchlight.nn.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchlight.shortcuts import ImageClassifierShortcut
from torchlight.nn.learner import Learner
from torchlight.nn.metrics import CategoricalAccuracy
import torch.optim as optim
import torch.nn.functional as F


class CatsDogsDataset(Dataset):
    def __init__(self, csv_file: Path, root_dir: Path):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
        """
        self.data_df = pd.read_csv(csv_file)
        self.data_df['type'] = self.data_df['type'].astype('category')
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_infos = self.root_dir / self.data_df.iloc[idx, 0]

        image = io.imread(img_infos["image_name"])
        # TODO resize
        # TODO normalize

        return image, img_infos['type'].cat.codes


def main():
    root_dir = "/tmp/dogscat"
    fetcher.WebFetcher.download_dataset("https://www.dropbox.com/s/3ua7ocnuhukmvzs/dogscats.zip", root_dir, True)
    root_dir = Path(root_dir)
    tmp_dir = root_dir / "tmp"

    resnet = torchvision.models.resnet34(pretrained=True)
    # Take the head of resnet up until AdaptiveAvgPool2d
    resnet_head = tools.children(resnet)[:-2]
    # Fine tuning
    net = nn.Sequential(*resnet_head,
                        nn.Conv2d(512, 2, 3, padding=1),
                        nn.AdaptiveAvgPool2d(1), models.Flatten(),
                        nn.LogSoftmax())

    # Image augmentation
    transformations = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),
                                          transforms.Resize((224, 224)),
                                          ])
    shortcut = ImageClassifierShortcut.from_paths(root_dir=root_dir,
                                                  preprocess_dir=tmp_dir,
                                                  transforms=transformations)


if __name__ == "__main__":
    main()
