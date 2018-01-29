"""
This script implements the grad-cam++ method to look inside
a CNN and see a heatmap of where the network is looking to do
its predictions. Link to the paper: https://arxiv.org/abs/1710.11063

The presented code uses a pretrained VGG model on a cat/dog classification
task as a demo of using Grad-cam++ in that context.

https://github.com/fastai/fastai/blob/master/courses/dl1/lesson7-CAM.ipynb
https://github.com/adityac94/Grad_CAM_plus_plus
"""
from pathlib import Path
import pandas as pd
from skimage import io
from torchlight.nn.learner import Learner
from torchlight.nn.metrics import CategoricalAccuracy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

        return image, img_infos['type'].cat.codes


def main():
    test_data = CatsDogsDataset(csv_file=Path("datasets/catsdogs/labels.csv"),
                                root_dir=Path("datasets/catsdogs/"))
    net = None


if __name__ == "__main__":
    main()
