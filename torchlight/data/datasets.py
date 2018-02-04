import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
import numpy as np


class ImagesDataset(Dataset):
    def __init__(self, images_path: list, y: np.ndarray, transforms=None, image_type="image"):
        """
            Dataset class for images classification.
            Works for single and multi label classes.
        Args:
            images_path (list): The path the images
            y (np.ndarray): The image labels as int
            transforms (Compose): A list of composable transforms
            image_type (str): Either:
             - image
             - blosc-array
        """
        self.image_type = image_type
        self.transforms = transforms
        self.images_path = images_path
        self.y_onehot = torch.LongTensor(len(y), len(np.unique(y)))
        self.y_onehot.zero_()
        labels_tensors = torch.unsqueeze(torch.from_numpy(y.astype(np.long)), 1)
        self.y_onehot.scatter_(1, labels_tensors, 1)

    def __len__(self):
        return len(self.y_onehot)

    def __getitem__(self, idx):
        if self.image_type == "blosc-array":
            # TODO finish
            image = Image.fromarray(io.imread(self.images_path[idx]))
        else:
            image = Image.open(self.images_path[idx])

        # Transforms can include resize, normalization and torch tensor transformation
        if self.transforms:
            image = self.transforms(image)

        return image, self.y_onehot[idx]


class ColumnarDataset(Dataset):
    # https://github.com/EKami/carvana-challenge/blob/master/src/data/dataset.py
    def __init__(self, cats, conts, y):
        n = len(cats[0]) if cats else len(conts[0])
        self.cats = np.stack(cats, 1).astype(np.int64) if cats else np.zeros((n, 1))
        self.conts = np.stack(conts, 1).astype(np.float32) if conts else np.zeros((n, 1))
        # Fill y with 0 for the test dataset, they will be ignored during the prediction phase
        self.y = np.zeros((n, 1)) if y is None else y[:, None]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.y[idx]]

    @classmethod
    def from_data_frames(cls, df_cat, df_cont, y=None):
        cat_cols = [c.values for n, c in df_cat.items()]
        cont_cols = [c.values for n, c in df_cont.items()]
        return cls(cat_cols, cont_cols, y)

    @classmethod
    def from_data_frame(cls, df, cat_flds, y=None):
        return cls.from_data_frames(df[cat_flds], df.drop(cat_flds, axis=1), y)
