from typing import Union
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os


class ImagesDataset(Dataset):
    def __init__(self, images_path: list, y: Union[np.ndarray, list], x_transforms=None, y_transforms=None):
        """
            Dataset class for images classification.
            Works for single and multi label classes.
        Args:
            images_path (list): A list of image path
            y (np.ndarray, list): The image labels as int or the path to the mapping files
            x_transforms (Compose): A list of composable transforms. The transformations will be
                applied to the images_path files
            y_transforms (Compose): A list of composable transforms. The transformations will be
                applied to the y files
        """
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.images_path = images_path
        if isinstance(y, list):
            self.y = y
        else:
            self.y = torch.from_numpy(y.astype(np.int32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = Image.open(self.images_path[idx])

        # Transforms can include resize, normalization and torch tensor transformation
        if self.x_transforms:
            image = self.x_transforms(image)

        if isinstance(self.y, list):
            y = Image.open(self.y[idx])
            if self.y_transforms:
                y = self.y_transforms(y)
        else:
            y = self.y[idx]
        return image, y

    def get_by_name(self, name: str, transform: bool = False):
        """
        Get an image given its name.
        The image will pass through the preprocessing operations/transformations
        if transform is True.
        Args:
            name (str): The image name
            transform (bool): If True apply the transformations to the image

        Returns:
            tuple: (image, label, image_index)
        """
        for i, path in enumerate(self.images_path):
            _, file = os.path.split(path)
            if name == file:
                if not transform:
                    transform_old = self.x_transforms
                    self.x_transforms = False
                    ret = self[i][0], self[i][1], i
                    self.x_transforms = transform_old
                else:
                    ret = self[i][0], self[i][1], i
                return ret


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
