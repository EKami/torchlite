import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os


class ImageDataset(Dataset):
    def __init__(self, images_path: list):
        """
        Dataset class for images classification.
        Args:
            images_path (list): A list of image path
        """
        self.images_path = images_path

    def __len__(self):
        return len(self.images_path)

    def get_by_name(self, name: str):
        """
        Get an image given its name.
        The image will not pass through the preprocessing operations/transformations.
        Args:
            name (str): The image name

        Returns:
            tuple: (image, label, image_index)
        """
        for i, path in enumerate(self.images_path):
            _, file = os.path.split(path)
            if name == file:
                return Image.open(path), self[i][1], i


class ImageClassificationDataset(ImageDataset):
    def __init__(self, images_path: list, y: np.ndarray, transforms=None):
        """
            Dataset class for images classification.
        Args:
            images_path (list): A list of image path
            y (np.ndarray): The image labels as a list of int
            transforms (Compose): A list of composable transforms. The transformations will be
                applied to the images_path files
        """
        super().__init__(images_path)
        self.transforms = transforms
        if isinstance(y, list):
            self.y = y
        else:
            self.y = torch.from_numpy(y.astype(np.int64))

    def __getitem__(self, idx):
        image = Image.open(self.images_path[idx])

        # Transforms can include resize, normalization and torch tensor transformation
        if self.transforms:
            image = self.transforms(image)

        return image, self.y[idx]


class ColumnarDataset(Dataset):
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
