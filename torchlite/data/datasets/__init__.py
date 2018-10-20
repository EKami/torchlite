import torch
import tensorflow as tf
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os


class DatasetWrapper:
    def __init__(self, dataset, steps=None, batch_size=None):
        self.batch_size = batch_size
        self.steps = steps
        self.dataset = dataset

    @classmethod
    def wrap_tf_dataset(cls, dataset: tf.data.Dataset, steps, batch_size):
        """
            A Dataset abstraction around tf.data.Dataset
        Args:
            batch_size (int): Just for indicative purposes, don't actually apply it to the passed dataset
            dataset (tf.data.Dataset): The tf.data.Dataset, yielding batches of data
            steps (int): Number of steps to iterate over the dataset before no data is left
        """
        return cls(dataset, steps, batch_size)

    @classmethod
    def wrap_torch_dataloader(cls, dataloader: torch.utils.data.DataLoader):
        """
        A Dataset abstraction around Pytorch Dataloaders
        Args:
            dataloader (torch.utils.data.DataLoader): A Pytorch Dataloader

        Returns:

        """
        return cls(dataloader)

    @property
    def get_batch_size(self):
        if isinstance(self.dataset, torch.utils.data.DataLoader):
            return self.dataset.batch_size
        else:
            return self.batch_size

    def __len__(self):
        if isinstance(self.dataset, torch.utils.data.DataLoader):
            return len(self.dataset)
        else:
            return self.steps

    def __iter__(self):
        return self.dataset.__iter__()

    def __next__(self):
        return self.dataset.__next__()


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
