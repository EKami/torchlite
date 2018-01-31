"""
    This class contains models shortcuts to automatically fit a particular kind of dataset
    in them. The serve as the "default way to go" when you train for a particular dataset
    but don't want to spend time creating the architecture of a model.
"""
from typing import Union
from pathlib import Path
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchlight.data.loaders import ModelData
from torchlight.data.datasets import ColumnarDataset
from torchlight.nn.models import MixedInputModel
import torch.nn as nn
import os

import numpy as np


class ColumnarShortcut(ModelData):
    def __init__(self, train_ds, val_ds=None, test_ds=None, batch_size=64, shuffle=True):
        """
        A shortcut used for structured/columnar data
        Args:
            trn_ds (Dataset): The train dataset
            val_ds (Dataset): The validation dataset
            test_ds (Dataset): The test dataset
            batch_size (int): The batch size for the training
            shuffle (bool): If True shuffle the training set
        """
        self.train_dl = DataLoader(train_ds, batch_size, shuffle=shuffle)
        self.val_dl = DataLoader(val_ds, batch_size, shuffle=False) if val_ds else None
        self.test_dl = DataLoader(test_ds, batch_size, shuffle=False) if test_ds else None
        super().__init__(self.train_dl, self.val_dl, self.test_dl)

    @classmethod
    def from_data_frames(cls, train_df, val_df, train_y, val_y, cat_fields, batch_size, test_df=None):
        test_ds = ColumnarDataset.from_data_frame(test_df, cat_fields) if test_df is not None else None
        return cls(ColumnarDataset.from_data_frame(train_df, cat_fields, train_y),
                   ColumnarDataset.from_data_frame(val_df, cat_fields, val_y), test_ds, batch_size)

    @classmethod
    def from_data_frame(cls, df, val_idxs, y, cat_fields, batch_size, test_df=None):
        ((val_df, trn_df), (val_y, train_y)) = split_by_idx(val_idxs, df, y)
        return cls.from_data_frames(trn_df, val_df, train_y, val_y, cat_fields, batch_size, test_df=test_df)

    @property
    def get_train_loader(self):
        return self.train_dl

    @property
    def get_val_loader(self):
        return self.val_dl

    @property
    def get_test_loader(self):
        return self.test_dl

    def get_model(self, card_cat_features, n_cont, output_size, emb_drop, hidden_sizes, hidden_dropouts,
                  max_embedding_size=50, y_range=None, use_bn=False):
        """
            Generates a default model. You can use it or create your own instead.
            This model will automatically create embeddings for the cat_features
            passed in this method. All the other features will be treated as
            continuous up to n_cont
        Args:
            card_cat_features (dict): Dictionary containing the name and cardinality of each categorical features
                Ex: {'Store': 6, 'Year': 3, 'Client_type': 5}
            n_cont (int): Number of continuous fields
            output_size (int): Size of the output
            emb_drop (float): Dropout for the embeddings
            hidden_sizes (list): List of hidden layers sizes.
                Ex: [1000, 500, 200] Will create 3 hidden layers with the respective sizes
            hidden_dropouts (list): List of hidden layers dropout.
                Ex: [0.001, 0.01, 0.1] Will apply dropout to the 3 hidden layers respectively
            max_embedding_size (int): The maximum embedding sizes
            y_range:
            use_bn (bool): Use batch normalization

        Returns:
            nn.Module: The predefined model
        """
        # Maximum embedding size: https://youtu.be/5_xFdhfUnvQ?t=36m14s
        embedding_sizes = [(count, min(max_embedding_size, (count + 1) // 2)) for _, count in
                           card_cat_features.items()]

        return MixedInputModel(embedding_sizes, n_cont, emb_drop, output_size,
                               hidden_sizes, hidden_dropouts, y_range, use_bn)


class ImageClassifierShortcut(ModelData):
    @classmethod
    def from_paths(cls, root_dir: Path, preprocess_dir: Union[Path, None],
                   train_folder_name='train', val_folder_name='valid', test_folder_name=None,
                   batch_size=64, transforms=None, num_workers=os.cpu_count()):
        """
        Read in images and their labels given as sub-folder names

        Args:
            root_dir (Path): The root directory where the datasets are stored.
            preprocess_dir (Path, None): The directory where the preprocessed images will be stored or
                None to not preprocess the images.
                The preprocessing consist of converting the image files to blosc arrays so they
                are loaded from disk much faster.
            train_folder_name (str): The name of the train folder to append to the root_dir
            val_folder_name (str, None) : The name of the validation folder to append to the root_dir
            test_folder_name (str, None) : The name of the test folder to append to the root_dir
            batch_size (int): The batch_size
            transforms (torchvision.transforms.Compose): List of transformations (for data augmentation)
            num_workers (int): The number of workers for preprocessing

        Returns:

        """
        trn, val = [folder_source(root_dir, o) for o in (train_folder_name, val_folder_name)]
        # test_fnames = read_dir(path, test_name) if test_name else None
        # datasets = cls.get_ds(FilesIndexArrayDataset, trn, val, tfms, path=path, test=test_fnames)
        # return cls(path, datasets, batch_size, num_workers, classes=trn[2])


def split_by_idx(idxs, *a):
    mask = np.zeros(len(a[0]), dtype=bool)
    mask[np.array(idxs)] = True
    return [(o[mask], o[~mask]) for o in a]


def read_dirs(path, folder):
    labels, filenames, all_labels = [], [], []
    full_path = os.path.join(path, folder)
    for label in sorted(os.listdir(full_path)):
        all_labels.append(label)
        for fname in os.listdir(os.path.join(full_path, label)):
            filenames.append(os.path.join(folder, label, fname))
            labels.append(label)
    return filenames, labels, all_labels


def folder_source(path, folder):
    fnames, lbls, all_labels = read_dirs(path, folder)
    label2idx = {v:k for k,v in enumerate(all_labels)}
    idxs = [label2idx[lbl] for lbl in lbls]
    c = len(all_labels)
    label_arr = np.array(idxs, dtype=int)
    return fnames, label_arr, all_labels