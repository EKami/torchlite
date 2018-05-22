"""
    This class contains models shortcuts to automatically fit a particular kind of dataset
    in them. The serve as the "default way to go" when you train for a particular dataset
    but don't want to spend time creating the architecture of a model.
"""
import os
import numpy as np
import torch.nn as nn
from typing import Union
import torchvision
from torch.utils.data import Dataset, DataLoader

import torchlite.data.files as tfiles
from torchlite.data.datasets import ColumnarDataset, ImageClassificationDataset
from torchlite.torch.models import TabularModel, FinetunedConvModel
from torchlite.torch.tools import tensor_tools


class BaseLoader:
    def __init__(self, train_ds, val_ds, batch_size, shuffle, test_ds=None, num_workers=os.cpu_count()):
        self.train_dl = DataLoader(train_ds, batch_size, shuffle=shuffle, num_workers=num_workers)
        self.val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers) if val_ds else None
        self.test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers) if test_ds else None

    @property
    def get_train_loader(self):
        return self.train_dl

    @property
    def get_val_loader(self):
        return self.val_dl

    @property
    def get_test_loader(self):
        return self.test_dl


class ColumnarShortcut(BaseLoader):
    def __init__(self, train_ds, val_ds=None, test_ds=None, batch_size=64, shuffle=True):
        """
        A shortcut used for structured/columnar data
        Args:
            train_ds (Dataset): The train dataset
            val_ds (Dataset): The validation dataset
            test_ds (Dataset): The test dataset
            batch_size (int): The batch size for the training
            shuffle (bool): If True shuffle the training set
        """
        super().__init__(train_ds, val_ds, batch_size, shuffle, test_ds)

    @classmethod
    def from_data_frames(cls, train_df, val_df, y_field, cat_fields, batch_size, test_df=None):
        """
        Create a columnar shortcut from DataFrames.
        Args:
            train_df (DataFrame): The train DataFrame
            val_df (DataFrame, None): The test DataFrame
            y_field (str): The dependant field. This field will be removed from training
            cat_fields (list): List of categorical fields
            batch_size (int): Batch size
            test_df (DataFrame, None): The test DataFrame

        Returns:
            ColumnarShortcut: A ColumnarShortcut object
        """
        y_train = train_df[y_field]
        train_df.drop(y_field, axis=1, inplace=True)
        train_ds = ColumnarDataset.from_data_frame(train_df, cat_fields, y_train)
        if val_df is not None:
            y_val = val_df[y_field]
            val_df.drop(y_field, axis=1, inplace=True)
            val_ds = ColumnarDataset.from_data_frame(val_df, cat_fields, y_val)
        else:
            val_ds = None
        test_ds = ColumnarDataset.from_data_frame(test_df, cat_fields) if test_df is not None else None
        return cls(train_ds, val_ds, test_ds, batch_size)

    def get_stationary_model(self, card_cat_features, n_cont, output_size, emb_drop, hidden_sizes, hidden_dropouts,
                             max_embedding_size=50, y_range=None, use_bn=False):
        """
            Generates a default model. You can use it or create your own instead.
            This model will automatically create embeddings for the cat_features
            passed in this method. All the other features will be treated as
            continuous up to n_cont.

            /!\ This model is useful only for data which were turned stationary. Otherwise it will give
            very bad results.
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
            y_range (tuple): The range in which y must fit
            use_bn (bool): Use batch normalization

        Returns:
            nn.Module: The predefined model
        """
        # Maximum embedding size: https://youtu.be/5_xFdhfUnvQ?t=36m14s
        embedding_sizes = [(count, min(max_embedding_size, (count + 1) // 2)) for _, count in
                           card_cat_features.items()]

        return TabularModel(embedding_sizes, n_cont, emb_drop, output_size,
                            hidden_sizes, hidden_dropouts, y_range, use_bn)


class ImageClassifierShortcut(BaseLoader):
    def __init__(self, train_ds, val_ds=None, test_ds=None, y_mapping=None, batch_size=64, shuffle=True):
        """
        A shortcut used for image data
        Args:
            train_ds (Dataset): The train dataset
            val_ds (Dataset): The validation dataset
            test_ds (Dataset): The test dataset
            y_mapping (dict): Mapping between the labels and the indexes
            batch_size (int): The batch size for the training
            shuffle (bool): If True shuffle the training set
        """
        self.y_mapping = y_mapping
        super().__init__(train_ds, val_ds, batch_size, shuffle, test_ds)

    @classmethod
    def from_paths(cls, train_folder: str, val_folder: Union[str, None], test_folder: Union[str, None] = None,
                   batch_size=64, transforms=None):
        """
        Read in images and their labels given as sub-folder names

        Args:
            train_folder (str): The path to the train folder
            val_folder (str, None): The path to the validation folder
            test_folder (str, None): The path to the test folder
            batch_size (int): The batch_size
            transforms (torchvision.transforms.Compose): List of transformations (for data augmentation)

        Returns:
            ImageClassifierShortcut: A ImageClassifierShortcut object
        """
        datasets = []

        files, y_mapping = tfiles.get_labels_from_folders(train_folder)
        datasets.append(ImageClassificationDataset(files[:, 0], files[:, 1], transforms=transforms))

        if val_folder:
            files, _ = tfiles.get_labels_from_folders(val_folder, y_mapping)
            datasets.append(ImageClassificationDataset(files[:, 0], files[:, 1], transforms=transforms))
        else:
            datasets.append(None)

        if test_folder:
            files = tfiles.get_files(test_folder)
            datasets.append(ImageClassificationDataset(files, np.repeat(-1, len(files)), transforms=transforms))
        else:
            datasets.append(None)

        return cls(datasets[0], datasets[1], datasets[2], y_mapping, batch_size)

    @property
    def get_y_mapping(self):
        """
            The mapping between labels and indexes
        Returns:
            dict: The {index: label} mapping
        """
        mapping = {v: k for k, v in self.y_mapping.items()}
        return mapping

    def get_resnet_model(self):
        resnet = torchvision.models.resnet34(pretrained=True)
        # Take the head of resnet up until AdaptiveAvgPool2d
        resnet_head = tensor_tools.children(resnet)[:-2]
        net = FinetunedConvModel(resnet_head, nn.LogSoftmax(dim=1))
        return net
