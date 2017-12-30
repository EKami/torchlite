"""
    This class contains models shortcuts to automatically fit a particular kind of dataset
    in them. The serve as the "default way to go" when you train for a particular dataset
    but don't want to spend time creating the architecture of a model.
"""
from torch.utils.data import DataLoader, Dataset
from loaders import ModelData
from datasets import PassthruDataset, ColumnarDataset
from nn.models import MixedInputModel
import torch.nn as nn

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
    def from_arrays(cls, val_idxs, xs, y, batch_size=64, test_xs=None, shuffle=True):
        ((val_xs, trn_xs), (val_y, trn_y)) = split_by_idx(val_idxs, xs, y)
        test_ds = PassthruDataset(*test_xs.T, [0] * len(test_xs)) if test_xs is not None else None
        return cls(PassthruDataset(*trn_xs.T, trn_y), PassthruDataset(*val_xs.T, val_y),
                   test_ds, batch_size, shuffle)

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

    def get_model(self, card_cat_features, n_cont, emb_drop, output_size, hidden_sizes, hidden_dropout,
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
            emb_drop (float): Dropout for the embeddings
            output_size (int): Size of the output
            hidden_sizes (list): List of hidden layers sizes.
                Ex: [1000, 500, 200] Will create 3 hidden layers with the respective sizes
            hidden_dropout (list): List of hidden layers dropout.
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
                               hidden_sizes, hidden_dropout, y_range, use_bn)


def split_by_idx(idxs, *a):
    mask = np.zeros(len(a[0]), dtype=bool)
    mask[np.array(idxs)] = True
    return [(o[mask], o[~mask]) for o in a]
