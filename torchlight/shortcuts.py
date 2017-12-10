"""
    This class contains shortcuts to automatically fit a particular kind of dataset
    in them
"""
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from data.loaders import ModelData
from data.datasets import PassthruDataset, ColumnarDataset
from nn.models import MixedInputModel, StructuredModel
from utils import tools

import numpy as np


class ColumnarShortcut(ModelData):
    def __init__(self, path, batch_size, trn_ds, val_ds=None, test_ds=None, shuffle=True):
        """
        A shortcut used for structured/columnar data
        Args:
            path (str):
            batch_size (int): The batch size for the training
            trn_ds (Dataset): The train dataset
            val_ds (Dataset): The validation dataset
            test_ds (Dataset): The test dataset
            shuffle (bool): If True shuffle the training set
        """
        train_dl = DataLoader(trn_ds, batch_size, shuffle=shuffle)
        val_dl = DataLoader(val_ds, batch_size, shuffle=False) if val_ds else None
        test_dl = DataLoader(test_ds, batch_size, shuffle=False) if test_ds else None
        super().__init__(path, train_dl, val_dl, test_dl)

    @classmethod
    def from_arrays(cls, path, val_idxs, xs, y, bs=64, test_xs=None, shuffle=True):
        ((val_xs, trn_xs), (val_y, trn_y)) = split_by_idx(val_idxs, xs, y)
        test_ds = PassthruDataset(*test_xs.T, [0] * len(test_xs)) if test_xs is not None else None
        return cls(path, PassthruDataset(*trn_xs.T, trn_y), PassthruDataset(*val_xs.T, val_y),
                   bs=bs, shuffle=shuffle, test_ds=test_ds)

    @classmethod
    def from_data_frames(cls, path, trn_df, val_df, trn_y, val_y, cat_flds, bs, test_df=None):
        test_ds = ColumnarDataset.from_data_frame(test_df, cat_flds) if test_df is not None else None
        return cls(path, ColumnarDataset.from_data_frame(trn_df, cat_flds, trn_y),
                   ColumnarDataset.from_data_frame(val_df, cat_flds, val_y), bs, test_ds=test_ds)

    @classmethod
    def from_data_frame(cls, path, val_idxs, df, y, cat_flds, bs, test_df=None):
        ((val_df, trn_df), (val_y, trn_y)) = split_by_idx(val_idxs, df, y)
        return cls.from_data_frames(path, trn_df, val_df, trn_y, val_y, cat_flds, bs, test_df=test_df)

    def get_classifier(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops, opt_fn=optim.Adam,
                       y_range=None, use_bn=False, **kwargs):
        """
        Generates a default classifier. You can use it or create your own instead
        Args:
            emb_szs:
            n_cont:
            emb_drop:
            out_sz:
            szs:
            drops:
            opt_fn:
            y_range:
            use_bn:
            **kwargs:

        Returns:

        """
        # Change this to Classifier
        model = MixedInputModel(emb_szs, n_cont, emb_drop, out_sz, szs, drops, y_range, use_bn)
        return StructuredLearner(self, StructuredModel(tools.to_gpu(model)), opt_fn, **kwargs)


def split_by_idx(idxs, *a):
    mask = np.zeros(len(a[0]), dtype=bool)
    mask[np.array(idxs)] = True
    return [(o[mask], o[~mask]) for o in a]
