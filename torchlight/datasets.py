from torch.utils.data import Dataset
import numpy as np


class PassthruDataset(Dataset):
    def __init__(self, *args):
        *xs, y = args
        self.xs, self.y = xs, y

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return [o[idx] for o in self.xs] + [self.y[idx]]

    @classmethod
    def from_data_frame(cls, df, cols_x, col_y):
        cols = [df[o] for o in cols_x + [col_y]]
        return cls(*cols)


class ColumnarDataset(Dataset):
    def __init__(self, cats, conts, y):
        n = len(cats[0]) if cats else len(conts[0])
        self.cats = np.stack(cats, 1).astype(np.int64) if cats else np.zeros((n, 1))
        self.conts = np.stack(conts, 1).astype(np.float32) if conts else np.zeros((n, 1))
        self.y = np.zeros((n, 1)) if y is None else y[:, None]

    def __len__(self): return len(self.y)

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
