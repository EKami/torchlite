from torch.utils.data import DataLoader


class ModelDataLoader:
    def __init__(self, dl): self.dl = dl

    @classmethod
    def create_dl(cls, *args, **kwargs):
        return cls(DataLoader(*args, **kwargs))

    def __iter__(self):
        self.it, self.i = iter(self.dl), 0
        return self

    def __len__(self): return len(self.dl)

    def __next__(self):
        if self.i >= len(self.dl): raise StopIteration
        self.i += 1
        return next(self.it)

    @property
    def dataset(self): return self.dl.dataset


class ModelData:
    def __init__(self, path, trn_dl, val_dl, test_dl=None):
        self.path = path
        self.trn_dl = trn_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

    @classmethod
    def from_dls(cls, path, trn_dl, val_dl, test_dl=None):
        trn_dl, val_dl = ModelDataLoader(trn_dl), ModelDataLoader(val_dl)
        if test_dl:
            test_dl = ModelDataLoader(test_dl)
        return cls(path, trn_dl, val_dl, test_dl)

    @property
    def is_reg(self): return self.trn_ds.is_reg

    @property
    def trn_ds(self): return self.trn_dl.dataset

    @property
    def val_ds(self): return self.val_dl.dataset

    @property
    def test_ds(self): return self.test_dl.dataset

    @property
    def trn_y(self): return self.trn_ds.y

    @property
    def val_y(self): return self.val_ds.y