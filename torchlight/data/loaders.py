from torch.utils.data import DataLoader


class BaseLoader:
    def __init__(self, train_ds, val_ds, batch_size, shuffle, test_ds=None):
        self.train_dl = DataLoader(train_ds, batch_size, shuffle=shuffle)
        self.val_dl = DataLoader(val_ds, batch_size, shuffle=False) if val_ds else None
        self.test_dl = DataLoader(test_ds, batch_size, shuffle=False) if test_ds else None

    @property
    def get_train_loader(self):
        return self.train_dl

    @property
    def get_val_loader(self):
        return self.val_dl

    @property
    def get_test_loader(self):
        return self.test_dl

    def freeze_to(self, n):
        c=self.get_layer_groups()
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)
