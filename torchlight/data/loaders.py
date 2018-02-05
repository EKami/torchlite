from torch.utils.data import DataLoader
import os


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
