import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg_mom = 0.98
        self.avg_loss_mom = 0.

    def reset(self):
        self.__init__()

    def update(self, val):
        self.val = val
        self.count += 1
        self.sum += val * self.count
        self.avg_loss_mom = self.avg_loss_mom * self.avg_mom + val * (1 - self.avg_mom)

    @property
    def avg(self):
        return self.sum / self.count

    @property
    def debias_loss(self):
        # More info: https://youtu.be/J99NV9Cr75I?t=2h4m
        return self.avg_loss_mom / (1 - self.avg_mom ** self.count)


def to_np(v):
    """

    Args:
        v (Variable, Tensor): Pytorch Variable/Tensor

    Returns:
        np.ndarray: A numpy array
    """
    if isinstance(v, Variable):
        v = v.data
    return v.cpu().numpy()


def to_gpu(x, *args, **kwargs):
    """
    Moves torch tensor to gpu if possible

    Returns:
        torch.Tensor: Moved to the GPU or not
    """
    return x.cuda(*args, **kwargs) if torch.cuda.is_available() else x


def to_csv(test_file, output_file, identifier_field, predicted_field,
           predictions, read_format='csv'):
    df = None
    if read_format == 'csv':
        df = pd.read_csv(test_file)
    elif read_format == 'feather':
        df = pd.read_feather(test_file)
    df = df[[identifier_field]]
    df[predicted_field] = predictions
    df.to_csv(output_file, index=False)


def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())
