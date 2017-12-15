import torch
from torch.autograd import Variable
import numpy as np


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

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
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


def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())
