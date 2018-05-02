from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import PIL
import torchvision.transforms.functional as t_vision


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val):
        self.val = val
        self.count += 1
        self.sum += val

    @property
    def avg(self):
        return self.sum / self.count


def to_np(v):
    """
        Turn Pytorch tensor to numpy array
    Args:
        v (Tensor, PIL.Image.Image):
            Pytorch tensor or Pillow image
    Returns:
        np.ndarray: A numpy array
    """
    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    elif isinstance(v, PIL.Image.Image):
        v = np.asarray(v)
    return v


def children(module: nn.Module):
    """
        Returns a list of an torch.Module children modules
        (in other terms the list of layers of a given model)
    Args:
        module (nn.Module):
            A Pytorch module
    Returns:
        list: A list of the module children
    """
    return module if isinstance(module, (list, tuple)) else list(module.children())


def to_onehot_tensor(y: np.ndarray):
    """
    Turn a numpy array with indices to a torch onehot tensor
    Args:
        y (np.ndarray): The numpy array with indices

    Returns:
        torch.IntTensor: The onehot tensor
    """
    y_onehot = torch.IntTensor(len(y), len(np.unique(y)))
    y_onehot.zero_()
    labels_tensors = torch.unsqueeze(torch.from_numpy(y.astype(np.long)), 1)
    y_onehot.scatter_(1, labels_tensors, 1)
    return y_onehot


def normalize_batch(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Normalize a batch tensor of size () given the mean and standard deviation
    Args:
        tensor (Tensor): A Pytorch tensor
        mean (list): The mean of each channels (defaults to all torchvision pretrained models).
        std (list): The standard deviation of each channels (defaults to all torchvision pretrained models).

    Returns:
        Tensor: A pytorch tensor normalized
    """
    tensor = tensor.clone()
    for i, img in enumerate(tensor):
        tensor[i, :] = t_vision.normalize(img, mean, std)
    return tensor