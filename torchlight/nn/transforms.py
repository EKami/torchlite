"""
This package contains the methods/classes used for data transformation/augmentation.
If it doesn't contains what we are looking for it may be wise to have a look at:
https://github.com/aleju/imgaug

These classes are intended to be used with torchvision.transforms.
Typically you could do: transforms.Compose([FactorNormalize()])
"""
from typing import Union
import cv2
import numpy as np
import torch.nn.functional as F


class FactorNormalize:
    """Normalize a tensor image given a factor

    Args:
        factor (float): A factor to normalize the image
    """

    def __init__(self, factor: float=1./255):
        self.factor = factor

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor = F.mul(tensor, self.factor)
        return tensor
