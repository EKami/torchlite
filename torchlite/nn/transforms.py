"""
This package contains the methods/classes used for data transformation/augmentation.
If it doesn't contains what we are looking for it may be wise to have a look at:
https://github.com/aleju/imgaug

These classes are intended to be used with torchvision.transforms.
Typically you could do: transforms.Compose([FactorNormalize()])
"""
from typing import Union
import cv2
import torch
import numpy as np
import random
from PIL import Image, ImageFilter
import torch.nn.functional as F


class FactorNormalize:
    """Normalize a tensor image given a factor

    Args:
        factor (float): A factor to normalize the image
    """

    def __init__(self, factor: float = 1. / 255):
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


class RandomSmooth:
    def __init__(self, active_range: float=0.3):
        """
        Add random smooth filters

        Args:
            active_range (float): A number between 0 and 1 which is used to know how
            often the filter is activated. E.g with active_range=0.3
        """
        self.active_range = active_range

    def __call__(self, image: Image):
        """
        Add random smooth filters
        Args:
            image (Image): A Pillow image
        Returns:
            Image: The pillow transformed image (or not)
        """
        rnd = random.random()
        if rnd <= self.active_range:
            image = image.filter(ImageFilter.SMOOTH)
        return image