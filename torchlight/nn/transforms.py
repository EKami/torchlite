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


class CenterCrop:
    def __init__(self, new_size):
        """
            /!\ Center cropping already exists as a torchvision transform
            Resize with center cropping
        Args:
            new_size (tuple): The size as tuple (h, w)
        """
        self.new_size = new_size

    def __call__(self, tensor):
        """
            Resize an image and keep its aspect ratio
        Args:
            tensor (ndarray): The tensor image to resize
        Returns:
            tensor: The resized/cropped tensor
        """
        largest = max(img.width, img.height)
        new_h = np.round(np.multiply(new_size[0] / largest, img.size[0])).astype(int)
        new_w = np.round(np.multiply(new_size[1] / largest, img.size[1])).astype(int)
        return img.resize((new_h, new_w), Image.ANTIALIAS)
