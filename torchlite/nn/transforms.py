"""
This package contains the methods/classes used for data transformation/augmentation.
If it doesn't contains what we are looking for it may be wise to have a look at:
https://github.com/aleju/imgaug

These classes are intended to be used with torchvision.transforms.
Typically you could do: transforms.Compose([FactorNormalize()])
"""
import os
import random
import numpy as np
from pathlib import Path
from typing import Union
import torchlite.nn.tools.image_tools as im_tools
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
import Augmentor
from imgaug import augmenters as iaa


class AugmentorWrapper:
    def __init__(self, operations_list: list):
        """
        A wrapper around Augmentor: https://github.com/mdbloice/Augmentor
        Args:
            operations_list (list): A list of Augmentor.Operations
        """
        self.p = Augmentor.Pipeline()
        for operation in operations_list:
            self.p.add_operation(operation)

    def __call__(self, image: Image):
        return self.p.torch_transform


class ImgAugWrapper:
    def __init__(self, operations_list: list):
        """
            A wrapper around Augmentor: https://github.com/aleju/imgaug
            Args:
                operations_list (list): A list of imgaug transformations
        """
        self.seq = iaa.Sequential(operations_list)

    def __call__(self, image: Image):
        image_arr = np.array(image)
        image_aug = self.seq.augment_images([image_arr])
        return Image.fromarray(*image_aug)


class ImgSaver:
    def __init__(self, to_file):
        """
        Save an image to disk

        Args:
            to_file (str): Path where to save the image (the directories will be created if they doesn't exist)
        """
        self.to_file = Path(to_file)
        if not os.path.exists(self.to_file.parent):
            os.makedirs(self.to_file.parent)

    def __call__(self, image: Union[Image.Image, torch.Tensor]):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        image.save(self.to_file, "png")


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
    def __init__(self, active_range: float = 0.3):
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


class Denormalize:
    def __init__(self, mean, std):
        """
            Denormalize a Tensor

        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channels.
        """
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return im_tools.denormalize(tensor, self.mean, self.std)
