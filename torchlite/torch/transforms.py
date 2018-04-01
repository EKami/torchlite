"""
This package contains the methods/classes used for data transformation/augmentation.
If it doesn't contains what we are looking for it may be wise to have a look at:
https://github.com/aleju/imgaug

These classes are intended to be used with torchvision.transforms.
Typically you could do: transforms.Compose([FactorNormalize()])
"""
import os
import random
from pathlib import Path
from typing import Union
import torchlite.torch.tools.image_tools as im_tools
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torch.nn.functional as F
import Augmentor


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


class PillowAug:
    def __init__(self, operations_list: list):
        """
        A wrapper around Pillow for image augmentation

        Args:
            operations_list (list): A list of tuple. Each tuple containing
            a Pillow transformation and a probability 0 <= p <= 1. E.g: [(ImageFilter.SHARPEN, 0.5)]
        """
        self.operations_list = operations_list

    def __call__(self, image: Image):
        for op, p in self.operations_list:
            assert 0 <= p <= 1, "p should be between 0 and 1"
            s = random.uniform(0, 1)
            if p >= s:
                image = op(image)

        return image

    @staticmethod
    def gaussian_blur(radius_range):
        radius = random.uniform(*radius_range)
        imf = ImageFilter.GaussianBlur(radius)
        return lambda img: img.filter(imf)

    @staticmethod
    def sharpen(value_range):
        """
        Sharpen an image
        Args:
            value_range (tuple): A value range for contrast, will be picked uniformly

        Returns:
            callable: A callable function
        """

        def f(image):
            factor = random.uniform(*value_range)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(factor)
            return image

        return f

    @staticmethod
    def contrast(value_range):
        """
        Contrast an image
        Args:
            value_range (tuple): A value range for contrast, will be picked uniformly

        Returns:
            callable: A callable function
        """

        def f(image):
            factor = random.uniform(*value_range)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)
            return image

        return f

    @staticmethod
    def brighten(value_range):
        """
        Brighten an image
        Args:
            value_range (tuple): A value range for brightness, will be picked uniformly

        Returns:
            callable: A callable function
        """

        def f(image):
            factor = random.uniform(*value_range)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
            return image

        return f


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
