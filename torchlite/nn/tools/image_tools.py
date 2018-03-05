import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def denormalize(tensor, mean, std):
    """
        Denormalize a tensor image with mean and standard deviation.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channels.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def image_to_tensor(image, mean=0, std=1.):
    """
    Transforms an image to a tensor and eventually normalize it
    Args:
        image (np.ndarray): A RGB array image
        mean: The mean of the image values
        std: The standard deviation of the image values
    Returns:
        tensor: A Pytorch tensor
    """
    image = image.astype(np.float32)
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image)
    return tensor


def save_tensor_as_png(image, to_file):
    """
    Takes a Pytorch tensor or a Pillow Image and save it as the file given in
    parameter
    Args:
        image (Tensor, Image): A Pytorch tensor or a Pillow Image
        to_file (str): The path to the output file
    Returns:

    """
    if not isinstance(image, Image.Image):
        image = transforms.ToPILImage()(image.cpu())
    image.save(to_file, "PNG")
