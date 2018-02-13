import numpy as np
import torch


def denormalize(image: np.ndarray, std, mean, channel_type="channel_first"):
    """
        Reverse the normalization done to an image.
    Args:
        image (np.ndarray): Image matrix
        std (np.ndarray, list): Standard deviation over channels
        mean (np.ndarray, list): Mean over channels
        channel_type (str): Either channel_first or channel_last
    Returns:
        np.ndarray: The image denormalized as (Height, Width, Channels)
    """
    if channel_type == "channel_first":
        image = np.transpose(image, (1, 2, 0))

    image = image * std + mean
    image = (image * 255).astype(np.uint8)
    return image


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