from typing import Union
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


class TrainDataset(Dataset):
    def __init__(self, hr_image_filenames: list, lr_image_filenames: Union[list, None], crop_size, upscale_factor):
        """
        The train dataset for SRGAN.
        The dataset takes one unique list of files
        Args:
            hr_image_filenames (list): The HR images filename
            lr_image_filenames (list, None): The LR images. If None then the HR images will be bicubic resized
                according to crop_size and upscale_factor
        """
        self.lr_image_filenames = lr_image_filenames
        self.hr_image_filenames = hr_image_filenames
        if not self.lr_image_filenames:
            self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
            self.hr_transform = transforms.Compose([
                transforms.RandomCrop(self.crop_size),
                transforms.ToTensor(),
            ])
            self.lr_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.crop_size // upscale_factor, interpolation=Image.BICUBIC),
                transforms.ToTensor()
            ])
        else:
            # TODO finish when the lr_image_filenames is given
            assert len(self.lr_image_filenames) == len(self.hr_image_filenames)
            self.crop_size = None
            self.hr_transform = None
            self.lr_transform = None

    def __getitem__(self, index):
        if not self.lr_image_filenames:
            hr_image = self.hr_transform(Image.open(self.hr_image_filenames[index]))
            lr_image = self.lr_transform(hr_image)
        else:
            hr_image = self.hr_transform(Image.open(self.hr_image_filenames[index]))
            lr_image = self.lr_transform(Image.open(self.lr_image_filenames[index]))

        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_image_filenames)


class ValDataset(Dataset):
    def __init__(self, hr_image_filenames, crop_size, upscale_factor):
        super(ValDataset, self).__init__()
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.upscale_factor = upscale_factor
        self.hr_image_filenames = hr_image_filenames

    def __getitem__(self, index):
        """
        Returns the low resolution image, the upscaled lr image and the
        original hr_image. The goal being to compare the upscaled bicubic lr image
        to the original hr image.
        Args:
            index (int): The image index
        Returns:
            tuple: (lr_image, lr_upscaled_image, hr_original_image)
        """
        hr_original_image = Image.open(self.hr_image_filenames[index])
        w, h = hr_original_image.size
        assert min(w, h) >= self.crop_size, \
            "Your crop size is too low for validation, an image with lower dimensions has been detected. " \
            "Either change/remove your validation set or lower the crop size."

        hr_original_transform = transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor()
        ])

        lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.crop_size // self.upscale_factor, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

        lr_upscale_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.crop_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

        hr_original_image = hr_original_transform(hr_original_image)
        lr_image = lr_transform(hr_original_image)
        lr_upscaled_image = lr_upscale_transform(lr_image)
        return lr_image, lr_upscaled_image, hr_original_image

    def __len__(self):
        return len(self.hr_image_filenames)
