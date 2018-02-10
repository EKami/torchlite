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
        self.mode = 0
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

        if self.mode == 0:
            return lr_image, hr_image
        else:
            hr_original_image = Image.open(self.hr_image_filenames[index])
            hr_transform = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor()
            ])
            hr_original_image = hr_transform(hr_original_image)
            return lr_image, hr_image, hr_original_image

    def __len__(self):
        return len(self.hr_image_filenames)

    def set_mode(self, mode):
        """
            Mode, either 0 or 1.
            Mode 0: the __getitem__ returns (lr_image, hr_image)
            Mode 1: the __getitem__ returns (lr_image, hr_image, hr_original_image)
        Args:
            mode (int): Mode value
        """
        self.mode = mode


class AdvValDataset(Dataset):
    def __init__(self, hr_image_filenames, upscale_factor):
        super(AdvValDataset, self).__init__()
        self.upscale_factor = upscale_factor
        self.hr_image_filenames = hr_image_filenames

    def __getitem__(self, index):
        hr_original_image = Image.open(self.hr_image_filenames[index])
        w, h = hr_original_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)

        lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        hr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(crop_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        hr_original_transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ])

        lr_image = lr_transform(hr_original_image)
        hr_img = hr_transform(lr_image)
        hr_original_image = hr_original_transform(hr_original_image)
        return lr_image, hr_img, hr_original_image

    def __len__(self):
        return len(self.hr_image_filenames)
