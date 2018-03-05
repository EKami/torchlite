import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchlight.nn.transforms as ttransforms
from PIL import Image


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


class TrainDataset(Dataset):
    def __init__(self, hr_image_filenames: list, crop_size, upscale_factor):
        """
        The train dataset for SRPGAN.
        The dataset takes one unique list of files
        Args:
            hr_image_filenames (list): The HR images filename
        """
        self.hr_image_filenames = hr_image_filenames
        # http://pillow.readthedocs.io/en/latest/reference/ImageFilter.html
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            transforms.ToTensor(),
        ])
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.crop_size // upscale_factor, interpolation=Image.BICUBIC),
            # TODO augment the training data in the following ways:
            # (1) Random Rotation: Randomly rotate the images by 90 or 180 degrees.
            # (2) Brightness adjusting: Randomly adjust the brightness of the images.
            # (3) Saturation adjusting: Randomly adjust the saturation of the images
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.hr_image_filenames[index]))
        lr_image = self.lr_transform(hr_image)

        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_image_filenames)


class EvalDataset(Dataset):
    def __init__(self, images_filenames):
        self.images_filenames = images_filenames

    def __getitem__(self, index):
        image = Image.open(self.images_filenames[index])
        tfs = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = tfs(image)
        return image, image

    def get_file_from_index(self, index):
        path, file = os.path.split(self.images_filenames[index])
        return file

    def __len__(self):
        return len(self.images_filenames)
