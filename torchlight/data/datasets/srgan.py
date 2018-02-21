import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


class TrainDataset(Dataset):
    def __init__(self, hr_image_filenames: list, crop_size, upscale_factor):
        """
        The train dataset for SRGAN.
        The dataset takes one unique list of files
        Args:
            hr_image_filenames (list): The HR images filename
        """
        self.hr_image_filenames = hr_image_filenames
        # http://pillow.readthedocs.io/en/latest/reference/ImageFilter.html
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            transforms.ToTensor()  # Is normalized in the range [0, 1]
        ])
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.crop_size // upscale_factor, interpolation=Image.BICUBIC),
            # TODO augment the training data in the following ways:
            # (1) Random Rotation: Randomly rotate the images by 90 or 180 degrees.
            # (2) Brightness adjusting: Randomly adjust the brightness of the images.
            # (3) Saturation adjusting: Randomly adjust the saturation of the images
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.hr_image_filenames[index]))
        lr_image = self.lr_transform(hr_image)

        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_image_filenames)


class EvalDataset(Dataset):
    def __init__(self, images):
        """
        The evaluation dataset
        Args:
            images (list): A list of Pillow images
        """
        self.images = images
        self.tfs = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = self.tfs(self.images[index])
        return image, image

    def __len__(self):
        return len(self.images)


class VggTransformDataset(Dataset):
    def __init__(self, images_batch):
        """
        This dataset receive a batch of images and apply a transformation on them
        Args:
            images_batch (Tensor, Variable): A pytorch tensor of size (batch_size, C, H, W)
        """
        self.images_batch = images_batch.clone()
        self.vgg_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        res = self.vgg_transforms(self.images_batch[index])
        return res

    def __len__(self):
        return len(self.images_batch)
