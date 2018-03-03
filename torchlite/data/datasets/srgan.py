from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchlite.nn.transforms as ttransforms
from PIL import Image
from imgaug import augmenters as iaa


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


class TrainDataset(Dataset):
    def __init__(self, hr_image_filenames: list, crop_size, upscale_factor, random_augmentations=True):
        """
        The train dataset for SRGAN.
        The dataset takes one unique list of files
        Args:
            hr_image_filenames (list): The HR images filename
            crop_size (int): Size of the crop
            upscale_factor (int): The upscale factor, either 2, 4 or 8
            random_augmentations (bool): True if the images need to be randomly augmented, False otherwise
        """
        # Imgaug augmentations
        rarely = lambda aug: iaa.Sometimes(0.1, aug)
        sometimes = lambda aug: iaa.Sometimes(0.25, aug)

        self.hr_image_filenames = hr_image_filenames
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size) if random_augmentations else transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),  # Is normalized in the range [0, 1]
        ])
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.crop_size // upscale_factor, interpolation=Image.BICUBIC),
            ttransforms.ImgAugWrapper([
                rarely(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
                rarely(iaa.GaussianBlur((0, 2.0))),
                rarely(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
                rarely(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5)),
            ]) if random_augmentations else lambda x: x,
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.hr_image_filenames[index]))
        lr_image = self.lr_transform(hr_image.clone())

        # ---- Used to check the transformations (Uncomment to test)
        # # HR save
        # transforms.Compose([
        #     ttransforms.ImgSaver("/tmp/images/" + str(index) + "/hr_img.png")])(hr_image.clone())
        # # AUG save
        # transforms.Compose([
        #     transforms.ToPILImage(),
        #
        #     ttransforms.ImgAugWrapper([
        #         # Put your test image transformations here
        #         iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))
        #     ]),
        #
        #     ttransforms.ImgSaver("/tmp/images/" + str(index) + "/aug_img.png")])(hr_image.clone())
        # # LR save
        # transforms.Compose([
        #     ttransforms.ImgSaver("/tmp/images/" + str(index) + "/lr_img.png")])(lr_image.clone())

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
