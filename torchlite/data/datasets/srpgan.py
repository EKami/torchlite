from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchlite.torch.transforms import PillowAug
from PIL import Image


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


class TrainDataset(Dataset):
    def __init__(self, hr_image_filenames: list, crop_size, upscale_factor, random_augmentations=True):
        """
        The train dataset for SRPGAN.
        The dataset takes one unique list of files
        Args:
            hr_image_filenames (list): The HR images filename
            crop_size (int): Size of the crop
            upscale_factor (int): The upscale factor, either 2, 4 or 8
            random_augmentations (bool): True if the images need to be randomly augmented, False otherwise
        """

        self.hr_image_filenames = hr_image_filenames
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size) if random_augmentations else transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),  # Is normalized in the range [0, 1]
        ])
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.crop_size // upscale_factor, interpolation=Image.BICUBIC),
            PillowAug([
                (PillowAug.brighten((0.7, 1.3)), 0.5),
                (PillowAug.contrast((0.6, 1.4)), 0.5),
                (PillowAug.sharpen((0.6, 1.2)), 0.3),
                (PillowAug.gaussian_blur((1, 2)), 0.6)
            ]) if random_augmentations else lambda x: x,
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(self.hr_image_filenames[index])
        assert img.height >= self.crop_size and img.width >= self.crop_size, \
            "Image {} too little for crop_size".format(self.hr_image_filenames[index])
        hr_image = self.hr_transform(img)
        lr_image = self.lr_transform(hr_image.clone())

        # ---- Used to check the transformations (Uncomment to test)
        # # HR save
        # transforms.Compose([
        #     ttransforms.ImgSaver("/tmp/images/" + str(index) + "/hr_img.png")])(hr_image.clone())
        # # AUG save
        # transforms.Compose([
        #     transforms.ToPILImage(),
        #
        #     PillowAug([
        #         (PillowAug.gaussian_blur((1, 1)), 1.0),
        #     ]),
        #
        #     ttransforms.ImgSaver("/tmp/images/" + str(index) + "/aug_img.png")])(hr_image.clone())
        # # LR save
        # transforms.Compose([
        #     ttransforms.ImgSaver("/tmp/images/" + str(index) + "/lr_img.png")])(lr_image.clone())

        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_image_filenames)


class VggTransformDataset(Dataset):
    def __init__(self, images_batch):
        """
        This dataset receive a batch of images and apply a transformation on them
        Args:
            images_batch (Tensor): A Pytorch tensor of size (batch_size, C, H, W)
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
        image = self.images[index]
        # Check if the image is 4 channels wide

        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            res_img = Image.new("RGB", image.size, (255, 255, 255))
            res_img.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        else:
            res_img = image

        image = self.tfs(res_img)
        return image, image

    def __len__(self):
        return len(self.images)
