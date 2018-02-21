import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlite.nn.losses.losses import TVLoss
from torchvision.models.vgg import vgg19
from torchlite.nn.models.models import FinetunedModelTools
from torchlite.data.datasets.srgan import VggTransformDataset
from torch.utils.data import DataLoader
import torchlite.nn.tools.tensor_tools as ttools
from torch.autograd import Variable


class PerceptualLoss:
    def __init__(self, use_cuda=True, num_workers=os.cpu_count()):
        """
        The Generator perceptual loss
        Args:
            use_cuda (bool): If True moves the model onto the GPU.
            /!\ If the model is on the GPU the PerceptualLoss should be on the GPU too
            num_workers (int):  Number of process for preprocessing the vgg images
        """
        super(PerceptualLoss, self).__init__()

        self.num_workers = num_workers
        if use_cuda:
            if torch.cuda.is_available():
                self.use_cuda = True
            else:
                self.use_cuda = False
                print("/!\ Warning: Cuda set but not available, using CPU...")
        # TODO try VGG54
        vgg = vgg19(pretrained=True)
        # Don't take the classifier block
        vgg_network = nn.Sequential(*FinetunedModelTools.freeze(ttools.children(vgg.features))).eval()
        if self.use_cuda:
            vgg_network.cuda()
        self.vgg_network = vgg_network
        self.mse_loss = nn.MSELoss()
        # TODO add this loss
        self.tv_loss = TVLoss()  # Total variation loss (not included in the original paper)

    def __call__(self, d_sr_out, sr_images, target_images):
        # Adversarial Loss
        #adversarial_loss = 1e-3 * F.binary_cross_entropy(d_sr_out, torch.ones_like(d_sr_out))
        adversarial_loss = 1e-3 * torch.sum(-torch.log(d_sr_out))

        # MSE Loss
        mse_loss = self.mse_loss(sr_images, target_images)

        # Content Loss
        sr_images_vgg = next(iter(DataLoader(VggTransformDataset(sr_images.data.cpu()), sr_images.size(0),
                                             shuffle=False, num_workers=self.num_workers)))
        hr_images_vgg = next(iter(DataLoader(VggTransformDataset(target_images.data.cpu()), target_images.size(0),
                                             shuffle=False, num_workers=self.num_workers)))

        if self.use_cuda:
            sr_images_vgg = sr_images_vgg.cuda()
            hr_images_vgg = hr_images_vgg.cuda()

        vgg_sr_out = self.vgg_network(Variable(sr_images_vgg))
        vgg_hr_out = self.vgg_network(Variable(hr_images_vgg))
        vgg_loss = 0.006 * self.mse_loss(vgg_sr_out, vgg_hr_out)

        return mse_loss, adversarial_loss, vgg_loss
