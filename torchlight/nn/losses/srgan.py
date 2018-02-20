import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight.nn.losses.losses import TVLoss
from torchvision.models.vgg import vgg19
from torchlight.nn.models.models import FinetunedModelTools
import torchlight.nn.tools.tensor_tools as ttools


class PerceptualLoss:
    def __init__(self, use_cuda=True):
        """
        The Generator perceptual loss
        Args:
            use_cuda (bool): If True moves the model onto the GPU.
            /!\ If the model is on the GPU the PerceptualLoss should be on the GPU too
        """
        super(PerceptualLoss, self).__init__()

        if use_cuda:
            if torch.cuda.is_available():
                self.use_cuda = True
            else:
                print("/!\ Warning: Cuda set but not available, using CPU...")
        vgg = vgg19(pretrained=True)
        vgg_network = nn.Sequential(*FinetunedModelTools.freeze(ttools.children(vgg.features))).eval()
        if use_cuda:
            vgg_network.cuda()
        self.vgg_network = vgg_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()  # Total variation loss (not included in the original paper)

    def __call__(self, d_sr_out, sr_images, target_images):
        # Adversarial Loss
        adversarial_loss = 1e-3 * F.binary_cross_entropy(d_sr_out, torch.ones_like(d_sr_out))
        # Image Loss
        mse_loss = self.mse_loss(sr_images, target_images)

        # sr_images = ttools.normalize_batch(sr_images)
        # target_images = ttools.normalize_batch(target_images)

        # Perception Loss
        vgg_loss = 2e-6 * self.mse_loss(self.vgg_network(sr_images), self.vgg_network(target_images))

        return mse_loss, adversarial_loss, vgg_loss
