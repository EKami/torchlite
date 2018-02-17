import torch
import torch.nn as nn
from torchlight.nn.losses.losses import TVLoss
from torchvision.models.vgg import vgg16
from torchlight.nn.models.models import FinetunedModelTools


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
        vgg = vgg16(pretrained=True)
        vgg_network = nn.Sequential(*FinetunedModelTools.freeze(vgg.features)).eval()
        if use_cuda:
            vgg_network.cuda()
        self.vgg_network = vgg_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()  # Total variation loss (not included in the original paper)

    def __call__(self, out_labels, gen_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(-torch.log(out_labels))
        # Perception Loss
        vgg_loss = self.mse_loss(self.vgg_network(gen_images), self.vgg_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(gen_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(gen_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * vgg_loss + 2e-8 * tv_loss

