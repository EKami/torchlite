import torch
import torch.nn as nn
from torchlight.nn.losses.losses import TVLoss
from torchvision.models.vgg import vgg16
from torchlight.nn.models.models import FinetunedModelTools


class GeneratorLoss:
    def __init__(self, use_cuda=True):
        """
        The Generator loss
        Args:
            use_cuda (bool): If True moves the model onto the GPU.
            /!\ If the model is on the GPU the GeneratorLoss should be on the GPU too
        """
        super(GeneratorLoss, self).__init__()

        if use_cuda:
            if torch.cuda.is_available():
                self.use_cuda = True
            else:
                print("/!\ Warning: Cuda set but not available, using CPU...")
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*FinetunedModelTools.freeze(vgg.features)).eval()
        if use_cuda:
            loss_network.cuda()
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()  # Total variation loss (not included in the original paper)

    def __call__(self, out_labels, gen_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(gen_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(gen_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(gen_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss

