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


class AdvLoss:
    def __init__(self):
        self.g_loss = None

    def __call__(self, outputs, target_images):
        # https://github.com/leftthomas/SRGAN/blob/master/train.py#L88
        gen_imgs = outputs[0]  # Generated images from the Generator
        d_real_out = outputs[1]  # Real outputs from the discriminator
        d_fake_out = outputs[2]  # Fake outputs from the discriminator

        self.g_loss = GeneratorLoss()(d_fake_out, gen_imgs, target_images)
        self.d_loss = 1 - d_real_out + d_fake_out
        return {"g_loss": self.g_loss.data[0], "d_loss": self.d_loss.data[0]}

