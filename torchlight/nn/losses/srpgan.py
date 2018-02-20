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

    def __call__(self, d_sr_out, sr_images, target_images):


        return None

