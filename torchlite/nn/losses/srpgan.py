import torch
import torch.nn.functional as F
from torchlite.nn.losses.losses import CharbonnierLoss


class GeneratorLoss:
    def __init__(self):
        """
        The Generator perceptual loss
        """
        super(GeneratorLoss, self).__init__()
        self.charbonnier = CharbonnierLoss()

    def __call__(self, d_hr_out, d_sr_out, d_hr_feat_maps, d_sr_feat_maps, sr_images, target_images):
        # Adversarial loss (takes discriminator outputs)
        adversarial_loss = 0.001 * F.binary_cross_entropy(d_sr_out, torch.ones_like(d_sr_out))

        # Content loss (charbonnier between target and super resolution images)
        content_loss = self.charbonnier(sr_images, target_images, eps=1e-8)

        # Perceptual loss
        perceptual_loss = 0
        for hr_feat_map, sr_feat_map in zip(d_hr_feat_maps, d_sr_feat_maps):
            perceptual_loss += self.charbonnier(sr_feat_map, hr_feat_map, eps=1e-8)

        return adversarial_loss, content_loss, perceptual_loss
