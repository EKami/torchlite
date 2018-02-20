import torch
from torchlight.nn.losses.losses import CharbonnierLoss, L1CharbonnierLoss


class GeneratorLoss:
    def __init__(self):
        """
        The Generator perceptual loss
        """
        super(GeneratorLoss, self).__init__()
        self.charbonnier = CharbonnierLoss()
        self.l1_charbonnier = L1CharbonnierLoss()

    def __call__(self, d_hr_out, d_sr_out, d_hr_feat_maps, d_sr_feat_maps, sr_images, target_images):
        # Adversarial loss (takes discriminator outputs)
        adversarial_loss = torch.log(d_hr_out) + torch.log(1 - d_sr_out)

        # Content loss (charbonnier between target and super resolution images)
        content_loss = self.charbonnier(sr_images, target_images)

        # Perceptual loss
        perceptual_loss = None
        return adversarial_loss, content_loss, perceptual_loss

