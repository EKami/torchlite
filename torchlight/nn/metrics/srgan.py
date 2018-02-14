from torchlight.nn.metrics.metrics import Metric
import torchlight.nn.tools.ssim as ssim


class SSIM(Metric):
    def __init__(self):
        self.count = 0
        self.sum = 0

    @property
    def get_name(self):
        return "ssim"

    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.count = 0
        self.sum = 0

    def __call__(self, step, logits, target):
        if step == "validation":
            lr_images, lr_upscaled_images, hr_original_images, sr_images = logits

            res = ssim.ssim(sr_images, hr_original_images).data[0]
            self.count += lr_images.size()[0]  # Batch size
            self.sum += res
            return res


class PSNR(Metric):
    def __init__(self):
        self.count = 0
        self.sum = 0

    def get_name(self):
        return "psnr"

    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.count = 0
        self.sum = 0

    def __call__(self, step, logits, target):
        if step == "validation":
            lr_images, lr_upscaled_images, hr_original_images, sr_images = logits
            # TODO finish
        pass
