from torchlight.nn.metrics.metrics import Metric
import torchlight.nn.tools.ssim as ssim


class SSIM(Metric):
    def get_name(self):
        pass

    def avg(self):
        pass

    def reset(self):
        pass

    def __call__(self, step, logits, target):
        if step == "validation:":
            lr_images, lr_upscaled_images, hr_original_images, sr_images = logits
            batch_mse = ((sr_images - lr_upscaled_images) ** 2).data.mean()

            # self.mse_meter.update(batch_mse)
            # self.ssim_meter.update(ssim.ssim(sr_images, hr_original_images).data[0])
            # self.psnr_meter.update(10 * np.log10(1 / self.mse_meter.avg))
            #
            # self.logs.update({"epoch_logs": {"ssim": self.ssim_meter.avg,
            #                                  "psnr": self.psnr_meter.avg,
            #                                  "mse": self.mse_meter.avg}})


class PSNR(Metric):
    def get_name(self):
        pass

    def avg(self):
        pass

    def reset(self):
        pass

    def __call__(self, *logits):
        pass
