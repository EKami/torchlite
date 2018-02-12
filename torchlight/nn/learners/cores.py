import torch.nn as nn
from torchlight.nn import tools


class BaseCore:
    def on_train_mode(self):
        raise NotImplementedError()

    def on_eval_mode(self):
        raise NotImplementedError()

    def to_gpu(self):
        """
        Move the model onto the GPU
        """
        raise NotImplementedError()

    @property
    def get_logs(self):
        """
        Returns the logs for display

        Returns:
            dict: The logs from the forward batch
        """
        raise NotImplementedError()

    def on_forward_batch(self, step, inputs, targets=None):
        """
        Callback called during training, validation and prediction batch processing steps
        Args:
            step (str): Either:
                - training
                - validation
                - prediction
            inputs (Tensor): The batch inputs to feed to the model
            targets (Tensor): The expected outputs

        Returns:
            Tensor: The logits
        """
        raise NotImplementedError()


class ClassifierCore(BaseCore):
    def __init__(self, model, optimizer, criterion):
        """
        The learner core for classification models
        Args:
            model (nn.Module): The pytorch model
            optimizer (Optimizer): The optimizer function
            criterion (callable): The objective criterion.
        """
        self.crit = criterion
        self.optim = optimizer
        self.model = model
        self.logs = {}
        self.avg_meter = tools.AverageMeter()

    @property
    def get_logs(self):
        return self.logs

    def on_train_mode(self):
        self.model.train()

    def on_eval_mode(self):
        self.model.eval()

    def to_gpu(self):
        tools.to_gpu(self.model)

    def on_forward_batch(self, step, inputs, targets=None):
        # forward
        logits = self.model.forward(*inputs)

        if step != "prediction":
            loss = self.crit(logits, targets)

            # Update logs
            self.avg_meter.update(loss.data[0])
            self.logs.update({"batch_logs": {"loss": loss.data[0]}})

            # backward + optimize
            if step == "training":
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.logs.update({"epoch_logs": {"train loss": self.avg_meter.debias_loss}})
            else:
                self.logs.update({"epoch_logs": {"valid loss": self.avg_meter.debias_loss}})
        return logits


class SRGanCore(BaseCore):
    def __init__(self, generator: nn.Module, discriminator: nn.Module,
                 g_optimizer, d_optimizer, g_criterion):
        """
        A GAN core classifier which takes as input a generator and a discriminator
        Args:
            generator (nn.Module): Model definition of the generator
            discriminator (nn.Module): Model definition of the discriminator
            g_optimizer (Optimizer): Generator optimizer
            d_optimizer (Optimizer): Discriminator optimizer
            g_criterion (callable): The Generator criterion
        """
        self.g_criterion = g_criterion
        self.d_optim = d_optimizer
        self.g_optim = g_optimizer
        self.netD = discriminator
        self.netG = generator
        self.logs = {}
        self.g_avg_meter = tools.AverageMeter()
        self.d_avg_meter = tools.AverageMeter()

        self.val_mse = 0

    def on_train_mode(self):
        self.netG.train()
        self.netD.train()

    def on_eval_mode(self):
        self.netG.eval()
        self.netD.eval()

    def to_gpu(self):
        tools.to_gpu(self.netG)
        tools.to_gpu(self.netD)

    @property
    def get_logs(self):
        return self.logs

    def _optimize(self, model, optim, loss, retain_graph=False):
        model.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optim.step()

    def _on_eval(self, lr_images, hr_images, hr_original_images):
        pass

    def _on_validation(self, lr_images, hr_images, hr_original_images):
        # https://github.com/leftthomas/SRGAN/blob/master/train.py#L111
        batch_size = lr_images.size(0)
        sr_images = self.netG(lr_images)

        batch_mse = ((sr_images - hr_images) ** 2).data.mean()
        self.val_mse += batch_mse * batch_size
        # batch_ssim = pytorch_ssim.ssim(sr, hr).data[0]
        # valing_results['ssims'] += batch_ssim * batch_size
        # valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
        # valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
        # val_bar.set_description(
        #     desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
        #         valing_results['psnr'], valing_results['ssim']))
        #
        # val_images.extend(
        #     [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
        #      display_transform()(sr.data.cpu().squeeze(0))])

    def _on_training(self, lr_images, hr_images):
        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        sr_img = self.netG(lr_images)
        d_real_out = self.netD(hr_images).mean()
        d_fake_out = self.netD(sr_img).mean()
        d_loss = 1 - d_real_out + d_fake_out
        self._optimize(self.netD, self.d_optim, d_loss, retain_graph=True)

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        g_loss = self.g_criterion(d_fake_out, sr_img, hr_images)
        self._optimize(self.netG, self.g_optim, g_loss)

        # Update logs
        self.g_avg_meter.update(g_loss.data[0])
        self.d_avg_meter.update(d_loss.data[0])
        self.logs.update({"batch_logs": {"g_loss": g_loss.data[0], "d_loss": d_loss.data[0]}})
        self.logs.update({"epoch_logs": {"generator loss": self.g_avg_meter.debias_loss,
                                         "discriminator loss": self.d_avg_meter.debias_loss}})

        return sr_img, d_real_out, d_fake_out

    def on_forward_batch(self, step, inputs, targets=None):
        if step == "training":
            return self._on_training(*inputs, targets)
        elif step == "validation":
            return self._on_validation(*inputs, targets)
        elif step == "eval":
            return self._on_eval(*inputs, targets)

