"""
This class contains different cores to pass to the learner class.
Most of the time you'll make use of ClassifierCore.
"""
import torch
import torch.nn as nn
from torchlite.torch.tools import tensor_tools
from torchlite.torch.models.srpgan import Generator, Discriminator


class BaseCore:
    def on_train_mode(self):
        raise NotImplementedError()

    def on_eval_mode(self):
        raise NotImplementedError()

    def on_new_epoch(self):
        """
        A callback called when a new epoch starts.
        You typically want to reset your logs here.
        """
        raise NotImplementedError()

    def to_device(self, device):
        """
        Move the model onto the GPU

        Args:
            device (torch.device): Pytorch device object
        """
        raise NotImplementedError()

    @property
    def get_models(self):
        """
        Returns the core model(s) as dictionary
        Returns:
            dict: A dictionary of models in the form {"model_name": torch.Module}
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
            Tensor: The logits (used only for the metrics)
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
        self.avg_meter = tensor_tools.AverageMeter()

    @property
    def get_models(self):
        return {self.model.__class__.__name__: self.model}

    @property
    def get_logs(self):
        return self.logs

    def on_new_epoch(self):
        self.logs = {}
        self.avg_meter = tensor_tools.AverageMeter()

    def on_train_mode(self):
        self.model.train()

    def on_eval_mode(self):
        self.model.eval()

    def to_device(self, device):
        self.model.to(device)

    def on_forward_batch(self, step, inputs, targets=None):
        # forward
        logits = self.model.forward(*inputs)

        if step != "prediction":
            loss = self.crit(logits, targets)

            # Update logs
            self.avg_meter.update(loss.item())
            self.logs.update({"batch_logs": {"loss": loss.item()}})

            # backward + optimize
            if step == "training":
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.logs.update({"epoch_logs": {"train loss": self.avg_meter.avg}})
            else:
                self.logs.update({"epoch_logs": {"valid loss": self.avg_meter.avg}})
        return logits


class SRPGanCore(BaseCore):
    def __init__(self, generator: Generator, discriminator: Discriminator,
                 g_optimizer, d_optimizer, g_criterion, d_criterion):
        """
        A GAN core classifier which takes as input a generator and a discriminator
        Args:
            generator (Generator): Model definition of the generator
            discriminator (Discriminator): Model definition of the discriminator
            g_optimizer (Optimizer): Generator optimizer
            d_optimizer (Optimizer): Discriminator optimizer
            g_criterion (callable): The Generator criterion
            d_criterion (callable): The discriminator criterion
        """
        self.d_criterion = d_criterion
        self.g_criterion = g_criterion
        self.d_optim = d_optimizer
        self.g_optim = g_optimizer
        self.netD = discriminator
        self.netG = generator
        self.on_new_epoch()

    def on_train_mode(self):
        self.netG.train()
        self.netD.train()

    def on_eval_mode(self):
        self.netG.eval()
        self.netD.eval()

    def to_device(self, device):
        self.netG.to(device)
        self.netD.to(device)

    @property
    def get_models(self):
        return {self.netD.__class__.__name__: self.netD,
                self.netG.__class__.__name__: self.netG}

    @property
    def get_logs(self):
        return self.logs

    def on_new_epoch(self):
        self.logs = {}
        self.g_avg_meter = tensor_tools.AverageMeter()
        self.d_avg_meter = tensor_tools.AverageMeter()
        self.adversarial_loss_meter = tensor_tools.AverageMeter()
        self.content_loss_meter = tensor_tools.AverageMeter()
        self.perceptual_loss_meter = tensor_tools.AverageMeter()

    def _update_loss_logs(self, g_loss, d_loss, adversarial_loss, content_loss, perceptual_loss):
        # Update logs
        self.g_avg_meter.update(g_loss)
        self.d_avg_meter.update(d_loss)
        self.adversarial_loss_meter.update(adversarial_loss)
        self.content_loss_meter.update(content_loss)
        self.perceptual_loss_meter.update(perceptual_loss)

        self.logs.update({"batch_logs": {"g_loss": g_loss, "d_loss": d_loss}})
        self.logs.update({"epoch_logs": {"generator": self.g_avg_meter.avg,
                                         "discriminator": self.d_avg_meter.avg,
                                         "adversarial": self.adversarial_loss_meter.avg,
                                         "content": self.content_loss_meter.avg,
                                         "perceptual": self.perceptual_loss_meter.avg}})

    def _on_eval(self, images):
        sr_images = self.netG(images)  # Super resolution images
        return sr_images

    def _on_validation(self, lr_images, hr_images):
        sr_images = self.netG(lr_images)

        return sr_images

    def _optimize(self, model, optim, loss, retain_graph=False):
        model.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optim.step()

    def _on_training(self, lr_images, hr_images):
        sr_images = self.netG(lr_images)
        d_hr_out, d_hr_feat_maps = self.netD(hr_images)  # Sigmoid output
        d_sr_out, d_sr_feat_maps = self.netD(sr_images)  # Sigmoid output

        # torchlite.torch.losses.srpgan.GeneratorLoss
        g_loss, adversarial_loss, content_loss, perceptual_loss = self.g_criterion(d_hr_out, d_sr_out,
                                                                                   d_hr_feat_maps, d_sr_feat_maps,
                                                                                   sr_images, hr_images)

        d_loss = self.d_criterion(d_hr_out, d_sr_out)

        self._optimize(self.netD, self.d_optim, d_loss, retain_graph=True)
        self._optimize(self.netG, self.g_optim, g_loss)

        self._update_loss_logs(g_loss, d_loss, adversarial_loss, content_loss, perceptual_loss)

        return sr_images

    def on_forward_batch(self, step, inputs, targets=None):
        self.logs = {}
        if step == "training":
            return self._on_training(*inputs, targets)
        elif step == "validation":
            return self._on_validation(*inputs, targets)
        elif step == "eval":
            return self._on_eval(*inputs)
