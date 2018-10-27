"""
This class contains different cores to pass to the learner class.
Most of the time you'll make use of ClassifierCore.
"""
import tensorflow as tf
from tensorflow import keras
from torchlite.tools import tensor_tools


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
            device (torch.device, str): Pytorch device object or a string for TF
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


class TorchClassifierCore(BaseCore):
    def __init__(self, model, optimizer, criterion, input_index=0):
        """
        The learner core for classification models
        Args:
            model (nn.Module): The pytorch model
            optimizer (Optimizer): The optimizer function
            criterion (callable): The objective criterion.
            input_index (int): An index pointing to the location of the input data in the batch.
                N.B: The target is always the last item in the batch
        """
        self.input_index = input_index
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
        logits = self.model.forward(inputs[self.input_index])

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


class TFClassifierCore(BaseCore):
    def __init__(self, model, loss_function, optimizer, input_index=0):
        """
        The core engine for the Hupaic project
        Args:
            model (keras.Model): A model object
            loss_function (callable): The loss function
            optimizer (tf.train.optimizer.Optimizer): A TF optimizer
            input_index (int): An index pointing to the location of the input data in the batch.
                N.B: The target is always the last item in the batch
        """
        self.model = model
        self.input_index = input_index
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.logs = {}
        self.PHASE_TRAIN = 1
        self.PHASE_EVAL = 0
        self.avg_meter = tensor_tools.AverageMeter()

    def on_train_mode(self):
        keras.backend.set_learning_phase(self.PHASE_TRAIN)

    def on_eval_mode(self):
        keras.backend.set_learning_phase(self.PHASE_EVAL)

    def on_new_epoch(self):
        pass

    def to_device(self, device):
        # Ignored for TF
        pass

    @property
    def get_models(self):
        return {self.model.__class__.__name__: self.model}

    @property
    def get_logs(self):
        return self.logs

    def on_forward_batch(self, step, inputs, targets=None):
        # TODO TF 2.0
        # y_pred = self.model(inputs)
        # loss = self.loss_function(batch[data_ind[1]], y_pred)
        # opt_op = self.optimizer.minimize(lambda: loss, var_list=self.model.trainable_variables)
        # opt_op.run()

        with tf.GradientTape() as tape:
            y_pred = self.model(inputs[self.input_index])
            loss = self.loss_function(targets, y_pred)

        loss_np = tf.reduce_mean(loss).numpy()
        self.avg_meter.update(loss_np)
        self.logs.update({"batch_logs": {"loss": loss_np}})

        if step == "training":
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables),
                                           global_step=tf.train.get_or_create_global_step())
            self.logs.update({"epoch_logs": {"train loss": self.avg_meter.avg}})
        elif step == "validation":
            self.logs.update({"epoch_logs": {"valid loss": self.avg_meter.avg}})

        return y_pred
