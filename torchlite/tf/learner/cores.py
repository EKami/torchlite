import tensorflow as tf
from tensorflow import keras

from torchlite.learner import BaseCore
from torchlite.common.tools import tensor_tools


class ClassifierCore(BaseCore):
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