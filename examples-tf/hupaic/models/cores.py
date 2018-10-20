from torchlite.learner.cores import BaseCore
from tensorflow import keras


class HupaicCore(BaseCore):
    def __init__(self, model, loss_function, optimizer, data_ind):
        """
        The core engine for the Hupaic project
        Args:
            model (object): A model object
            loss_function (callable): The loss function
            optimizer (tf.train.optimizer.Optimizer): A TF optimizer
            data_ind (tuple): A tuple containing the input data and the labels position in the batches
        """
        self.model = model
        self.data_ind = data_ind
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.logs = {}
        self.PHASE_TRAIN = 1
        self.PHASE_EVAL = 0

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
        d = 0
