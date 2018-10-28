"""
This class contains different cores to pass to the learner class.
Most of the time you'll make use of ClassifierCore.
"""


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


