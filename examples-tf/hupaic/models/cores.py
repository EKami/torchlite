from torchlite.learner.cores import BaseCore


class RsnaCore(BaseCore):
    def on_train_mode(self):
        pass

    def on_eval_mode(self):
        pass

    def on_new_epoch(self):
        pass

    def to_device(self, device):
        pass

    @property
    def get_models(self):
        pass

    @property
    def get_logs(self):
        pass

    def on_forward_batch(self, step, inputs, targets=None):
        pass
