from collections.__init__ import OrderedDict

from tqdm import tqdm


class TrainCallback:
    def on_epoch_begin(self, logger, epoch, logs=None):
        pass

    def on_epoch_end(self, logger, epoch, logs=None):
        pass

    def on_batch_begin(self, logger, batch, logs=None):
        pass

    def on_batch_end(self, logger, batch, logs=None):
        pass

    def on_train_begin(self, logger, logs=None):
        pass

    def on_train_end(self, logger, logs=None):
        pass


class TrainCallbackList:
    """Container abstracting a list of callbacks.
    Args:
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    def __init__(self, logger, callbacks=None, queue_length=10):
        self.logger = logger
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        assert isinstance(callback, TrainCallback), \
            "Your callback is not an instance of TrainCallback: {}".format(callback)
        self.callbacks.append(callback)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        Args:
            epoch: integer, index of epoch (starts at 1).
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(self.logger, epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        Args:
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(self.logger, epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.
        Args:
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(self.logger, batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.
        Args:
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(self.logger, batch, logs)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        Args:
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(self.logger, logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.
        Args:
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(self.logger, logs)

    def __iter__(self):
        return iter(self.callbacks)


class TQDM(TrainCallback):
    def __init__(self):
        super().__init__()
        self.train_pbar = None
        self.val_pbar = None
        self.total_epochs = 0
        self.train_dataset_len = 0
        self.val_dataset_len = 0

    def on_epoch_begin(self, logger, epoch, logs=None):
        step = logs["step"]
        if step == 'training':
            self.train_pbar = tqdm(total=self.train_dataset_len,
                                   desc="Epochs {}/{}".format(epoch, self.total_epochs),
                                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]'
                                   )
        elif step == 'validation':
            self.val_pbar = tqdm(total=self.val_dataset_len, desc="Validating", leave=False)

    def on_epoch_end(self, logger, epoch, logs=None):
        step = logs["step"]

        if step == 'training':
            self.train_pbar.close()
            train_logs = logs['epoch_logs']
            train_metrics = logs['metrics_logs']
            if len(train_logs) > 0:
                logger.info(*["{}={:03f}".format(k, v) for k, v in train_logs.items()], end=' ')

            if len(train_metrics) > 0:
                logger.info("{:>14}".format("Train metrics:"), end=' ')
                logger.info(*["{}={:03f}".format(k, v) for k, v in train_metrics.items()])
            else:
                print()
        elif step == 'validation':
            self.val_pbar.close()
            val_logs = logs.get('epoch_logs')
            if val_logs and len(val_logs) > 0:
                logger.info(*["{}={:03f}".format(k, v) for k, v in val_logs.items()], end=' ')

            val_metrics = logs.get('metrics_logs')
            if val_metrics and len(val_metrics) > 0:
                logger.info("{:>14}".format("Val metrics:"), end=' ')
                logger.info(*["{}={:03f}".format(k, v) for k, v in val_metrics.items()])

    def on_batch_begin(self, logger, batch, logs=None):
        pass

    def on_batch_end(self, logger, batch, logs=None):
        step = logs["step"]
        batch_logs = logs.get("batch_logs")

        if step == "validation":
            self.train_pbar.set_description(step)  # training or validating
        postfix = OrderedDict()
        if batch_logs:
            for name, value in batch_logs.items():
                postfix[name] = '{0:1.5f}'.format(value)

        self.train_pbar.set_postfix(postfix)
        self.train_pbar.update(1)

    def on_train_begin(self, logger, logs=None):
        self.total_epochs = logs["total_epochs"]
        self.train_dataset_len = logs["train_steps"]
        self.val_dataset_len = logs["val_steps"] if logs["val_steps"] else None
