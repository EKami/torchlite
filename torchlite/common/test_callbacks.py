from tqdm import tqdm


class TestCallback:
    def __init__(self):
        self.validation_data = None
        self.params = None
        self.model = None

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class TestCallbackList(object):
    """Container abstracting a list of callbacks.
    Args:
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        assert isinstance(callback, TestCallback), \
            "Your callback is not an instance of TestCallback: {}".format(callback)
        self.callbacks.append(callback)

    def on_test_begin(self, logs=None):
        """Called at the beginning of testing.
        Args:
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        """Called at the end of testing.
        Args:
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.
        Args:
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.
        Args:
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def __iter__(self):
        return iter(self.callbacks)


class TQDM(TestCallback):
    def __init__(self):
        super().__init__()
        self.pbar = None

    def on_test_begin(self, logs=None):
        test_loader_len = len(logs["loader"])
        self.pbar = tqdm(total=test_loader_len, desc="Classifying")

    def on_batch_end(self, batch, logs=None):
        self.pbar.update(1)

    def on_test_end(self, logs=None):
        print()