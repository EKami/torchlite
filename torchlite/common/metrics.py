import copy


class Metric:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()


class MetricsList:
    def __init__(self, metrics):
        if metrics:
            self.metrics = [copy.copy(m) for m in metrics]
        else:
            self.metrics = []

        self.train_acc = {}
        self.val_acc = {}
        self.step_count = 0

    def acc_batch(self, step, y_true, y_pred):
        """
        Called on each batch prediction.
        Will accumulate the metrics results.
        Args:
            step (str): Either "training" or "validation"
            y_true (Tensor): The output targets
            y_pred (Tensor): The output logits
        """

        if step == "training":
            for metric in self.metrics:
                result = metric(y_true, y_pred)
                if str(metric) in self.train_acc.keys():
                    self.train_acc[str(metric)] += result
                else:
                    self.train_acc[str(metric)] = result
        elif step == "validation":
            for metric in self.metrics:
                result = metric(y_true, y_pred)
                if str(metric) in self.val_acc.keys():
                    self.val_acc[str(metric)] += result
                else:
                    self.val_acc[str(metric)] = result

        self.step_count += 1

    def avg(self, step):
        """
        Will calculate and return the metrics average results
        Args:
            step (str): Either "training" or "validation"
        Returns:
            dict: A dictionary containing the average of each metric
        """
        logs = {}
        if step == "training":
            for name, total in self.train_acc.items():
                logs[name] = total / self.step_count
        elif step == "validation":
            for name, total in self.val_acc.items():
                logs[name] = total / self.step_count
        return logs

    def reset(self):
        logs = {}
        for metric in self.metrics:
            metric.reset()
        return logs