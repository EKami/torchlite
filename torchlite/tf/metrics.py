import numpy as np
from sklearn.metrics import fbeta_score
from typing import Union

from torchlite.common.metrics import Metric
from torchlite.common.tools import stats


class FBetaScore(Metric):
    def __init__(self, beta, average="binary", threshold=None):
        """
        Returns the FBeta score
        Args:
            beta (int): Beta for F score
            average (str): [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
            threshold (None, float): Threshold for y_pred. None to not use any.
        """
        super().__init__()
        self.threshold = threshold
        self.average = average
        self.beta = beta

    @staticmethod
    def calculate_f2(y_true: np.ndarray, y_pred: np.ndarray, beta: int, average: str, threshold: Union[None, float]):
        if threshold is not None:
            y_pred = stats.get_thresholded_predictions(y_pred, threshold=threshold)
        return fbeta_score(y_true, y_pred, beta=beta, average=average)

    def __call__(self, logger, y_true, y_pred, *args, **kwargs):
        return FBetaScore.calculate_f2(y_true.numpy(), y_pred.numpy(), self.beta, self.average, self.threshold)

    def __repr__(self):
        return f"f{self.beta}_score"
