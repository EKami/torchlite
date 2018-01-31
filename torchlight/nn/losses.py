import numpy as np


def nwrmsle(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray):
    """
    Calculates and returns the Normalized Weighted Root Mean Squared Logarithmic Error
    Args:
        y_true (np.ndarray): The true labels
        y_pred (np.ndarray): The predicted labels
        weights (np.ndarray): The weights for each predictions

    Returns:
        int: The NWRMSLE
    """
    assert y_true.shape == y_pred.shape == weights.shape, "Arguments are not of same shape"
    y_true = y_true.clip(min=0)
    y_pred = y_pred.clip(min=0)
    return np.sqrt(np.sum(weights * np.square(np.log1p(y_pred) - np.log1p(y_true))) / np.sum(weights))
