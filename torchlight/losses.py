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
    return np.sqrt((weights * np.exp((np.log1p(y_pred) - np.log1p(y_true)), 2)) / weights)
