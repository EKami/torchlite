import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plot_model_results(model, X_train, y_train, X_test, y_test,
                       cross_val_obj, plot_intervals=False, plot_anomalies=False):
    """
    Plots modelled vs fact values, prediction intervals and anomalies
        Args:
            model : The linear regression model
            X_train (np.ndarray): The train dataset
            y_train (nd.ndrray): The train labels
            X_test (np.ndarray): The test or validation data
            y_test (np.ndarray): The test or validation labels
            cross_val_obj:
            plot_intervals (bool): True to plot the confidence intervals
            plot_anomalies (bool): True to plot the anomalies

        Returns:

        """

    prediction = model.predict(X_test)

    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test, label="actual", linewidth=2.0)

    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train,
                             cv=cross_val_obj,
                             scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()

        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)

        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(y_test))
            anomalies[y_test < lower] = y_test[y_test < lower]
            anomalies[y_test > upper] = y_test[y_test > upper]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_coefficients(model, X_train_columns):
    """
    Plots sorted coefficient values of a linear model
    Args:
        X_train_columns (list): The list of X_train columns
    Returns:
        None
    """

    coefs = pd.DataFrame(model.coef_, X_train_columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')
    plt.show()
