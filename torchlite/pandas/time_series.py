# TODO include https://github.com/blue-yonder/tsfresh
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class MovingAverage:
    def __init__(self, series):
        """
        Class with multiple function for plotting/predicting time series
        Args:
            series (pd.Series): A pandas Series with date as index and predictor as values
        """
        self.series = series

    def get_moving_average(self, window, show_plot=False, *args, **kwargs):
        """
            Returns a simple moving average
            Moving average is based on the assumption: "Tomorrow will be the same as today".
            So to predict for next month m for example, take the moving average at m - 1
        Args:
            window (int): The number of entries by which to apply the smoothing factor
            show_plot (bool): If set to True will show moving average plot.
                The arguments passed to (*args, **kwargs) will be transferred to
                _plotMovingAverage().
            *args (list): Arguments to pass to _plotMovingAverage()
            **kwargs (dict): Arguments to pass to _plotMovingAverage()

        Returns:
            pd.Series: A pandas Series with moving average
        """
        rolling_mean = self.series.rolling(window=window).mean()
        if show_plot:
            self._plotMovingAverage(self.series, rolling_mean, window, *args, **kwargs)
        return rolling_mean

    def _plotMovingAverage(self, series, rolling_mean, window, plot_intervals=False,
                           conf_interval=1.96, plot_anomalies=False,
                           title_prefix=""):
        """
        Plot moving average over a given pandas Series
        Args:
            series (pd.Series): A pandas Series with date index
            rolling_mean (pd.Series): A pandas series
            window (int): Rolling window size
            plot_intervals (bool): Show confidence intervals
            conf_interval (float): The confidence interval:
                0.95 = 95% interval with z = 1.96
                0.99 = 99% interval with z = 2.576
                0.995 = 99.5% interval with z = 2.807
                0.999 = 99.9% interval with z = 3.291
            plot_anomalies (bool): show anomalies
            title_prefix (str): A prefix for the plot title
        Returns:

        """
        z = {0.95: 1.96, 0.99: 2.576, 0.995: 2.807, 0.999: 3.291}

        plt.figure(figsize=(15, 5))
        plt.title(title_prefix + ": Moving average\n window size = {}".format(window))
        plt.plot(rolling_mean, "g", label="Rolling mean trend")

        # Plot confidence intervals for smoothed values
        if plot_intervals:
            mae = mean_absolute_error(series[window:], rolling_mean[window:])
            deviation = np.std(series[window:] - rolling_mean[window:])
            print(f"MAE: {mae}, Deviation: {deviation}")
            # z = 1.96 is the 95% confidence interval
            lower_bond = rolling_mean - (mae + z[conf_interval] * deviation)
            upper_bond = rolling_mean + (mae + z[conf_interval] * deviation)
            plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
            plt.plot(lower_bond, "r--")

            # Having the intervals, find abnormal values
            if plot_anomalies:
                anomalies = pd.DataFrame(index=series.index, columns=series.columns)
                anomalies[series < lower_bond] = series[series < lower_bond]
                anomalies[series > upper_bond] = series[series > upper_bond]
                plt.plot(anomalies, "ro", markersize=10)

        plt.plot(series[window:], label="Actual values")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.show()

    def get_exponential_smoothing(self, alpha):
        """
        Exponential smoothing with a smoothing factor alpha
        Args:
            alpha (float): Smoothing parameter, float which value is between [0.0, 1.0]
        Returns:
            pd.Series: The smoothed values
        """
        result = [self.series[0]]  # first value is same as series
        for n in range(1, len(self.series)):
            result.append(alpha * self.series[n] + (1 - alpha) * result[n - 1])
        return result

    def plot_exponential_smoothing(self, alphas):
        """
        Plots exponential smoothing with different alphas
        Args:
            alphas (list): Smoothing parameter for level, list of floats which value are between [0.0, 1.0]

        Returns:
            None
        """
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(15, 7))
            for alpha in alphas:
                plt.plot(self.get_exponential_smoothing(alpha), label="Alpha {}".format(alpha))
            plt.plot(self.series.values, "c", label="Actual")
            plt.legend(loc="best")
            plt.axis('tight')
            plt.title("Exponential Smoothing")
            plt.grid(True)
            plt.show()

    def get_double_exponential_smoothing(self, alpha, beta):
        """
        Double exponential smoothing with a smoothing level alpha and a trend beta.
        Alpha is responsible for the series smoothing around the trend, Beta for the smoothing of the trend itself
        The larger the values, the more weight the most recent observations will have and the
        less smoothed the model series will be.
        Args:
            alpha (float): Smoothing parameter for level, float which value is between [0.0, 1.0]
            beta (float): Smoothing parameter for trend, float which value is between [0.0, 1.0]
        Returns:
            pd.Series: Smoothed values
        """
        # first value is same as series
        result = [self.series[0]]
        for n in range(1, len(self.series) + 1):
            if n == 1:
                level, trend = self.series[0], self.series[1] - self.series[0]
            if n >= len(self.series):  # forecasting
                value = result[-1]
            else:
                value = self.series[n]
            last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            result.append(level + trend)
        return result

    def plot_double_exponential_smoothing(self, alphas, betas):
        """
        Plots double exponential smoothing with different alphas and betas
        Args:
            alphas (list): Smoothing parameter for level, list of floats which value are between [0.0, 1.0]
            betas (list): Smoothing parameter for trend, list of floats which value are between [0.0, 1.0]

        Returns:
            None
        """

        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(20, 8))
            for alpha in alphas:
                for beta in betas:
                    plt.plot(self.get_double_exponential_smoothing(alpha, beta),
                             label="Alpha {}, beta {}".format(alpha, beta))
            plt.plot(self.series.values, label="Actual")
            plt.legend(loc="best")
            plt.axis('tight')
            plt.title("Double Exponential Smoothing")
            plt.grid(True)
            plt.show()


class HoltWinters:
    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        """
        Holt-Winters model with the anomalies detection using Brutlag method.
        This model is best used when the data has seasonality.
        Args:
            series (pd.Series): initial time series
            slen (int): length of a season
            alpha (float): Holt-Winters model coefficient
            beta (float): Holt-Winters model coefficient
            gamma (float): Holt-Winters model coefficient
            n_preds (int): predictions horizon
            scaling_factor (float):  z value: sets the width of the confidence interval by Brutlag
                95% interval with scaling_factor = 1.96
                99% interval with scaling_factor = 2.576
                99.5% interval with scaling_factor = 2.807
                99.9% interval with scaling_factor = 3.291
        """
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        self.result = []
        self.upper_bond = []
        self.lower_bond = []

    def _initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i + self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def _initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen * j:self.slen * j + self.slen]) / float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen * j + i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def triple_exponential_smoothing(self):
        smooth = []
        season = []
        trend = []
        predicted_deviation = []
        self.result = []
        self.upper_bond = []
        self.lower_bond = []

        seasonals = self._initial_seasonal_components()

        for i in range(len(self.series) + self.n_preds):
            if i == 0:  # components initialization
                smooth = self.series[0]
                trend = self._initial_trend()
                self.result.append(self.series[0])
                smooth.append(smooth)
                trend.append(trend)
                season.append(seasonals[i % self.slen])

                predicted_deviation.append(0)

                self.upper_bond.append(self.result[0] + self.scaling_factor * predicted_deviation[0])

                self.lower_bond.append(self.result[0] - self.scaling_factor * predicted_deviation[0])
                continue

            if i >= len(self.series):  # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i % self.slen])

                # when predicting we increase uncertainty on each step
                predicted_deviation.append(predicted_deviation[-1] * 1.01)

            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha * (val - seasonals[i % self.slen]) + (1 - self.alpha) * (
                        smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i % self.slen]
                self.result.append(smooth + trend + seasonals[i % self.slen])

                # Deviation is calculated according to Brutlag algorithm.
                predicted_deviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) +
                                           (1 - self.gamma) * predicted_deviation[-1])

            self.upper_bond.append(self.result[-1] + self.scaling_factor * predicted_deviation[-1])
            self.lower_bond.append(self.result[-1] - self.scaling_factor * predicted_deviation[-1])

            smooth.append(smooth)
            trend.append(trend)
            season.append(seasonals[i % self.slen])
        return self.result

    def plot_holt_winters(self, series, plot_intervals=False, plot_anomalies=False):
        """
        Plot holt winters
        Args:
            series (pd.Series): initial time series
            plot_intervals (bool): Show confidence intervals
            plot_anomalies (bool): show anomalies

        Returns:
            None
        """

        plt.figure(figsize=(20, 10))
        plt.plot(self.result, label="Model")
        plt.plot(series.values, label="Actual")
        error = mean_absolute_percentage_error(series.values, self.result[:len(series)])
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(series))
            anomalies[series.values < self.lower_bond[:len(series)]] = \
                series.values[series.values < self.lower_bond[:len(series)]]
            anomalies[series.values > self.upper_bond[:len(series)]] = \
                series.values[series.values > self.upper_bond[:len(series)]]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

        if plot_intervals:
            plt.plot(self.upper_bond, "r--", alpha=0.5, label="Up/Low confidence")
            plt.plot(self.lower_bond, "r--", alpha=0.5)
            plt.fill_between(x=range(0, len(self.result)), y1=self.upper_bond,
                             y2=self.lower_bond, alpha=0.2, color="grey")

        plt.vlines(len(series), ymin=min(self.lower_bond), ymax=max(self.upper_bond), linestyles='dashed')
        plt.axvspan(len(series) - 20, len(self.result), alpha=0.3, color='lightgrey')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc="best", fontsize=13)
        plt.show()
