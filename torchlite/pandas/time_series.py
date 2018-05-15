# TODO include https://github.com/blue-yonder/tsfresh
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from scipy.optimize import minimize
from tqdm import tqdm
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import numpy as np
import pandas as pd


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class SARIMAX:
    def __init__(self, series, d, D, s):
        """
        SARIMA model
        Args:
            series (pd.Series): The pandas series to work on
            d (int): integration order in ARIMA model
            D (int): seasonal integration order
            s (int): length of season
        """
        self.s = s
        self.D = D
        self.d = d
        self.series = series

    def optimize(self, parameters_list, freq='H'):
        """
            Return DataFrame with parameters and corresponding AIC
        Args:
            parameters_list (list): list with (p, q, P, Q) tuples
            freq (str): Frequency

        Returns:
            model: The model with the best parameters
        """

        results = []
        best_aic = float("inf")

        for param in tqdm(parameters_list):
            # we need try-except because on some combinations model fails to converge
            try:
                model = sm.tsa.statespace.SARIMAX(self.series, order=(param[0], self.d, param[1]),
                                                  seasonal_order=(param[3], self.D, param[3], self.s),
                                                  freq=freq).fit(disp=-1)
            except:
                continue
            aic = model.aic
            # saving best model, AIC and parameters
            if aic < best_aic:
                best_aic = aic
            results.append([param, model.aic])

        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        # sorting in ascending order, the lower AIC is - the better
        result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

        p, q, P, Q = result_table.parameters[0]

        # Fit the best model
        best_model = sm.tsa.statespace.SARIMAX(self.series, order=(p, self.d, q),
                                               seasonal_order=(P, self.D, Q, self.s)).fit(disp=-1)

        return best_model

    def plot(self, model, n_steps):
        """
        Plots model vs predicted values

            series - dataset with timeseries
            model - fitted SARIMA model
            n_steps - number of steps to predict in the future
        Args:
            model (statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper): The model to plot
            n_steps (int): t steps

        Returns:
            None
        """
        # adding model values
        data = pd.DataFrame(self.series)
        data.columns = ['actual']
        data['arima_model'] = model.fittedvalues
        # making a shift on s+d steps, because these values were unobserved by the model
        # due to the differentiating
        data.loc[:self.s + self.d, 'arima_model'] = np.NaN

        # forecasting on n_steps forward
        forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
        forecast = data.arima_model.append(forecast)
        # calculate error, again having shifted on s+d steps from the beginning
        error = mean_absolute_percentage_error(data['actual'][self.s + self.d:], data['arima_model'][self.s + self.d:])

        plt.figure(figsize=(15, 7))
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
        plt.plot(forecast, color='r', label="model")
        plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
        plt.plot(data.actual, label="actual")
        plt.legend()
        plt.grid(True)
        plt.show()


class MovingAverage:
    def __init__(self, series):
        """
        Class with multiple function for plotting/predicting time series
        Args:
            series (pd.Series): A pandas Series with date as index and predictor as values
        """
        self.series = series

    def get_simple_moving_average(self, window, show_plot=False, *args, **kwargs):
        """
            Returns a simple moving average (SMA)
            Moving average is based on the assumption: "Tomorrow will be the same as today".
            So to predict for next month m for example, take the moving average at m - 1.
            Most common moving averages are 15, 20, 30, 50, 100 and 200 days.
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
            self._plot_simple_moving_average(self.series, rolling_mean, window, *args, **kwargs)
        return rolling_mean

    def _plot_simple_moving_average(self, series, rolling_mean, window, plot_intervals=False,
                                    conf_interval=1.96, plot_anomalies=False,
                                    title_prefix=""):
        """
        Plot moving average (SMA) over a given pandas Series
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
            print("MAE: {}, Deviation: {}".format(mae, deviation))
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

    def get_exponential_moving_average(self, window):
        """
        Exponential moving average (EMA) with a smoothing factor alpha.
        Args:
            window (int): Rolling window size
        Returns:
            pd.Series: The smoothed values
        """
        alpha = 2 / (1 + window)
        result = [self.series[0]]  # first value is same as series
        for n in range(1, len(self.series)):
            result.append(alpha * self.series[n] + (1 - alpha) * result[n - 1])
        return result

    def plot_exponential_moving_average(self, windows):
        """
        Plots exponential moving average (EMA) with different window size
        Args:
            windows (list): List of rolling window size

        Returns:
            None
        """
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(15, 7))
            for w in windows:
                plt.plot(self.get_exponential_moving_average(w), label="Window size {}".format(w))
            plt.plot(self.series.values, "c", label="Actual")
            plt.legend(loc="best")
            plt.axis('tight')
            plt.title("Exponential Smoothing")
            plt.grid(True)
            plt.show()

    def get_double_exponential_moving_average(self, alpha, beta):
        """
        Double exponential moving average (DEMA) with a smoothing level alpha and a trend beta.
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

    def plot_double_exponential_moving_average(self, alphas, betas):
        """
        Plots double exponential moving average with different alphas and betas
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
                    plt.plot(self.get_double_exponential_moving_average(alpha, beta),
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

    def _plot_holt_winters(self, plot_intervals=False, plot_anomalies=False):
        """
        Plot holt winters
        Args:
            plot_intervals (bool): Show confidence intervals
            plot_anomalies (bool): show anomalies

        Returns:
            None
        """

        plt.figure(figsize=(20, 10))
        plt.plot(self.result, label="Model")
        plt.plot(self.series.values, label="Actual")
        error = mean_absolute_percentage_error(self.series.values, self.result[:len(self.series)])
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(self.series))
            anomalies[self.series.values < self.lower_bond[:len(self.series)]] = \
                self.series.values[self.series.values < self.lower_bond[:len(self.series)]]
            anomalies[self.series.values > self.upper_bond[:len(self.series)]] = \
                self.series.values[self.series.values > self.upper_bond[:len(self.series)]]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

        if plot_intervals:
            plt.plot(self.upper_bond, "r--", alpha=0.5, label="Up/Low confidence")
            plt.plot(self.lower_bond, "r--", alpha=0.5)
            plt.fill_between(x=range(0, len(self.result)), y1=self.upper_bond,
                             y2=self.lower_bond, alpha=0.2, color="grey")

        plt.vlines(len(self.series), ymin=min(self.lower_bond), ymax=max(self.upper_bond), linestyles='dashed')
        plt.axvspan(len(self.series) - 20, len(self.result), alpha=0.3, color='lightgrey')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc="best", fontsize=13)
        plt.show()

    def triple_exponential_smoothing(self, plot_results=False, *args, **kwargs):
        """
        Returns the triple exponential smoothing results
        Args:
            plot_results (bool): If True the results will be plotted
            *args (list): Arguments to pass to _plot_holt_winters()
            **kwargs (dict): Arguments to pass to _plot_holt_winters()

        Returns:
            list: Triple exponential smoothing results
        """
        smooth_l = []
        season_l = []
        trend_l = []
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
                smooth_l.append(smooth)
                trend_l.append(trend)
                season_l.append(seasonals[i % self.slen])

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

            smooth_l.append(smooth)
            trend_l.append(trend)
            season_l.append(seasonals[i % self.slen])

        if plot_results:
            self._plot_holt_winters(*args, **kwargs)
        return self.result

    def get_best_parameters(self, inplace=True, loss_function=mean_squared_log_error, n_folds=3):
        """
        Optimize for getting the best alpha, beta and gamma parameters
        on cross validation time series split.
        Args:
            inplace (bool): If True the internal Alpha, Beta, Gamma of this class will be replaced
                by the optimal ones
            n_folds (int): Number of folds for cross validation
            loss_function (function): Sklearn metric loss function
        Returns:
            tuple: Alpha, Beta, Gamma
        """
        # initializing model parameters alpha, beta and gamma
        x = np.array([0, 0, 0])

        # Minimizing the loss function
        opt = minimize(time_series_cv_score, x0=x, args=(self.series, loss_function, self.slen, n_folds),
                       method="TNC", bounds=((0, 1), (0, 1), (0, 1)))

        # Take optimal values...
        alpha, beta, gamma = opt.x
        print("Alpha: {}, Beta: {}, Gamma: {}".format(alpha, beta, gamma))
        if inplace:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

        return alpha, beta, gamma


def time_series_cv_score(params, series, loss_function, slen, n_folds=3):
    """
    Returns error on Cross validation for time series
    Args:
        params (list): Vector of parameters for optimization
        series (pd.Series): dataset with timeseries
        loss_function (function): Sklearn metric loss function
        slen (int): length of a season
        n_folds (int): Number of folds for cross validation
    Returns:
        float: Error
    """
    # errors array
    errors = []

    values = series.values
    alpha, beta, gamma = params

    # set the number of folds for cross-validation
    tscv = TimeSeriesSplit(n_splits=n_folds)

    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tscv.split(values):
        model = HoltWinters(series=values[train], slen=slen,
                            alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()

        predictions = model.result[-len(test):]
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)

    return np.mean(np.array(errors))


def test_stationary(y, show_plots=True, lags=None, figsize=(12, 7), style='bmh'):
    """
    Plot time series, its ACF (Autocorrelation function) and PACF (Partial autocorrelation function),
    calculate Augmented Dickey–Fuller test.
    Used to check if a time series is stationary or not.

        - p-value > 0.05: Accept the null hypothesis (H0), the data has a unit root and is non-stationary.
        - p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

    Args:
        y (pd.Series, list): Time series pandas series
        show_plots (bool): True to show the TS/ACF/PACF plots
        lags (int): How many lags to include in ACF, PACF plot calculation
        figsize (tuple): Size of plot
        style (str): Style of plot

    Returns:
        tuple: (p-value, is_stationary)
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    p_value = sm.tsa.stattools.adfuller(y)[1]
    is_stationary = False

    if show_plots:
        with plt.style.context(style):
            plt.figure(figsize=figsize)
            layout = (2, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))

            y.plot(ax=ts_ax)
            ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
            smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
            smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
            plt.tight_layout()
            plt.show()

    if p_value <= 0.05:
        is_stationary = True

    return p_value, is_stationary
