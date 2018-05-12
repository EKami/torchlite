# TODO include https://github.com/blue-yonder/tsfresh
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import re


def lookup(s, date_format):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    dates = {date: pd.to_datetime(date, format=date_format) for date in s.unique()}
    return s.map(dates)


def get_datepart(df, field_name, transform_list=('Year', 'Month', 'Week', 'Day',
                                                 'Dayofweek', 'Dayofyear',
                                                 'Is_month_end', 'Is_month_start',
                                                 'Is_quarter_end', 'Is_quarter_start',
                                                 'Is_year_end', 'Is_year_start'),
                 drop=True, inplace=False, date_format='%Y-%m-%d'):
    """
    Converts a column of df from a datetime64 to many columns containing
    the information from the date. `transform_list` is the list of transformations.
    A field "Elapsed" is always added to the resulting DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame
        field_name (str): A string that is the name of the date column you wish to expand
        transform_list (list): List of data transformations to add to the original dataset
        drop (bool): If True then the original date column will be removed
        inplace (bool): If the operations are done inplace or not
        date_format (str): The datetime format for parsing
    Returns:
        A pandas DataFrame
    """
    if not inplace:
        df = df.copy()
    field = df[field_name]
    targ_pre = re.sub('[Dd]ate$', '', field_name)

    if isinstance(df, pd.DataFrame):
        if not np.issubdtype(field.dtype, np.datetime64):
            df[field_name] = field = lookup(field, date_format)
    for n in transform_list:
        df[targ_pre + n] = getattr(field.dt, n.lower())
        if df[targ_pre + n].dtype == np.int64:
            df[targ_pre + n] = df[targ_pre + n].astype(np.int16)

    if drop:
        df = df.drop(field_name, axis=1)
    return df


def get_elapsed(df, date_field, from_date=np.datetime64('1970-01-01'), prefix='Elapsed_',
                inplace=False, dtype='timedelta64[s]'):
    """
    This function will add a new column which will count the time elapsed relative to a particular
    date (1970-01-01 by default)
    Args:
        df (pd.DataFrame): A pandas DataFrame
        date_field (str): The field containing the field of type datetime64
        from_date (np.datetime64): The date from which you want to start the counter
        prefix (str): The prefix you want to add to the newly created column. The prefix tail will depends on
            the dtype
        inplace (bool): If the operations are done inplace or not
        dtype (str): "timedelta64[s]" for seconds, "timedelta64[m]" for minutes, "timedelta64[h]" for hours,
            "timedelta64[D]" for days, "timedelta64[M]" for months, "timedelta64[Y]" for years.
    Returns:
        DataFrame: The passed DataFrame with the elapsed time column
    """
    if not inplace:
        df = df.copy()

    res = [None] * df.shape[0]
    ts_type = "unknown"
    for i, v in enumerate(df[date_field].values):
        diff = (v - from_date).astype(dtype)
        ts_type = str(diff).split(" ")[-1]
        res[i] = diff.astype(np.int64)

    df[prefix + ts_type] = res
    return df


def plotMovingAverage(series, window, plot_intervals=False, conf_interval=1.96, plot_anomalies=False):
    """
    Plot moving average over a given pandas Series
    Args:
        series (pd.Series): A pandas Series with date index
        window (int): Rolling window size
        plot_intervals (bool): Show confidence intervals
        conf_interval (float): The confidence interval:
            0.95 = 95% interval with z = 1.96
            0.99 = 99% interval with z = 2.576
            0.995 = 99.5% interval with z = 2.807
            0.999 = 99.9% interval with z = 3.291
        plot_anomalies (bool): show anomalies

    Returns:

    """
    z = {0.95: 1.96, 0.99: 2.576, 0.995: 2.807, 0.999: 3.291}
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.title("Moving average\n window size = {}".format(window))
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
