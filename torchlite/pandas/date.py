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


def add_lag(df_list, column, by=None, t=1):
    """
    Add lag values to the df in df_list
    Args:
        df_list (list): A list of pandas DataFrame. The DataFrames must
            have matching columns as they will be concatenated together to
            calculate the lags and then split back.
            /!\ Pass the DataFrame in order, for instance: [train_df, val_df, test_df]
        column (str): The column on which to apply the lag
        by (list, str, None): List of columns to group by before applying the lag
        t (int): Lag steps

    Returns:
        list: List of pandas DataFrames similar to the one passed in parameter
        with lag values.
    """
    df_col_count = np.array([0] + [df.shape[0] for df in df_list]).cumsum()
    df_col_names = [(list(df.columns), list(df.dtypes)) for df in df_list]

    df = pd.DataFrame()
    for df_c in df_list:
        df = pd.concat([df, df_c], axis=0)

    if by is not None:
        if type(by) == str:
            by = [by]
        shifted_col = df.groupby(by)[column].shift(t)
        shifted_col.rename(column + '_BY[' + '_'.join(by) + "]_lag_" + str(t), inplace=True)
    else:
        shifted_col = df[column].shift(t)
        shifted_col.rename(column + "_lag_" + str(t), inplace=True)
    df = pd.concat([df, shifted_col], axis=1)

    # Restore DataFrames
    df_list = [df.iloc[df_col_count[i]:df_col_count[i + 1]] for i in range(len(df_col_count) - 1)]
    df_list = [df[cols + [shifted_col.name]].astype(dict(zip(cols, dtypes))) for df, (cols, dtypes) in
               zip(df_list, df_col_names)]

    return df_list
