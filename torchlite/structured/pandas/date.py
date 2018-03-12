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
        df (pd.DataFrame, dd.DataFrame): A pandas or dask dataframe
        field_name (str): A string that is the name of the date column you wish to expand.
            Assumes the column is of type datetime64 if df is a dask dataframe
        transform_list (list): List of data transformations to add to the original dataset
        drop (bool): If true then the original date column will be removed
        inplace (bool): If the operations are done inplace or not
        date_format (str): The datetime format for parsing
    Returns:
        A pandas or dask DataFrame depending on what was passed in
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
    df[targ_pre + 'Elapsed'] = field.astype(np.int64) // 10 ** 9
    if drop:
        df = df.drop(field_name, axis=1)
    return df
