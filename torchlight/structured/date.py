import numpy as np
import re


def add_datepart(df, field_name, transform_list=('Year', 'Month', 'Week', 'Day',
                                                 'Dayofweek', 'Dayofyear',
                                                 'Is_month_end', 'Is_month_start',
                                                 'Is_quarter_end', 'Is_quarter_start',
                                                 'Is_year_end', 'Is_year_start'),
                 drop=True, inplace=False):
    """
    Converts a column of df from a datetime64 to many columns containing
    the information from the date. `transform_list` is the list of transformations.
    A field "Elapsed" is always added to the resulting DataFrame.

    Args:
        df (pd.DataFrame, dd.DataFrame): A pandas or dask dataframe
        field_name (str): A string that is the name of the date column you wish to expand.
            Assumes the column is of type datetime64.
        transform_list (list): List of data transformations to add to the original dataset
        drop (bool): If true then the original date column will be removed.
        inplace (bool): Do the operation inplace or not
    Returns:
        A pandas or dask DataFrame depending on what was passed in
    """
    if not inplace:
        df = df.copy()
    field = df[field_name]
    targ_pre = re.sub('[Dd]ate$', '', field_name)
    for n in transform_list:
        df[targ_pre + n] = getattr(field.dt, n.lower())
        if df[targ_pre + n].dtype == np.int64:
            df[targ_pre + n] = df[targ_pre + n].astype(np.int16)
    df[targ_pre + 'Elapsed'] = field.astype(np.int64) // 10 ** 9
    if drop:
        df = df.drop(field_name, axis=1)
    return df


def add_elapsed(df, field_name, prefix, inplace=False):
    """
    Add durations to the dataframe. The dataframe must already
    be sorted in the right order as the duration interval depends on
    the time of a given row relative to the one before.
    Args:
        df (pd.DataFrame, dd.DataFrame): A pandas or dask dataframe
        field_name (str): A string that is the name of the column you wish to expand
        prefix (str): The prefix to add to the expanded field
        inplace (bool): Do the operation inplace or not

    Returns:

    """
    day1 = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    last_store = 0
    res = []

    if not inplace:
        df = df.copy()
    for s, v, d in zip(df.Store.values, df[field_name].values, df.Date.values):
        if s != last_store:
            last_date = np.datetime64()
            last_store = s
        if v:
            last_date = d
        res.append(((d - last_date).astype('timedelta64[D]') / day1).astype(int))
    df[prefix + field_name] = res
