import numpy as np
import re


def add_datepart(df, fldname, transform_list=('Year', 'Month', 'Week', 'Day',
                                              'Dayofweek', 'Dayofyear',
                                              'Is_month_end', 'Is_month_start',
                                              'Is_quarter_end', 'Is_quarter_start',
                                              'Is_year_end', 'Is_year_start'),
                 drop=True, inplace=False):
    """
    Converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.

    Args:
        df (pd.DataFrame, dd.DataFrame): A pandas or dask dataframe
        fldname (str): A string that is the name of the date column you wish to expand.
            Assumes the column is of type datetime64.
        transform_list (list): List of data transformations to add to the original dataset
        drop (bool): If true then the original date column will be removed.
        inplace (bool): Do the operation inplace or not
    Returns:
        A pandas or dask DataFrame depending on what was passed in
    """
    if not inplace:
        df = df.copy()
    fld = df[fldname]
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in transform_list:
        df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop:
        df = df.drop(fldname, axis=1)
    return df
