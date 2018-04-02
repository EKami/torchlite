from datetime import datetime
import numpy as np


def time_split(df, val_start_date, val_stop_date, split_field=None):
    """
    Split on a datetime64 field. Take the date between val_start_date and val_stop_date as validation split
    and leave the rest for as the train set
    Args:
        df (DataFrame): A pandas DataFrame
        val_start_date (datetime): A datetime date (E.g datetime.datetime(2014, 8, 1))
        val_stop_date (datetime): A datetime date (E.g datetime.datetime(2014, 9, 17))
        split_field (str, None): The column name contained the datetime64 values, or None to use the index

    Returns:

    """
    if split_field is None:
        val_idxs = np.flatnonzero((df.index <= val_stop_date) & (df.index >= val_start_date))
    else:
        val_idxs = np.flatnonzero((df[split_field] <= val_stop_date) & (df[split_field] >= val_start_date))
    return split_by_idx(val_idxs, df)


def split_by_idx(idxs, df):
    mask = np.zeros(len(df), dtype=bool)
    mask[np.array(idxs)] = True
    return df[mask], df[~mask]
