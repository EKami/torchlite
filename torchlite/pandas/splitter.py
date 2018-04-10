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
        split_field (str, None): The column name containing the datetime64 values, or None to use the index

    Returns:
        tuple: (train_df, val_df)
    """
    if split_field is None:
        val_df = df[(df.index <= val_stop_date) & (df.index >= val_start_date)]
    else:
        val_df = df[(df[split_field] <= val_stop_date) & (df[split_field] >= val_start_date)]

    train_df = df.drop(val_df.index)
    return train_df, val_df


def id_split(df, val_ids, split_field=None):
    """
    Split on an identifier field. Take the id between start_id and stop_id as validation split
    and leave the rest for as the train set.
    This kind of split is particularly useful when the predictions on the test set have to be
    made on unknown id from the train/val sets.
    Args:
        df (DataFrame): A pandas DataFrame
        val_ids (list): A list of ids to use in the validation set
        split_field (str, None): The column name containing the id values, or None to use the index

    Returns:
        tuple: (train_df, val_df)
    """
    # TODO finish
    pass
