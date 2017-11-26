import os
from dask.diagnostics import ProgressBar
import dask
import dask.dataframe as dd
import pandas as pd
from multiprocessing import cpu_count
import gc


def _save_split(train_df, split_df, name, split_file):
    if os.path.exists(split_file):
        print(f"{name} split already exits")
        return

    with ProgressBar():
        print(f"Saving {name} split to file:")
        if isinstance(train_df, dask.dataframe.core.DataFrame):
            split_df = split_df.compute()
        split_df.reset_index().to_feather(split_file)


def _read_split(split_file):
    f = pd.read_feather(split_file, nthreads=cpu_count())
    f.set_index(f.columns[0], inplace=True)
    return f


def split_train_val(train_df, split_on, split_train_range, split_val_range, output_dir):
    """
    Splits train_df into train/val sets
    Args:
        train_df (Dataframe): The training dataframe (either a pandas or dask dataframe)
        split_on (str): The column on which to split
        split_train_range (tuple): The range of train split according to the passed column in split_on
        split_val_range (tuple, None): The range of test split according to the passed column in split_on
        output_dir (str): The output dir where to sve the splits (their name will be based on the split
        names and won't be regenerated if the files already exist)
    Returns:
        tuple: 2 pandas DataFrame (train_split_file_x, val_split_file_x)
    """

    print("Generating splits...")
    train_split_file = os.path.join(output_dir, "split_train_" + str(split_train_range[0]) +
                                    "_to_" + str(split_train_range[1]) + ".feather")

    train_split = train_df[(train_df[split_on] >= split_train_range[0]) &
                           (train_df[split_on] <= split_train_range[1])]

    _save_split(train_df, train_split, "Train", train_split_file)
    gc.collect()
    train_split = _read_split(train_split_file)
    if split_val_range:
        val_split_file = os.path.join(output_dir, "split_val_" + str(split_val_range[0]) +
                                      "_to_" + str(split_val_range[1]) + ".feather")
        val_split = train_df[(train_df[split_on] >= split_val_range[0]) &
                             (train_df[split_on] <= split_val_range[1])]

        _save_split(train_df, val_split, "Val", val_split_file)
        gc.collect()
        val_split = _read_split(val_split_file)
        return train_split, val_split
    return train_split
