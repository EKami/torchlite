import os
from dask.diagnostics import ProgressBar
import dask
import dask.dataframe as dd
import pandas as pd


def split_train_val(train_df, split_on: str,
                    split_train_range: tuple, split_val_range: tuple, output_dir):
    """
    Splits train_df into train/val sets
    Args:
        train_df (Dataframe): The training dataframe (either a pandas or dask dataframe)
        split_on (str): The column on which to split
        split_train_range (tuple): The range of train split according to the passed column in split_on
        split_val_range (tuple): The range of test split according to the passed column in split_on
        output_dir (str): The output dir where to sve the splits (their name will be based on the split
        names and won't be regenerated if the files already exist)
    Returns:
        tuple: 2 pandas DataFrame (train_split_file_x, val_split_file_x)
    """
    val_split_file = os.path.join(output_dir, "split_val_" + str(split_val_range[0]) +
                                  "_to_" + str(split_val_range[1]) + ".feather")
    train_split_file = os.path.join(output_dir, "split_train_" + str(split_train_range[0]) +
                                    "_to_" + str(split_train_range[1]) + ".feather")

    if os.path.exists(val_split_file) and os.path.exists(train_split_file):
        print("Splits already generated")
        return pd.read_feather(train_split_file), pd.read_feather(val_split_file)

    print("Generating splits...")
    val_split = train_df[(train_df[split_on] >= split_val_range[0]) &
                         (train_df[split_on] <= split_val_range[1])]
    train_split = train_df[(train_df[split_on] >= split_train_range[0]) &
                           (train_df[split_on] <= split_train_range[1])]

    with ProgressBar():
        print(f"Saving 2 splits to file:")
        if isinstance(train_df, dask.dataframe.core.DataFrame):
            val_split = val_split.compute()
            train_split = train_split.compute()
        val_split.reset_index().to_feather(val_split_file)
        train_split.reset_index().to_feather(train_split_file)

    return train_split, val_split
