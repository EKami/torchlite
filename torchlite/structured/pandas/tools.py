import numpy as np
from tqdm import tqdm
import pandas as pd


def count_missing(df_list):
    """
    Count the number of missing values in the passed DataFrame and return
    the ratio
    Args:
        df_list (list): List of DataFrames to

    Returns:
        float: A value between 0 and 1
    """
    total_missing = sum([sum(df.isnull().sum()) for df in df_list])
    total_cells = sum([np.product(df.shape) for df in df_list])

    return total_missing / total_cells


def get_duplicated_columns(df, drop_duplicates=False):
    """
    Get duplicated columns as a dictionary mapping between the duplicates.
    This implementation has tqdm to monitor the progress and is better fitted
    for large DataFrames.
    Args:
        df (Dataframe): The DataFrame to explore
        drop_duplicates (bool): Will drop the duplicates if set to True

    Returns:
        dict: A dictionary containing a mapping between the duplicated columns
    """
    enc_df = pd.DataFrame(index=df.index)
    for col in tqdm(df.columns, desc="Factorizing columns"):
        enc_df[col] = df[col].factorize()[0]

    dup_cols = {}
    for i, c1 in enumerate(tqdm(enc_df.columns, desc="Searching for duplicate columns")):
        for c2 in enc_df.columns[i + 1:]:
            if c2 not in dup_cols and np.all(enc_df[c1] == enc_df[c2]):
                dup_cols[c2] = c1

    if drop_duplicates:
        df.drop(dup_cols.keys(), axis=1, inplace=True)

    return dup_cols
