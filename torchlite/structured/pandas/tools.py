import numpy as np


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
