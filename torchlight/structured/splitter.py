

def split_train_val(train_df, split_on: str,
                    split_train_range: tuple, split_val_range: tuple):
    """
    Splits train_df into train/val sets and let test_df untouched
    Args:
        train_df (Dataframe): The training dataframe (either a pandas or dask dataframe)
        split_on (str): The column on which to split
        split_train_range (tuple): The range of train split according to the passed column in split_on
        split_val_range (tuple): The range of test split according to the passed column in split_on
    Returns:
        tuple: The train and validation dataframes
    """
    val_df = train_df[split_on][split_train_range[0]:split_train_range[1]]
    train_df = train_df[split_on][split_val_range[0]:split_val_range[1]]
    return val_df, train_df
