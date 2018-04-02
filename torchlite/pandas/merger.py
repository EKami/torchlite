def join_df(left_df, right_df, left_on, right_on=None, suffix='_y'):
    """
    Join two DataFrame together with a left join
    Args:
        left_df (DataFrame): The DataFrame on which right_df will be merged
        right_df (DataFrame): The DataFrame which will get merged into left_df
        left_on (str, list): The column name or a list of column names of left_df used for joining
        right_on (str, None): The column name of right_df used for joining or None if the column
            name is the same as left_on
        suffix (str, None):  Suffix to apply the merged column name or None for no suffix

    Returns:
        DataFrame: The merged DataFrame
    """
    if right_on is None:
        right_on = left_on
    return left_df.merge(right_df, how='left', left_on=left_on, right_on=right_on, suffixes=("", suffix))


def join_mult_df(left_df: list, right_df: list, left_on, right_on=None, suffixes=None):
    """
    Merges Dataframes from from_df to on_df. This allow
    for efficient merging of tables

    Args:
        left_df (list): List of DataFrames which will get metadata merged from from_df
        right_df (list): List of DataFrames to merge into on_df
        left_on (list): The column names of left_df used for joining (must be in the same order as the passed df)
        right_on (list, None): The column names of right_df used for joining or None if the column
            names are the same as left_on (must be in the same order as the passed df)
        suffixes (list):  List of suffix to apply to each merged column names
            (must be in the same order as the passed df)

        Example:
            join_mult_df([sales_df], [shops_df, items_df], ["shop_id", "item_id"])
    Returns:
        list: The merged DataFrame
    """
    if right_on is None:
        right_on = [None] * len(right_df)
    if suffixes is None:
        suffixes = [None] * len(right_df)

    rows_count = sum([df.shape[0] for df in left_df])
    res_df = [None] * len(left_df)
    for i, ldf in enumerate(left_df):
        res_df[i] = ldf
        for rdf, lon, ron, suffix in zip(right_df, left_on, right_on, suffixes):
            res_df[i] = join_df(res_df[i], rdf, lon, ron, suffix)

    # Ensure all df has the same number of row
    assert sum([df.shape[0] for df in res_df]) == rows_count, "Error: left_df size has changed during merging"
    return res_df


class CatSplit:
    def __init__(self, df_list):
        """
        A helper class used to join multiple DataFrame together and split
        them back. Useful when you want to apply an operation to a bunch of
        DataFrame at once (like train and test splits) and then split them
        back after the operation is done

        Args:
            df_list (list): A list of DataFrames
        """
        pass

    def get_joined(self):
        """
        Returns torch unique DataFrame resulting of all the DataFrames passed in the constructor
        Returns:
            DataFrame: A pandas DataFrame
        """
        return None

    def set_joined(self, df):
        """
        Set back the joined DataFrames.
        /!\ Do not try to use DataFrames which does not originates from get_joined()
        Args:
            df (DataFrame): A joined DataFrame
        """
        pass

    def get_splits(self):
        """
        Returns the split DataFrames passed in the constructor with the applied transformations
        Returns:
            list: A list of DataFrame
        """