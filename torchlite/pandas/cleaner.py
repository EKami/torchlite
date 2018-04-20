import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_float_dtype, is_integer_dtype
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def replace_matches_in_column(df, column, string_to_match, min_ratio=90, limit=10):
    """
    Function used to replace strings in a DataFrame columns by other similar strings.
    Very similar strings (such as "Paris" and "Paris " have a 100 fuzzy ratio and will
    be merged together by this function.
    E.g:
        replace_matches_in_column(df=df, column='City', string_to_match="Paris")
    Args:
        df (DataFrame): The DataFrame with fuzzy words
        column (str): The column name
        string_to_match (str): The string for which you want the similar word to be searched
        min_ratio (int): A fuzzy ratio, the less it is, the more you'll replace the similar words
        by the given string_to_match.
        limit (int): The limit of fuzzy words to look for

    Returns:
        DataFrame: The passed DataFrame (note the original DataFrame will also be changed)
    """
    # get a list of unique strings
    strings = df[column].unique()

    # get the closest matches to our input string
    matches = process.extract(string_to_match, strings, limit=limit, scorer=fuzz.token_sort_ratio)

    # only get matches with a ratio > min_ratio
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]
    print("List of values to be replaced: {}".format(close_matches))

    # get the rows of all the close matches in our DataFrame
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches
    df.loc[rows_with_matches, column] = string_to_match
    return df


def adjust_data_types(df_list, inplace=False):
    """
    Adjust the data type of a list of pandas DataFrames to take less space
    in memory. For instance it will turn int64/float64 to int32/float32 if
    the conversion can be made (sanity checks will be made).
    Args:
        df_list (list): A list of pandas DataFrame
        inplace (bool): Do the operation inplace or not (inplace take less memory)
    Returns:
        list: The list of the same DataFrame but with types converted
    """
    if not inplace:
        df_list = [df.copy() for df in df_list]

    for df in df_list:
        for name, dtype in zip(df.dtypes.index, df.dtypes.values):
            if is_numeric_dtype(dtype):
                col = df[name]
                if is_float_dtype(dtype):
                    df[name] = pd.to_numeric(col, downcast='float')
                elif is_integer_dtype(dtype):
                    df[name] = pd.to_numeric(col, downcast='integer')

    return df_list


