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
