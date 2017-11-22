import os
import sklearn
from sklearn.preprocessing.data import StandardScaler
from sklearn_pandas.dataframe_mapper import DataFrameMapper
import warnings
import pandas as pd

import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np
from pandas.api.types import is_numeric_dtype


def fix_missing(df, col, name, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.

    Parameters:
    -----------
    df: The data frame that will be changed.

    col: The column of data to fix by filling in missing data.

    name: The name of the new filled column in df.

    na_dict: A dictionary of values to create na's of and the value to insert. If
        name is not a key of na_dict the median will fill any missing data. Also
        if name is not a key of na_dict and there is no missing data in col, then
        no {name}_na column is not created.


    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2

    >>> fix_missing(df, df['col1'], 'col1', {})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1     2    2    True
    2     3    2   False


    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2

    >>> fix_missing(df, df['col1'], 'col1', {'col1' : 500})
    >>> df
       col1 col2
    0     1    5
    1   500    2
    2     3    2
    """
    with ProgressBar():
        col_c = col
        if isinstance(col, dask.dataframe.core.Series):
            col_c = col.compute()
        if is_numeric_dtype(col):
            if pd.isnull(col).sum() or (name in na_dict):
                filler = na_dict[name] if name in na_dict else col_c.median()
                na_dict[name] = filler
                df[name + '_na'] = col.isnull()
                df[name] = col.fillna(filler)
    return na_dict


def _fix_na(df, na_dict, verbose):
    columns = df.columns
    if verbose:
        print("Calculating NA...")
        if isinstance(df, dask.dataframe.core.DataFrame):
            print("------------ Ratio of NA values -------------\n" +
                  str(df.isnull().sum().compute() / len(df)))
        else:
            print("------------ Ratio of NA values -------------\n" +
                  str(df.isnull().sum().compute() / len(df)))

    print(f"--- Fixing NA values ({len(columns)} passes) ---")
    for i, c in enumerate(columns):
        print(f"Pass {i+1}/{len(columns)}")
        na_dict = fix_missing(df, df[c], c, na_dict)

    if verbose:
        print("Calculating NA...")
        if isinstance(df, dask.dataframe.core.DataFrame):
            print("---------- New Ratio of NA values -----------\n" +
                  str(df.isnull().sum().compute() / len(df)))
        else:
            print("---------- New Ratio of NA values -----------\n" +
                  str(df.isnull().sum().compute() / len(df)))
    return df


def scale_vars(df, mapper=None):
    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n], StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper


def apply_encoding(df, numeric_features, categ_features,
                   output_file, do_scale=False,
                   na_dict=None, verbose=False):
    """
    Apply encoding to the passed dataframes and return a new dataframe with the encoded features.
    /!\ At this point only pandas dataframes are accepted

    The features from the `dataframes` parameter which are not passed neither in `numeric_features`
    nor in `categ_features` are just ignored for the resulting dataframe.
    The columns which are listed in `numeric_features` and `categ_features` and not present in the
    dataset are also ignored.
    Args:
        df (DatFrame): DataFrame on which to apply the encoding
        numeric_features (dict): The list of features to encode as numeric values. Types can
            be any numpy type. For instance: {"index": np.int32, "sales": np.float32, ...}
        categ_features (dict): The list of features to encode as categorical features.
            The types can be of the following:
                - OneHot
                - Continuous
            Example: {"store_type": "OneHot", "holiday_type": "Continuous", ...}
        do_scale (bool): Whether or not to scale the continuous variables
        na_dict (dict, None): a dictionary of na columns to add. Na columns are also added if there
        are any missing values.
        output_file: (str, None): The output file where the encoded DataFrame will
            be stored. If this file already exist then the existing file is opened and
            returned as DataFrame from this function.
        verbose (bool): Whether to make this function verbose. If not set the function still
        returns few messages

    Returns:
        str: A path to a pandas dataframe with encoded features
    """
    # Check if the encoding has already been generated with
    # https://stackoverflow.com/questions/31567401/get-the-same-hash-value-for-a-pandas-dataframe-each-time

    na_dict = na_dict if na_dict else {}
    if os.path.exists(output_file):
        print("Encoding files already generated")
        return pd.read_feather(output_file)

    df = _fix_na(df, na_dict, verbose)
    if do_scale:
        mapper = scale_vars(df)

    columns = df.columns
    categ_cols = list(categ_features.keys())
    print(f"Categorizing features {categ_cols}")
    df[categ_cols].apply(lambda x: x.astype('category'))

    all_feat = list(numeric_features.keys()) + categ_cols
    missing_col = [col for col in columns if col not in all_feat]
    print(f"Warning: Missing columns: {missing_col}, dropping them...")
    for k, v in numeric_features.items():
        if k in columns:
            df[k] = df[k].astype(v)

    for i, (k, v) in enumerate(categ_features.items()):
        print(f"Categ transform {i+1}/{len(categ_cols)}")
        if k in columns:
            if isinstance(v, str):
                v = v.lower()
            if v == 'onehot':
                df = pd.get_dummies(df, columns=[k])

    df.to_feather(output_file)
    return df
