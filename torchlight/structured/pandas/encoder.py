import os
import sklearn
from sklearn.preprocessing.data import StandardScaler
from sklearn_pandas.dataframe_mapper import DataFrameMapper
from dask.diagnostics import ProgressBar
import warnings
import pandas as pd
import types

import dask
import dask.dataframe as dd
from tqdm import tqdm
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
    col_c = col
    if is_numeric_dtype(col):
        # TODO: What if a NAN is found in the test set and not in the train set?
        # https://github.com/fastai/fastai/issues/74
        if pd.isnull(col).sum() or (name in na_dict):
            filler = na_dict[name] if name in na_dict else col_c.median()
            na_dict[name] = filler
            df[name + '_na'] = col.isnull()
            df[name] = col.fillna(filler)
    return na_dict


def _fix_na(df, na_dict, verbose, name):
    columns = df.columns
    if verbose:
        print("Calculating NA...")
        print(f"---------- Ratio of NA values for {name} with {len(df.columns)} features -----------\n" +
              str(df.isnull().sum() / len(df)))

    print(f"--- Fixing NA values ({len(columns)} passes) ---")
    for c in tqdm(columns, total=len(columns)):
        na_dict = fix_missing(df, df[c], c, na_dict)

    print(f"List of NA columns fixed: {list(na_dict.keys())}")
    if verbose:
        print("Calculating NA...")
        print(f"-------- New Ratio of NA values for {name} with {len(df.columns)} features ---------\n" +
              str(df.isnull().sum() / len(df)))
    return df, na_dict


def scale_vars(df, mapper=None):
    # TODO Try RankGauss: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    for col in df:
        df[col] = df[col].astype(np.float32)
    if mapper is None:
        map_f = [([n], StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df).astype(np.float32)
    return mapper


def get_all_non_numeric(df):
    non_num_cols = []
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            non_num_cols.append(col)
    return non_num_cols


def apply_encoding(df, cont_features, categ_features,
                   do_scale=False, na_dict=None, verbose=True,
                   name='dataframe', check_all_num=True):
    """
    Apply encoding to the passed dataframes and return a new dataframe with the encoded features.

    The features from the `dataframes` parameter which are not passed neither in `numeric_features`
    nor in `categ_features` are just ignored for the resulting dataframe.
    The columns which are listed in `numeric_features` and `categ_features` and not present in the
    dataset are also ignored.
    Args:
        df (DataFrame): DataFrame on which to apply the encoding
        cont_features (dict, list): The list of features to encode as numeric values.
            If given as a list all continuous features are encoded as float32.
            If given as a dictionary types can be any numpy type.
            For instance: {"index": np.int32, "sales": np.float32, ...}.
            The type can also be a function with signature: (df: DataFrame, field: Series) -> DataFrame
        categ_features (dict, list): The list of features to encode as categorical features.
            If given as a list all categorical features are encoded as pandas 'categorical' (continuous vars).
            If given as a dictionary the types can be of the following:
                - OneHot
                - Continuous
                - As_is (don't change the variable)
            Example: {"store_type": "OneHot", "holiday_type": "Continuous", ...}
        do_scale (bool, sklearn_pandas.dataframe_mapper.DataFrameMapper):
            If of type bool: Whether or not to scale the continuous variables on a standard scaler
                (mean substraction and standard deviation division).
            If of type DataFrameMapper: A mappper variable to scale the features according to the
                given predefined mapper. Is mostly used on the val/test sets to use the same mapper
                returned by the train set.
            In all case of transformation the results will be of type float32.
        na_dict (dict, None): a dictionary of na columns to add. Na columns are also added if there
            are any missing values.
        verbose (bool): Whether to make this function verbose. If not set the function still
            returns few messages
        name (str): Only useful for the verbose output
        check_all_num (bool): Check if all the features are numeric, raise an error if they aren't
    Returns:
        DataFrame, dict, scale_mapper, sklearn_pandas.dataframe_mapper.DataFrameMapper:
            Returns:
                 - df: The original dataframe transformed according to the function parameters
                 - na_dict: Containing the list of columns found with NA values in the given df
                 - scale_mapper: The mapper for the values scaling
            You usually want to keep na_dict and scale_mapper to use them for a similar dataset
            on which you want the values to be on the same scale/have the same missing columns.
            For example you would recover na_dict and scale_mapper and pass them in for
            the test set.
    """

    if isinstance(df, dd.DataFrame):
        with ProgressBar():
            print("Turning dask DataFrame into pandas DataFrame")
            df = df.compute()

    # Turn categ_feat to a dict if it's a list
    if isinstance(categ_features, list):
        di = {}
        for key in categ_features:
            di[key] = 'continuous'
        categ_features = di

    # Turn cont_features to a dict if it's a list
    if isinstance(cont_features, list):
        di = {}
        for key in cont_features:
            di[key] = np.float32
        cont_features = di

    categ_feat = list(categ_features.keys())
    all_feat = list(cont_features.keys()) + categ_feat
    na_dict = na_dict if na_dict else {}
    df_columns = df.columns
    missing_col = [col for col in df_columns if col not in all_feat]
    df = df[[feat for feat in all_feat if feat in df_columns]].copy()

    df, na_dict = _fix_na(df, na_dict, verbose, name)

    print(f"Categorizing features {categ_feat}")
    df[categ_feat].apply(lambda x: x.astype('category'))

    print(f"Warning: Missing columns: {missing_col}, dropping them...")
    for k, v in cont_features.items():
        if k in df_columns:
            if isinstance(v, types.FunctionType):
                df = v(df)
            else:
                df[k] = df[k].astype(v)

    for k, v in tqdm(categ_features.items(), total=len(categ_features.items())):
        if k in df_columns:
            if isinstance(v, str):
                v = v.lower()
            if isinstance(v, types.FunctionType):
                df = v(df)
            elif v == "as_is":
                df[k] = df[k]
            elif v == 'onehot':
                df = pd.get_dummies(df, columns=[k])
            # Treat pre-embedding as continuous variables which are meant to be transformed to embedding matrices
            elif v == 'continuous' or v == 'pre_embedding':
                df[k] = df[k].astype('category').cat.codes
    if do_scale:
        scale_mapper = scale_vars(df)
        print(f"List of scaled columns: {scale_mapper}")
    if check_all_num:
        non_num_cols = get_all_non_numeric(df)
        if len(non_num_cols) > 0:
            raise Exception(f"Not all columns are numeric: {non_num_cols}, DataFrame not saved.")
    print(f"------- Dtypes of {name} with {len(df.columns)} features -------\n" + str(df.dtypes))
    print("---------- Preprocessing done -----------")
    return df, na_dict, scale_mapper
