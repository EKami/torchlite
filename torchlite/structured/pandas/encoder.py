from sklearn.preprocessing.data import StandardScaler
from dask.diagnostics import ProgressBar
import pandas as pd

import dask.dataframe as dd
from tqdm import tqdm
import numpy as np
from pandas.api.types import is_numeric_dtype


class EncoderBlueprint:
    def __init__(self):
        """
        This class keeps all the transformations that went through
        the encoding of a dataframe for later use on another dataframe
        """
        self.scaler = None
        self.na_dict = None
        self.categ_var_map = None

    def save_categ_vars_map(self, df):
        if not self.categ_var_map:
            self.categ_var_map = {}
        for col_name, values in df.items():
            if df[col_name].dtype.name == 'category':
                self.categ_var_map[col_name] = values


def fix_missing(df, col, name, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.

    Args:
        df (DataFrame): The data frame that will be changed.
        col: The column of data to fix by filling in missing data.
        name: The name of the new filled column in df.

    na_dict (dict): A dictionary of values to create na's of and the value to insert. If
        name is not a key of na_dict the median will fill any missing data. Also
        if name is not a key of na_dict and there is no missing data in col, then
        no {name}_na column is not created.
    """
    col_c = col
    if is_numeric_dtype(col):
        # TODO: What if a NAN are found in the test set and not in the train set?
        # https://github.com/fastai/fastai/issues/74
        if pd.isnull(col).sum() or (name in na_dict):
            filler = na_dict[name] if name in na_dict else col_c.median()
            na_dict[name] = filler
            df[name + '_na'] = col.isnull()
            df[name] = col.fillna(filler)
    return na_dict


def _fix_na(df, na_dict):
    columns = df.columns
    if na_dict is None:
        na_dict = {}
    print("Calculating NA...")

    print("--- Fixing NA values ({} passes) ---".format(len(columns)))
    for c in tqdm(columns, total=len(columns)):
        na_dict = fix_missing(df, df[c], c, na_dict)

    print("List of NA columns fixed: {}".format(list(na_dict.keys())))
    return df, na_dict


def scale_vars(df, scaler=None):
    # TODO Try RankGauss: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
    num_cols = [n for n in df.columns if is_numeric_dtype(df[n])]
    # This previous transformation to float32 is very important
    df[num_cols] = df[num_cols].astype(np.float32)
    if scaler is None:
        scaler = StandardScaler().fit(df[num_cols].as_matrix())
    df[num_cols] = scaler.transform(df[num_cols])
    df[num_cols] = df[num_cols].astype(np.float32)
    print("List of scaled columns: {}".format(num_cols))
    return scaler


def get_all_non_numeric(df):
    non_num_cols = []
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            non_num_cols.append(col)
    return non_num_cols


def apply_encoding(df, cont_features, categ_features, scale_continuous=False,
                   categ_encoding="categorical", encoder_blueprint=None):
    """
        Changes the passed dataframe to an entirely numeric dataframe and return
    a new dataframe with the encoded features.
    The features from the `dataframes` parameter which are not passed neither in `numeric_features`
    nor in `categ_features` are just ignored for the resulting dataframe.
    The columns which are listed in `numeric_features` and `categ_features` and not present in the
    dataframe are also ignored.
    Args:
        df (DataFrame): DataFrame on which to apply the encoding
        cont_features (list): The list of features to encode as continuous values.
        categ_features (list): The list of features to encode as categorical features.
        scale_continuous (bool): Whether or not to scale the *continuous variables* on a standard scaler
                (mean substraction and standard deviation division).
                All features types will be encoded as float32.
        categ_encoding (str): The type of encoding for the categorical variables.
            - "Categorical": Encode the categorical variables as pandas "categorical" type
                (turn them to discrete codes).
            - "OneHot": Turn the categorical variables to onehot encoding (pandas dummy variables).
            /!\ The specific encodings from this parameter will be ignored if an encoder_blueprint
            has been passed as well. Instead the encoding from encoder_blueprint will be used.
        encoder_blueprint (EncoderBlueprint): An encoder blueprint which map its encodings to
            the passed df. Typically the first time you run this method you won't have any, a new
            EncoderBlueprint will be returned from this function that you need to pass in to this
            same function next time you want the same encoding to be applied to a different dataframe.
            E.g:
                train_df, encoderBlueprint = apply_encoding(train_df, contin_vars, cat_vars, do_scale=True)
                test_df, _ = apply_encoding(test_df, contin_vars, cat_vars,
                                            do_scale=True, encoderBlueprint=encoderBlueprint)
    Returns:
        DataFrame, dict, scale_mapper, sklearn_pandas.dataframe_mapper.DataFrameMapper:
            Returns:
                 - df: The original dataframe transformed according to the function parameters
                 - EncoderBlueprint: Contains all the encoder transformations such as
                    the list of columns found with NA values in the given df or the
                    mapper for the values scaling
            You usually want to keep EncoderBlueprint to use it for a similar dataset
            on which you want the values to be on the same scale/have the same missing columns
            and have the same categorical codes.
    """

    if isinstance(df, dd.DataFrame):
        with ProgressBar():
            print("Turning dask DataFrame into pandas DataFrame")
            df = df.compute()

    categ_encoding = categ_encoding.lower()
    encoder_blueprint = encoder_blueprint if encoder_blueprint else EncoderBlueprint()
    all_feat = categ_features + cont_features
    missing_col = [col for col in df.columns if col not in all_feat]
    df = df[[feat for feat in all_feat if feat in df.columns]].copy()

    df, encoder_blueprint.na_dict = _fix_na(df, encoder_blueprint.na_dict)
    print("Warning: Missing columns: {}, dropping them...".format(missing_col))
    print("Categorizing features {}".format(categ_features))
    # If the categorical mapping exists
    if encoder_blueprint.categ_var_map:
        for col_name, values in df.items():
            if col_name in categ_features:
                var_map = encoder_blueprint.categ_var_map
                print("Encoding categorical feature {}...".format(col_name))
                df[col_name] = pd.Categorical(values,
                                              categories=var_map[col_name].cat.categories,
                                              ordered=True)
    else:
        for feat in tqdm(categ_features, total=len(categ_features)):
            if feat in df.columns:
                if categ_encoding == 'onehot':
                    # TODO newly created onehot columns are not turned to categorical
                    # TODO newly created onehot should be saved into encoderBlueprint (see sklearn OneHotEncoder)
                    df = pd.get_dummies(df, columns=[feat], dummy_na=True)

                # Transform all types of categorical columns to pandas category type
                # Usually useful to make embeddings or keep the columns as continuous
                df[feat] = df[feat].astype('category').cat.as_ordered()

    # Scale continuous vars
    if scale_continuous:
        encoder_blueprint.scaler = scale_vars(df, encoder_blueprint.scaler)

    # Save categorical codes into encoderBlueprint
    encoder_blueprint.save_categ_vars_map(df)

    for name, col in df.items():
        if not is_numeric_dtype(col):
            df[name] = col.cat.codes + 1

    non_num_cols = get_all_non_numeric(df)
    nan_ratio = df.isnull().sum() / len(df)

    print("------- Dataframe of len {} summary -------\n".format(len(df.columns)))
    for col, nan, dtype in zip(df.columns, nan_ratio.values, df.dtypes.values):
        print("Column {:<30}:\t dtype: {:<10}\t NaN ratio: {}".format(col, str(dtype), nan))
    if len(non_num_cols) > 0 and nan_ratio > 0:
        raise Exception("Not all columns are numeric or NaN has been found: {}.".format(non_num_cols))

    print("---------- Preprocessing done -----------")
    return df, encoder_blueprint
