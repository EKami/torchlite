from dask.diagnostics import ProgressBar
# TODO try pandas on ray: https://rise.cs.berkeley.edu/blog/pandas-on-ray/
import pandas as pd

import dask.dataframe as dd
from tqdm import tqdm
import numpy as np
from pandas.api.types import is_numeric_dtype


class EncoderBlueprint:
    def __init__(self, continuous_scaler=None):
        """
        This class keeps all the transformations that went through
        the encoding of a dataframe for later use on another dataframe

        Args:
            continuous_scaler (None, Scaler): None or a scaler from sklearn.preprocessing.data
                All features types will be encoded as float32.
                An sklearn StandardScaler() will fit most common cases.
                For a more robust scaling with outliers take a look at RankGauss:
                    https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
                and rankdata:
                    https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.rankdata.html

            Reference -> http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
        """
        self.continuous_scaler = continuous_scaler
        self.na_dict = None
        self.categ_var_map = None
        self.is_scaler_fit = False

    def scale_vars(self, df):
        if self.continuous_scaler is None:
            return
        num_cols = [n for n in df.columns if is_numeric_dtype(df[n])]
        # /!\ This previous transformation to float32 is very important
        df[num_cols] = df[num_cols].astype(np.float32)
        if not self.is_scaler_fit:
            self.continuous_scaler.fit(df[num_cols].as_matrix())
            self.is_scaler_fit = True
        df[num_cols] = self.continuous_scaler.transform(df[num_cols])
        df[num_cols] = df[num_cols].astype(np.float32)
        print("List of scaled columns: {}".format(num_cols))

    def save_categ_vars_map(self, df):
        if not self.categ_var_map:
            self.categ_var_map = {}
        for col_name, values in df.items():
            if df[col_name].dtype.name == 'category':
                self.categ_var_map[col_name] = values


class BaseEncoder:
    def _fix_missing(self, df, col, name, na_dict):
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
            # TODO xgboost can fix missing values itself
            # TODO: What if a NAN are found in the test set and not in the train set?
            # https://github.com/fastai/fastai/issues/74
            if pd.isnull(col).sum() or (name in na_dict):
                filler = na_dict[name] if name in na_dict else col_c.median()
                na_dict[name] = filler
                df[name + '_na'] = col.isnull()
                df[name] = col.fillna(filler)
        return na_dict

    def _fix_na(self, df, na_dict):
        columns = df.columns
        if na_dict is None:
            na_dict = {}
        print("Calculating NA...")

        print("--- Fixing NA values ({} passes) ---".format(len(columns)))
        for c in tqdm(columns, total=len(columns)):
            na_dict = self._fix_missing(df, df[c], c, na_dict)

        print("List of NA columns fixed: {}".format(list(na_dict.keys())))
        return df, na_dict

    def _get_all_non_numeric(self, df):
        non_num_cols = []
        for col in df.columns:
            if not is_numeric_dtype(df[col]):
                non_num_cols.append(col)
        return non_num_cols


class TreeEncoder(BaseEncoder):
    def __init__(self, df, cont_features, categ_features, encoder_blueprint):
        """
            An encoder used for tree based models (RandomForests, GBTs) as well
            as deep neural networks with embeddings (DNN)

        Args:
            df (DataFrame, dd.DataFrame): The DataFrame to manipulate.
            The DataFrame will be copied and the original one won't be affected by the changes
            in this class
            cont_features (list): The list of features to encode as continuous values.
            categ_features (list): The list of features to encode as categorical features.
            encoder_blueprint (EncoderBlueprint): An encoder blueprint which map its encodings to
                the passed df.
        """
        if isinstance(df, dd.DataFrame):
            with ProgressBar():
                print("Turning dask DataFrame into pandas DataFrame")
                df = df.compute()

        self.df = df.copy()
        self.categ_features = categ_features
        self.cont_features = cont_features
        self.encoder_blueprint = encoder_blueprint if encoder_blueprint else EncoderBlueprint()

    def apply_encoding(self):
        """
        Changes the passed DataFrame to an entirely numeric DataFrame and return
        a new DataFrame with the encoded features.
        The features from the `dataframes` parameter which are not passed neither in the constructor
        as `cont_features` nor as `categ_features` are just ignored for the resulting DataFrame.
        The columns which are listed in `cont_features` and `categ_features` and not present in the
        DataFrame are also ignored.
        This method will do the following:
            - Remove NaN values by using the feature mean and adding a feature_missing feature
            - Scale the continuous values according to the EncoderBlueprint scaler
            - Encode categorical features to numeric types
        What it doesn't do:
            - Deal with outliers
        Returns:
            (DataFrame, EncoderBlueprint):
                Returns:
                     - df: The original DataFrame transformed according to the function parameters
                     - EncoderBlueprint: Contains all the encoder transformations such as
                        the list of columns found with NA values in the given df or the
                        mapper for the values scaling
                You usually want to keep EncoderBlueprint to use it for a similar dataset
                on which you want the values to be on the same scale/have the same missing columns
                and have the same categorical codes.
                E.g:
                        train_df, encoder_blueprint = TreeEncoder(train_df, continuous_vars, cat_vars,
                                                                  EncoderBlueprint(StandardScaler())).apply_encoding()
                        test_df, _ = TreeEncoder(test_df, continuous_vars, cat_vars,
                                                encoder_blueprint=encoder_blueprint).apply_encoding()
        """

        all_feat = self.categ_features + self.cont_features
        missing_col = [col for col in self.df.columns if col not in all_feat]
        df = self.df[[feat for feat in all_feat if feat in self.df.columns]].copy()

        df, self.encoder_blueprint.na_dict = self._fix_na(df, self.encoder_blueprint.na_dict)
        print("Warning: Missing columns: {}, dropping them...".format(missing_col))
        print("Categorizing features {}".format(self.categ_features))
        # If the categorical mapping exists
        if self.encoder_blueprint.categ_var_map:
            for col_name, values in df.items():
                if col_name in self.categ_features:
                    var_map = self.encoder_blueprint.categ_var_map
                    print("Encoding categorical feature {}...".format(col_name))
                    df[col_name] = pd.Categorical(values,
                                                  categories=var_map[col_name].cat.categories,
                                                  ordered=True)
        else:
            for feat in tqdm(self.categ_features, total=len(self.categ_features)):
                if feat in df.columns:
                    # Transform all types of categorical columns to pandas category type
                    # Usually useful to make embeddings or keep the columns as continuous
                    df[feat] = df[feat].astype('category').cat.as_ordered()

        # Scale continuous vars
        self.encoder_blueprint.scale_vars(df)

        # Save categorical codes into encoderBlueprint
        self.encoder_blueprint.save_categ_vars_map(df)

        for name, col in df.items():
            if not is_numeric_dtype(col):
                df[name] = col.cat.codes + 1

        non_num_cols = self._get_all_non_numeric(df)
        nan_ratio = df.isnull().sum() / len(df)

        print("------- Dataframe of len {} summary -------\n".format(len(df.columns)))
        for col, nan, dtype in zip(df.columns, nan_ratio.values, df.dtypes.values):
            print("Column {:<30}:\t dtype: {:<10}\t NaN ratio: {}".format(col, str(dtype), nan))
        if len(non_num_cols) > 0 and nan_ratio > 0:
            raise Exception("Not all columns are numeric or NaN has been found: {}.".format(non_num_cols))

        print("---------- Preprocessing done -----------")
        return df, self.encoder_blueprint
