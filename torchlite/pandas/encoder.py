# TODO try pandas on ray: https://rise.cs.berkeley.edu/blog/pandas-on-ray/
import pandas as pd

from tqdm import tqdm
import numpy as np
from pandas.api.types import is_numeric_dtype


class EncoderBlueprint:
    def __init__(self, numeric_scaler=None):
        """
        This class keeps all the transformations that went through
        the encoding of a dataframe for later use on another dataframe

        Args:
            numeric_scaler (None, Scaler): None or a scaler from sklearn.preprocessing.data for scaling
            numeric features
                All features types will be encoded as float32.
                An sklearn StandardScaler() will fit most common cases.
                For a more robust scaling with outliers take a look at RankGauss:
                    https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
                and rankdata:
                    https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.rankdata.html

            Reference -> http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
        """
        self.numeric_scaler = numeric_scaler
        self.na_dict = None
        self.is_scaler_fit = False
        self.categ_var_map = None
        self.column_names = None

    def scale_vars(self, df, cols):
        if self.numeric_scaler is None:
            return
        num_cols = [n for n in df.columns if is_numeric_dtype(df[n]) and n in cols]
        # /!\ This previous transformation to float32 is very important
        df[num_cols] = df[num_cols].astype(np.float32)
        if not self.is_scaler_fit:
            self.numeric_scaler.fit(df[num_cols].as_matrix())
            self.is_scaler_fit = True
        df[num_cols] = self.numeric_scaler.transform(df[num_cols])
        df[num_cols] = df[num_cols].astype(np.float32)
        print("List of scaled columns: {}".format(num_cols))


class BaseEncoder:
    def __init__(self, df, numeric_cols, categorical_cols, encoder_blueprint, fix_missing):
        self.df = df.copy()
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.blueprint = encoder_blueprint if encoder_blueprint else EncoderBlueprint()
        self.fix_missing = fix_missing

    def _check_integrity(self, df):
        """
        Check if the columns registered in the EncoderBlueprint are the same as the ones
        in the passed df

        Returns:
            bool: Return True if the columns match, raise an exception otherwise
        """
        if self.blueprint.column_names is None:
            self.blueprint.column_names = df.columns
            return True

        diff = list(set(self.blueprint.column_names) - set(df.columns))
        if len(diff) > 0:
            raise Exception("Columns in EncoderBlueprint and DataFrame do not match: {}".format(diff))

        if list(self.blueprint.column_names) == list(df.columns):
            return True

        raise Exception("Columns in EncoderBlueprint and DataFrame do not have the same order")

    def _get_all_non_numeric(self, df):
        non_num_cols = []
        for col in df.columns:
            if not is_numeric_dtype(df[col]):
                non_num_cols.append(col)
        return non_num_cols

    def _fix_missing(self, df, col, name, na_dict):
        raise NotImplementedError()

    def encode_categorical(self, df):
        raise NotImplementedError()

    def _fix_na(self, df, na_dict):
        """
        Fix the missing values (must be implemented in superclass)
        Args:
            df (DataFrame): The DataFrame to fix
            na_dict (dict): The NaN values mapping

        Returns:
            tuple: (df, na_dict)
        """
        columns = df.columns
        if na_dict is None:
            na_dict = {}
        print("Calculating NA...")

        print("--- Fixing NA values ({} passes) ---".format(len(columns)))
        for c in tqdm(columns, total=len(columns)):
            na_dict = self._fix_missing(df, df[c], c, na_dict)

        print("List of NA columns fixed: {}".format(list(na_dict.keys())))
        return df, na_dict

    def apply_encoding(self):
        """
        Changes the passed DataFrame to an entirely numeric DataFrame and return
        a new DataFrame with the encoded features.
        The features from the `dataframes` parameter which are not passed neither in the constructor
        as `numeric_cols` nor as `categorical_cols` are just ignored for the resulting DataFrame.
        The columns which are listed in `numeric_cols` and `categorical_cols` and not present in the
        DataFrame are also ignored.
        This method will do the following:
            - Remove NaN values by using the feature mean and adding a feature_missing feature
            - Scale the numeric values according to the EncoderBlueprint scaler
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
                    train_df, encoder_blueprint = TreeEncoder(train_df, num_vars, cat_vars,
                                                              EncoderBlueprint(StandardScaler())).apply_encoding()
                    test_df, _ = TreeEncoder(test_df, num_vars, cat_vars,
                                             encoder_blueprint=encoder_blueprint).apply_encoding()
        """
        all_feat = self.categorical_cols + self.numeric_cols
        missing_col = [col for col in self.df.columns if col not in all_feat]
        df = self.df[[feat for feat in all_feat if feat in self.df.columns]].copy()

        if self.fix_missing:
            df, self.blueprint.na_dict = self._fix_na(df, self.blueprint.na_dict)
        print("Warning: Missing columns: {}, dropping them...".format(missing_col))
        print("Categorizing features {}".format(self.categorical_cols))

        # Encode numeric features
        df = self.encode_categorical(df)

        # Scale numeric vars
        self.blueprint.scale_vars(df, self.numeric_cols)

        non_num_cols = self._get_all_non_numeric(df)
        nan_ratio = df.isnull().sum() / len(df)

        print("------- Dataframe of len {} summary -------\n".format(len(df.columns)))
        for col, nan, dtype in zip(df.columns, nan_ratio.values, df.dtypes.values):
            print("Column {:<30}:\t dtype: {:<10}\t NaN ratio: {}".format(col, str(dtype), nan))
        if len(non_num_cols) > 0 and nan_ratio > 0:
            raise Exception("Not all columns are numeric or NaN has been found: {}.".format(non_num_cols))

        self._check_integrity(df)
        print("---------- Preprocessing done -----------")
        return df, self.blueprint


class TreeEncoder(BaseEncoder):
    def __init__(self, df, numeric_cols, categorical_cols, encoder_blueprint, fix_missing=True):
        """
            An encoder used for tree based models (RandomForests, GBTs) as well
            as deep neural networks with categorical embeddings features (DNN)

        Args:
            df (DataFrame, dd.DataFrame): The DataFrame to manipulate.
            The DataFrame will be copied and the original one won't be affected by the changes
            in this class.
            numeric_cols (list): The list of columns to encode as numeric values.
            categorical_cols (list): The list of columns to encode as categorical.
            encoder_blueprint (EncoderBlueprint): An encoder blueprint which map its encodings to
                the passed df.
            fix_missing (bool): True to fix the missing values (will add a new feature `is_missing` and replace
                the missing value by its median). For some models like Xgboost you may want to set this value to False.
        """
        super().__init__(df, numeric_cols, categorical_cols, encoder_blueprint, fix_missing)

    def _fix_missing(self, df, col, name, na_dict):
        """ Fill missing data in a column of df with the median, and add a {name}_na column
        which specifies if the data was missing.

        Args:
            df (DataFrame): The data frame that will be changed.
            col (pd.Series): The column of data to fix by filling in missing data.
            name (str): The name of the new filled column in df.
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

    def encode_categorical(self, df):
        # If the categorical mapping exists
        if self.blueprint.categ_var_map:
            for col_name, values in df.items():
                if col_name in self.categorical_cols:
                    df[col_name] = pd.Categorical(values, ordered=True,
                                                  categories=self.blueprint.categ_var_map[col_name].cat.categories)
        else:
            self.blueprint.categ_var_map = {}
            for col in tqdm(self.categorical_cols, total=len(self.categorical_cols)):
                if col in df.columns:
                    # Transform all types of categorical columns to pandas category type
                    # Usually useful to make embeddings or keep the columns as numeric
                    df[col] = df[col].astype('category').cat.as_ordered()
                    self.blueprint.categ_var_map[col] = df[col]

        for name, col in df.items():
            if not is_numeric_dtype(col):
                df[name] = col.cat.codes + 1

        return df


class LinearEncoder(BaseEncoder):
    def __init__(self, df, numeric_cols, categorical_cols, encoder_blueprint):
        """
            An encoder used for linear based models (Linear/Logistic regression) as well
            as deep neural networks with one hot encoded values.

        Args:
            df (DataFrame, dd.DataFrame): The DataFrame to manipulate.
            The DataFrame will be copied and the original one won't be affected by the changes
            in this class
            numeric_cols (list): The list of columns to encode as numeric values.
            categorical_cols (list): The list of columns to encode as categorical.
            encoder_blueprint (EncoderBlueprint): An encoder blueprint which map its encodings to
            the passed df.
        """
        super().__init__(df, numeric_cols, categorical_cols, encoder_blueprint, True)

    def _fix_missing(self, df, col, name, na_dict):
        """ Fill missing data in a column by filling it by -9999
        (which is considered as an outlier for linear models).

        Args:
            df (DataFrame): The data frame that will be changed.
            col (pd.Series): The column of data to fix by filling in missing data.
            name (str): The name of the new filled column in df.
            na_dict (dict): A dictionary of values to create na's of and the value to insert.
        """
        col_c = col
        if is_numeric_dtype(col):
            if pd.isnull(col).sum() or (name in na_dict):
                filler = na_dict[name] if name in na_dict else -9999
                na_dict[name] = filler
                df[name] = col.fillna(filler)
        return na_dict

    def encode_categorical(self, df):
        # Encode values as onehot encoding
        # If the categorical mapping exists
        if self.blueprint.categ_var_map:
            for col_name in df.keys():
                if col_name in self.categorical_cols:
                    onehot = pd.get_dummies(df[col_name], prefix=col_name, drop_first=True)
                    res_df = pd.DataFrame(data=onehot,
                                          columns=list(self.blueprint.categ_var_map[col_name].columns))
                    res_df = res_df.fillna(0)
                    df = pd.concat([df.drop(col_name, axis=1), res_df], axis=1)
        else:
            self.blueprint.categ_var_map = {}
            for col_name in tqdm(self.categorical_cols, total=len(self.categorical_cols)):
                if col_name in df.columns:
                    # TODO: Use feature hashing for categ > 20
                    # https://en.wikipedia.org/wiki/Feature_hashing#Feature_vectorization_using_the_hashing_trick
                    # https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-8-vowpal-wabbit-fast-learning-with-gigabytes-of-data-60f750086237
                    onehot = pd.get_dummies(df[col_name], prefix=col_name)
                    self.blueprint.categ_var_map[col_name] = onehot
                    df = pd.concat([df.drop(col_name, axis=1), onehot], axis=1)

        return df
