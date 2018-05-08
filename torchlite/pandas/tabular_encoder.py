"""
A structured data encoder based on sklearn API
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_numeric_dtype


class BaseEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_vars, categorical_vars, fix_missing, numeric_scaler):
        self.categorical_vars = categorical_vars
        self.numeric_vars = numeric_vars
        self.tfs_list = {}
        self.numeric_scaler = numeric_scaler
        self.fix_missing = fix_missing

    def _get_all_non_numeric(self, df):
        non_num_cols = []
        for col in df.columns:
            if not is_numeric_dtype(df[col]):
                non_num_cols.append(col)
        return non_num_cols

    def _check_integrity(self, df):
        """
        Check if the columns registered in self.tfs_list are the same as the ones
        in the passed df
        Returns:
            bool: Return True if the columns match, raise an exception otherwise
        """

        diff = list(set(self.tfs_list["cols"]) ^ set(df.columns))
        if len(diff) > 0:
            raise Exception("Columns in fitted and transformed DataFrame do not match: {}".format(diff))

        if list(self.tfs_list["cols"]) == list(df.columns):
            return True

        raise Exception("Columns in EncoderBlueprint and DataFrame do not have the same order")

    def _perform_na_fit(self, df, y):
        raise NotImplementedError()

    def _perform_na_transform(self, df):
        raise NotImplementedError()

    def _perform_categ_fit(self, df, y):
        raise NotImplementedError()

    def _perform_categ_transform(self, df):
        raise NotImplementedError()

    def fit(self, X, y=None, **kwargs):
        """
        Fit encoder according to X and y.
        The features from the `X` DataFrame which are not passed in `numeric_cols` or `categorical_cols`
        are just ignored during transformation.
        This method will fir the `X` dataset to achieve the following:
            - Remove NaN values by using the feature mean and adding a feature_missing feature
            - Scale the numeric values according to the EncoderBlueprint scaler
            - Encode categorical features to numeric types
        What it doesn't do:
            - Deal with outliers
        Args:
            X (pd.DataFrame): Array of DataFrame of shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples
                and n_features is the number of features.
            y (str): Column name of the target value of X. The column must be contained in X.
        Returns:
            self : encoder
                Returns self.
        """
        all_feat = self.categorical_vars + self.numeric_vars
        df = X[[feat for feat in all_feat if feat in X.columns]].copy()
        self.tfs_list["y"] = y
        if self.tfs_list["y"] is not None:
            df[self.tfs_list["y"]] = X[self.tfs_list["y"]].copy()

        # Missing values
        # TODO for ordered data (e.g. time series), take the adjacent value — next or previous
        self._perform_na_fit(df, y)
        self._perform_na_transform(df)

        # Categorical columns
        # TODO add an "n/a" category for every categorical feature
        self._perform_categ_fit(df, y)
        self._perform_categ_transform(df)

        # Scaling
        num_cols = [n for n in df.columns if is_numeric_dtype(df[n]) and n in self.numeric_vars]
        self.tfs_list["num_cols"] = num_cols
        if self.numeric_scaler is not None:
            # Turning all the columns to the same dtype before scaling is important
            self.numeric_scaler.fit(df[num_cols].astype(np.float32).values)

        self.tfs_list["cols"] = df.columns
        del df
        return self

    def transform(self, X):
        """
        Perform the transformation to new data.
        X (pd.DataFrame): Array of DataFrame of shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples
                and n_features is the number of features.

        Returns:

        """
        all_feat = self.categorical_vars + self.numeric_vars
        missing_col = [col for col in X.columns if col not in all_feat]
        df = X[[feat for feat in all_feat if feat in X.columns]].copy()

        print("Warning: Missing columns: {}, dropping them...".format(missing_col))
        print("--- Fixing NA values ({}) ---".format(len(self.tfs_list["missing"])))
        self._perform_na_transform(df)
        print("List of NA columns fixed: {}".format(list(self.tfs_list["missing"])))
        print("Categorizing features {}".format(self.categorical_vars))

        # Categorical columns
        self._perform_categ_transform(df)

        # Scaling
        if self.numeric_scaler is not None:
            num_cols = self.tfs_list["num_cols"]
            # Turning all the columns to the same dtype before scaling is important
            df[num_cols] = self.numeric_scaler.transform(df[num_cols].astype(np.float32).values)
            print("List of scaled columns: {}".format(num_cols))

        # Print stats
        non_num_cols = self._get_all_non_numeric(df)
        nan_ratio = df.isnull().sum() / len(df)

        print("------- Dataframe of len {} summary -------\n".format(len(df.columns)))
        for col, nan, dtype in zip(df.columns, nan_ratio.values, df.dtypes.values):
            print("Column {:<30}:\t dtype: {:<10}\t NaN ratio: {}".format(col, str(dtype), nan))
        if len(non_num_cols) > 0 and nan_ratio > 0:
            raise Exception("Not all columns are numeric or NaN has been found: {}.".format(non_num_cols))

        self._check_integrity(df)
        print("---------- Preprocessing done -----------")
        return df


class TreeEncoder(BaseEncoder):
    def __init__(self, numeric_vars, categorical_vars, fix_missing=True, numeric_scaler=None):
        """
            An encoder to encode data from structured (tabular) data
            used for tree based models (RandomForests, GBTs) as well
            as deep neural networks with categorical embeddings features (DNN)
        Args:
            numeric_vars (list): The list of variables to encode as numeric values.
            categorical_vars (list): The list of variables to encode as categorical.
            fix_missing (bool): True to fix the missing values (will add a new feature `is_missing` and replace
                the missing value by its median). For some models like Xgboost you may want to set this value to False.
            numeric_scaler (None, Scaler): None or a scaler from sklearn.preprocessing.data for scaling numeric features
                All features types will be encoded as float32.
                An sklearn StandardScaler() will fit most common cases.
                For a more robust scaling with outliers take a look at RankGauss:
                    https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
                and rankdata:
                    https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.rankdata.html

            Reference -> http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
        """
        super().__init__(numeric_vars, categorical_vars, fix_missing, numeric_scaler)

    def _perform_na_fit(self, df, y):
        missing = {}
        all_feat = self.categorical_vars + self.numeric_vars
        for feat in all_feat:
            if is_numeric_dtype(df[feat]):
                if pd.isnull(df[feat]).sum():
                    median = df[feat].median()
                    missing[feat] = median
        self.tfs_list["missing"] = missing

    def _perform_na_transform(self, df):
        for col, median in self.tfs_list["missing"].items():
            df[col + '_na'] = df[col].isnull()
            df[col].fillna(median, inplace=True)

    def _perform_categ_fit(self, df, y):
        categ_cols = {}
        for col in self.categorical_vars:
            if col in df.columns:
                # TODO Use pd.factorize() instead
                categs = df[col].astype(pd.api.types.CategoricalDtype()).cat.categories
                categ_cols[col] = categs
        self.tfs_list["categ_cols"] = categ_cols

    def _perform_categ_transform(self, df):
        for col, vals in self.tfs_list["categ_cols"].items():
            df[col] = df[col].astype(pd.api.types.CategoricalDtype(categories=vals)).cat.codes + 1


class LinearEncoder(BaseEncoder):
    def __init__(self, numeric_vars, categorical_vars, fix_missing, numeric_scaler=None):
        """
            An encoder used for linear based models (Linear/Logistic regression) as well
            as deep neural networks without embeddings.
        Args:
            numeric_vars (list): The list of variables to encode as numeric values.
            categorical_vars (list): The list of variables to encode as categorical.
            fix_missing (bool): True to fix the missing values (will add a new feature `is_missing` and replace
                the missing value by its median). For some models like Xgboost you may want to set this value to False.
            numeric_scaler (None, Scaler): None or a scaler from sklearn.preprocessing.data for scaling numeric features
                All features types will be encoded as float32.
                An sklearn StandardScaler() will fit most common cases.
                For a more robust scaling with outliers take a look at RankGauss:
                    https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
                and rankdata:
                    https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.rankdata.html

            Reference -> http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
        """
        super().__init__(numeric_vars, categorical_vars, fix_missing, numeric_scaler)

    def _perform_na_fit(self, df, y):
        missing = []
        all_feat = self.categorical_vars + self.numeric_vars
        for feat in all_feat:
            if is_numeric_dtype(df[feat]):
                if pd.isnull(df[feat]).sum():
                    missing.append(feat)
        self.tfs_list["missing"] = missing

    def _perform_na_transform(self, df):
        for col in self.tfs_list["missing"]:
            df[col].fillna(-999999, inplace=True)

    def _perform_categ_fit(self, df, y):
        categ_cols = {}
        for col in self.categorical_vars:
            # https://github.com/scikit-learn-contrib/categorical-encoding
            # TODO:
            #  - Ordinal
            #  - One - Hot
            #  - Binary
            #  - Helmert
            #  - Contrast
            #  - Sum
            #  - Contrast
            #  - Polynomial
            #  - Contrast
            #  - Backward
            #  - Difference
            #  - Contrast
            #  - Hashing
            #  - BaseN
            #  - LeaveOneOut
            #  - Target Encoding
            # https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/
            # https://en.wikipedia.org/wiki/Feature_hashing#Feature_vectorization_using_the_hashing_trick
            # https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-8-vowpal-wabbit-fast-learning-with-gigabytes-of-data-60f750086237
            if df[col].nunique() < 10:
                # If cardinality < 10, use OneHot (or create a sparse matrix with scipy.sparse.csc_matrix ?)
                enc = OneHotEncoder()
                enc.fit(df[col].values)
                categ_cols[col] = {"Onehot": enc}
            else:
                if self.tfs_list["y"] is None:
                    raise Exception("You have to pass your target variable to the fit() function for target encoding")
                # Otherwise use Mean/target/likelihood encoding
                cumsum = df.groupby(col)[self.tfs_list["y"]].cumsum() - df[self.tfs_list["y"]]
                cumcnt = df.groupby(col)[self.tfs_list["y"]].cumcount()
                means = cumsum / cumcnt
                categ_cols[col] = {"target_mean": means}
        self.tfs_list["categ_cols"] = categ_cols

    def _perform_categ_transform(self, df):
        for col, item in self.tfs_list["categ_cols"].items():
            method, enc = item.keys(), item.values()
            if method == "Onehot":
                onehot = enc.transform(df[col].values)
                df = pd.concat([df.drop(col, axis=1), onehot[:, 1:]], axis=1)
            elif method == "target_mean":
                # TODO BE CAREFUL TRAIN/VAL SPLIT SHOULD ALREADY BE DONE HERE!!
                # TODO on the test set the means of the train is used
                # Mean/target/likelihood encoding
                df[col + "_mean_target"] = df[col].map(enc)
