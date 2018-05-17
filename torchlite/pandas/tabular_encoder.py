"""
A structured data encoder based on sklearn API
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders.one_hot import OneHotEncoder as CategOneHot
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
        if y is not None:
            self.tfs_list["y"] = X[y]

        # Missing values
        if self.fix_missing:
            self._perform_na_fit(df, y)
            df = self._perform_na_transform(df)

        # Categorical columns
        # http://contrib.scikit-learn.org/categorical-encoding/
        self._perform_categ_fit(df, y)
        df = self._perform_categ_transform(df)

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

        if self.fix_missing:
            print("Warning: Missing columns: {}, dropping them...".format(missing_col))
            print("--- Fixing NA values ({}) ---".format(len(self.tfs_list["missing"])))
            df = self._perform_na_transform(df)
            print("List of NA columns fixed: {}".format(list(self.tfs_list["missing"])))
            print("Categorizing features {}".format(self.categorical_vars))

        # Categorical columns
        df = self._perform_categ_transform(df)

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
        if len(non_num_cols) > 0:
            raise Exception("Not all columns are numeric: {}.".format(non_num_cols))
        if self.fix_missing and nan_ratio.all() > 0:
            raise Exception("NaN has been found!")

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
        return df

    def _perform_categ_fit(self, df, y):
        categ_cols = {}
        for col in self.categorical_vars:
            if col in df.columns:
                categs = pd.factorize(df[col], na_sentinel=0, order=True)[1]
                categ_cols[col] = categs
        self.tfs_list["categ_cols"] = categ_cols

    def _perform_categ_transform(self, df):
        for col, vals in self.tfs_list["categ_cols"].items():
            # "n/a" category will be encoded as 0
            df[col] = df[col].astype(pd.api.types.CategoricalDtype(categories=vals, ordered=True)).cat.codes + 1
        return df


class LinearEncoder(BaseEncoder):
    def __init__(self, numeric_vars, categorical_vars, fix_missing, numeric_scaler=None,
                 categ_enc_method="target"):
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
            categ_enc_method (str, None): One of the following methods can be used:
                - "hashing": Better known as the "hashing trick"
                - "target": Also known as Mean encoding/Target encoding/Likelihood encoding.
                    The implementation is based on the Expanding mean scheme
                - "onehot": Onehot encoding.
                    Consider using one_hot_encode_sparse() instead to get a sparse matrix with lower
                    memory footprint if your categorical variable have high cardinality.
                - None: No encoding on categorical variables will be used
        """
        super().__init__(numeric_vars, categorical_vars, fix_missing, numeric_scaler)
        self.categ_enc_method = categ_enc_method.lower() if categ_enc_method is not None else categ_enc_method
        self.hash_space = 25

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
        return df

    def _perform_categ_fit(self, df, y):
        # https://github.com/scikit-learn-contrib/categorical-encoding
        # https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/
        # https://en.wikipedia.org/wiki/Feature_hashing#Feature_vectorization_using_the_hashing_trick
        categ_cols = {}
        onehot_cols = []
        for col in self.categorical_vars:
            categs = df[col].astype(pd.api.types.CategoricalDtype()).cat.categories
            if self.categ_enc_method == "onehot":
                card = df[col].nunique()
                if card > 10:
                    print("Warning, cardinality of {} = {}".format(col, card))
                onehot_cols.append(col)
            elif self.categ_enc_method == "target":
                if self.tfs_list["y"] is None:
                    raise Exception("You have to pass your target variable to the fit() "
                                    "function for target encoding")
                # Mean/target/likelihood encoding
                target_col_name = self.tfs_list["y"].name
                df_enc = df.copy()
                df_enc[target_col_name] = self.tfs_list["y"]
                cumsum = df_enc.groupby(col)[target_col_name].cumsum() - df_enc[target_col_name]
                cumcnt = df_enc.groupby(col)[target_col_name].cumcount()
                means = cumsum / cumcnt
                means.rename('mean_enc', inplace=True)

                mean_enc = pd.Series(means, index=self.tfs_list["y"]).to_dict()
                global_mean = self.tfs_list["y"].mean()
                categ_cols[col] = {"target": (global_mean, mean_enc)}
            elif self.categ_enc_method == "hashing":
                str_hashs = [col + "=" + str(val) for val in categs]
                hashs = [hash(h) % self.hash_space for h in str_hashs]
                categ_cols[col] = {"hashing": hashs}
        if len(onehot_cols) > 0:
            enc = CategOneHot(cols=onehot_cols, handle_unknown='impute')
            enc.fit(df)
            self.tfs_list["onehot"] = enc
        self.tfs_list["categ_cols"] = categ_cols

    def _perform_categ_transform(self, df):
        if self.tfs_list.get("onehot") is not None:
            enc = self.tfs_list["onehot"]
            # TODO check to avoid collinearity
            df = enc.transform(df)

        if self.categ_enc_method is None:
            print("Warning, no encoding set for features {}".format(self.tfs_list["categ_cols"].keys()))
            return df
        for col, item in self.tfs_list["categ_cols"].items():
            method = next(iter(item.keys()))
            if method == "target":
                # BE CAREFUL of the following points:
                # • Local experiments:
                #   ‒ Estimate encodings on X_train
                #   ‒ Map them to X_train and X_val
                #   ‒ Regularize on X_train
                #   ‒ Validate the model on X_train / X_val split
                # • Submission:
                #   ‒ Estimate encodings on whole Train data
                #   ‒ Map them to Train and Test
                #   ‒ Regularize on Train
                #   ‒ Fit on Train
                global_mean, mean_enc = list(item.values())[0]
                df[col + "_mean_target"] = df[col].map(mean_enc)
                df[col + "_mean_target"] = df[col + "_mean_target"].fillna(global_mean)
                df.drop(col, axis=1, inplace=True)
            elif method == "hashing":
                categs = df[col].astype(pd.api.types.CategoricalDtype()).cat.codes
                str_hashs = [col + "=" + str(val) for val in categs]
                hashs = [hash(h) % self.hash_space for h in str_hashs]
                df[col] = hashs
        return df


class SparseOneHotEncoder:
    def __init__(self, numeric_vars, categorical_vars):
        """
            One-hot encode the given categorical_vars and return a sparse matrix.
            Features neither in numeric_vars nor in categorical_vars won't be considered
            in the resulting sparse matrix.
        Args:
            numeric_vars (list): The list of variables to encode as numeric values.
            categorical_vars (list): List of categorical vars to transform to onehot
        """
        self.numeric_vars = numeric_vars
        self.categorical_vars = categorical_vars

    def fit_transform(self, df_list):
        """
        Perform the transformation. You should pass your train/valid/test sets to df_list if
        you have them.
        Args:
            df_list (list): A list of pandas DataFrame. The OneHot categories will be created
            from all the DataFrames of this list so they must have the same columns.
            Usually you will pass the train/valid/test sets to this list.

        Returns:
            list: List of sparse matrix in the same order as the passed df
        """
        # Concatenate
        df = pd.concat(df_list)
        df = df[self.numeric_vars + self.categorical_vars]
        lenc = LabelEncoder()
        inds = [0] + [df.shape[0] for df in df_list]
        categs_inds = []

        for categ_var in self.categorical_vars:
            categs_inds.append(df.columns.get_loc(categ_var))
            df[categ_var] = lenc.fit_transform(df[categ_var].values) + 1
            if len(lenc.classes_) > 10:
                print("Warning, cardinality of {} = {}".format(categ_var, len(lenc.classes_)))

        # Create an internal sparse matrix
        enc = OneHotEncoder(categorical_features=categs_inds)
        enc.fit(df.values)
        # Separate back the DataFrames with the right columns
        all_df = [df.iloc[inds[i]:ind + inds[i]] for i, ind in enumerate(inds[1:])]

        # Turn them into sparse matrix
        res = []
        for sep_df in all_df:
            # TODO check to avoid collinearity
            res.append(enc.transform(sep_df.values))
        return res
