"""
  This file is inspired by the work of the third place winner of the Rossman
  competition on Kaggle as well as this notebook by fast.ai:
  https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb

  The resulting csv of this notebook can be submitted on this page:
  https://www.kaggle.com/c/rossmann-store-sales
  The private leaderboard is the one to watch for the scoring
"""
import torchlite
torchlite.set_backend("torch")

import os
import pandas as pd
from multiprocessing import cpu_count
import numpy as np
import torch.nn.functional as F
import isoweek
import torch.optim as optim
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from torchlite.learner import Learner
from torchlite.learner.cores import TorchClassifierCore
import torchlite.torch.metrics as metrics
from torchlite.data.fetcher import WebFetcher
import torchlite.torch.shortcuts as shortcuts
import torchlite.pandas.date as edate
from torchlite.train_callbacks import CosineAnnealingCallback
from torchlite.pandas.tabular_encoder import TreeEncoder
import torchlite.pandas.merger as tmerger


def to_csv(test_file, output_file, identifier_field, predicted_field,
           predictions, read_format='csv'):
    df = None
    if read_format == 'csv':
        df = pd.read_csv(test_file)
    elif read_format == 'feather':
        df = pd.read_feather(test_file)
    df = df[[identifier_field]]
    df[predicted_field] = predictions
    df.to_csv(output_file, index=False)


def get_elapsed(df, monitored_field, prefix='elapsed_'):
    """
    Cumulative counting across a sorted dataframe.
    Given a particular field to monitor, this function will start tracking time since the
    last occurrence of that field. When the field is seen again, the counter is set to zero.
    Args:
        df (pd.DataFrame): A pandas DataFrame
        monitored_field (str): A string that is the name of the date column you wish to expand.
            Assumes the column is of type datetime64
        prefix (str): The prefix to add to the newly created field.
    """
    day1 = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    last_store = 0
    res = []

    for s, v, d in zip(df["Store"].values, df[monitored_field].values, df["Date"].values):
        if s != last_store:
            last_date = np.datetime64()
            last_store = s
        if v:
            last_date = d
        res.append(((d - last_date).astype('timedelta64[D]') / day1).astype(int))
    df[prefix + monitored_field] = res


def prepare_data(files_path, preprocessed_train_path, preprocessed_test_path):
    print("Preparing data...")
    with tqdm(total=16) as pbar:
        table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']
        train, store, store_states, state_names, googletrend, weather, test = \
            [pd.read_csv(os.path.join(files_path, f'{fname}.csv'), low_memory=False) for fname in table_names]

        # Turn state Holidays to boolean
        train.StateHoliday = train.StateHoliday != '0'
        test.StateHoliday = test.StateHoliday != '0'

        # Join tables
        weather = tmerger.join_df(weather, state_names, "file", "StateName")
        pbar.update(1)

        # Replace all instances of state name 'NI' to match the usage in the rest of the data: 'HB,NI'
        googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
        googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
        googletrend.loc[googletrend.State == 'NI', "State"] = 'HB,NI'
        pbar.update(1)

        # Extracts particular date fields from a complete datetime for the purpose of constructing categoricals
        edate.get_datepart(weather, "Date", drop=False, inplace=True)
        edate.get_datepart(googletrend, "Date", drop=False, inplace=True)
        edate.get_datepart(train, "Date", drop=False, inplace=True)
        edate.get_datepart(test, "Date", drop=False, inplace=True)

        edate.get_elapsed(weather, "Date", inplace=True)
        edate.get_elapsed(googletrend, "Date", inplace=True)
        edate.get_elapsed(train, "Date", inplace=True)
        edate.get_elapsed(test, "Date", inplace=True)

        # The Google trends data has a special category for the whole of the US
        trend_de = googletrend[googletrend.file == 'Rossmann_DE']
        pbar.update(1)

        # Outer join to a single dataframe
        store = tmerger.join_df(store, store_states, "Store")
        joined = tmerger.join_df(train, store, "Store")
        joined_test = tmerger.join_df(test, store, "Store")
        joined = tmerger.join_df(joined, googletrend, ["State", "Year", "Week"])
        joined_test = tmerger.join_df(joined_test, googletrend, ["State", "Year", "Week"])
        joined = joined.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
        joined_test = joined_test.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
        joined = tmerger.join_df(joined, weather, ["State", "Date"])
        joined_test = tmerger.join_df(joined_test, weather, ["State", "Date"])
        for df in (joined, joined_test):
            for c in df.columns:
                if c.endswith('_y'):
                    if c in df.columns:
                        df.drop(c, inplace=True, axis=1)
        pbar.update(1)

        for df in (joined, joined_test):
            # Fill in missing values to avoid complications
            df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)
            df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)
            df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)
            df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)
            pbar.update(1)

            # Extract features "CompetitionOpenSince" and "CompetitionDaysOpen"
            df["CompetitionOpenSince"] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear,
                                                             month=df.CompetitionOpenSinceMonth, day=15))
            df["CompetitionDaysOpen"] = df.Date.subtract(df.CompetitionOpenSince).dt.days
            pbar.update(1)

            # Replace some erroneous / outlying data
            df.loc[df.CompetitionDaysOpen < 0, "CompetitionDaysOpen"] = 0
            df.loc[df.CompetitionOpenSinceYear < 1990, "CompetitionDaysOpen"] = 0
            pbar.update(1)

            # Add "CompetitionMonthsOpen" field, limiting the maximum to 2 years to limit number of unique categories.
            df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"] // 30
            df.loc[df.CompetitionMonthsOpen > 24, "CompetitionMonthsOpen"] = 24
            df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: isoweek.Week(
                x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1).astype(pd.datetime))
            df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days
            df.loc[df.Promo2Days < 0, "Promo2Days"] = 0
            df.loc[df.Promo2SinceYear < 1990, "Promo2Days"] = 0
            df["Promo2Weeks"] = df["Promo2Days"] // 7
            df.loc[df.Promo2Weeks < 0, "Promo2Weeks"] = 0
            df.loc[df.Promo2Weeks > 25, "Promo2Weeks"] = 25
            pbar.update(1)

        # Durations
        columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]
        for name, df in zip(("train", "test"), (train[columns], test[columns])):
            field = 'SchoolHoliday'
            df = df.sort_values(['Store', 'Date'])
            get_elapsed(df, field, 'After')
            field = 'StateHoliday'
            df = df.sort_values(['Store', 'Date'])
            get_elapsed(df, field, 'After')
            field = 'Promo'
            df = df.sort_values(['Store', 'Date'])
            get_elapsed(df, field, 'After')
            # Set the active index to Date
            df = df.set_index("Date")
            # Set null values from elapsed field calculations to 0
            columns = ['SchoolHoliday', 'StateHoliday', 'Promo']
            for p in columns:
                a = 'After' + p
                df[a] = df[a].fillna(0)
            # Window functions in pandas to calculate rolling quantities
            bwd = df[['Store'] + columns].sort_index().groupby("Store").rolling(7, min_periods=1).sum()
            fwd = df[['Store'] + columns].sort_index(ascending=False).groupby("Store").rolling(7, min_periods=1).sum()
            # We want to drop the Store indices grouped together in the window function
            bwd.drop('Store', 1, inplace=True)
            bwd.reset_index(inplace=True)
            fwd.drop('Store', 1, inplace=True)
            fwd.reset_index(inplace=True)
            df.reset_index(inplace=True)
            df = df.merge(bwd, 'left', ['Date', 'Store'], suffixes=['', '_bw'])
            df = df.merge(fwd, 'left', ['Date', 'Store'], suffixes=['', '_fw'])
            df.drop(columns, 1, inplace=True)
            df["Date"] = pd.to_datetime(df.Date)

            if name == "train":
                joined = tmerger.join_df(joined, df, ['Store', 'Date'])
            elif name == "test":
                joined_test = tmerger.join_df(joined_test, df, ['Store', 'Date'])
            pbar.update(1)

        # The authors also removed all instances where the store had zero sale / was closed
        # We speculate that this may have cost them a higher standing in the competition
        joined = joined[joined.Sales != 0]
        joined.reset_index(inplace=True)
        joined_test.reset_index(inplace=True)
        pbar.update(1)

        # Save to feather
        joined.to_feather(preprocessed_train_path)
        joined_test.to_feather(preprocessed_test_path)
        pbar.update(1)
        print("Data saved to feather.")
        return joined, joined_test


def create_features(train_df, test_df):
    cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
                'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear',
                'Promo2SinceYear', 'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw',
                'StateHoliday_bw', 'SchoolHoliday_fw', 'SchoolHoliday_bw']
    num_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
                'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h',
                'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
                'AfterStateHoliday', 'Promo', 'AfterPromo', 'SchoolHoliday']

    y = 'Sales'
    y_log = np.log(train_df[y]).astype(np.float32)
    train_df.drop(y, axis=1, inplace=True)

    train_df = train_df.set_index("Date")
    test_df = test_df.set_index("Date")
    # Get the categorical fields cardinality
    card_cat_features = {c: len(train_df[c].astype('category').cat.categories) + 1 for c in cat_vars}

    enc = TreeEncoder(num_vars, cat_vars, fix_missing=True, numeric_scaler=StandardScaler())
    train_df = enc.fit_transform(train_df)
    test_df = enc.transform(test_df)

    assert len(train_df.columns) == len(test_df.columns)
    train_df["Sales_log"] = y_log.values
    return train_df, test_df, cat_vars, card_cat_features


def main():
    output_path = "/tmp/rossmann"

    preprocessed_train_path = os.path.join(output_path, 'joined.feather')
    preprocessed_test_path = os.path.join(output_path, 'joined_test.feather')
    WebFetcher.download_dataset("https://f002.backblazeb2.com/file/torchlite-data/rossmann.tgz", output_path, True)
    if os.path.exists(preprocessed_train_path) and os.path.exists(preprocessed_test_path):
        train_df = pd.read_feather(preprocessed_train_path, nthreads=cpu_count())
        test_df = pd.read_feather(preprocessed_test_path, nthreads=cpu_count())
    else:
        train_df, test_df = prepare_data(output_path,
                                         preprocessed_train_path,
                                         preprocessed_test_path)

    train_df, test_df, cat_vars, card_cat_features = create_features(train_df, test_df)

    # -- Model parameters
    batch_size = 256
    epochs = 20

    max_log_y = np.max(train_df['Sales_log'].values)
    y_range = (0, max_log_y * 1.2)

    # /!\ Uncomment this to get a real validation set
    #train_df, val_df = tsplitter.time_split(train_df, datetime.datetime(2014, 8, 1), datetime.datetime(2014, 9, 17))
    val_df = None
    # --

    shortcut = shortcuts.ColumnarShortcut.from_data_frames(train_df, val_df, "Sales_log", cat_vars, batch_size, test_df)
    model = shortcut.get_stationary_model(card_cat_features, len(train_df.columns) - len(cat_vars),
                                          output_size=1, emb_drop=0.04, hidden_sizes=[1000, 500],
                                          hidden_dropouts=[0.001, 0.01], y_range=y_range)
    optimizer = optim.Adam(model.parameters())
    learner = Learner(TorchClassifierCore(model, optimizer, F.mse_loss))
    learner.train(epochs, [metrics.RMSPE(to_exp=True)], shortcut.get_train_loader, shortcut.get_val_loader,
                  callbacks=[CosineAnnealingCallback(optimizer, T_max=epochs)])
    test_pred = learner.predict(shortcut.get_test_loader)
    test_pred = np.exp(test_pred)

    # Save the predictions as a csv file
    sub_file_path = os.path.join(output_path, "submit.csv")
    to_csv(preprocessed_test_path, sub_file_path, 'Id', 'Sales', test_pred, read_format='feather')
    print("Predictions saved to {}".format(sub_file_path))


if __name__ == "__main__":
    main()
