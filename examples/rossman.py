"""
  This file is inspired by the work of the third place winner of the Rossman
  competition on Kaggle as well as this notebook by fast.ai:
  https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb

  The resulting submission of this notebook can be submitted on this page:
  https://www.kaggle.com/c/rossmann-store-sales
  The private leaderboard is the one to watch for the scoring
"""
import os
import pandas as pd
import torchlight.structured.date as date
from multiprocessing import cpu_count
import numpy as np
import isoweek
import datetime
import math
from utils.fetcher import WebFetcher
import shortcuts
import structured.encoder as encoder
from nn.learner import Learner
import torch.optim as optim
from tqdm import tqdm


def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None:
        right_on = left_on
    return left.merge(right, how='left', left_on=left_on,
                      right_on=right_on, suffixes=("", suffix))


def get_elapsed(df, monitored_field, prefix='elapsed_'):
    """
    Cumulative counting across a sorted dataframe.
    Given a particular field to monitor, this function will start tracking time since the
    last occurrence of that field. When the field is seen again, the counter is set to zero.
    Args:
        df (pd.DataFrame): A pandas DataFrame
        monitored_field (str): A string that is the name of the date column you wish to expand.
            Assumes the column is of type datetime64 if df is a dask dataframe
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
    with tqdm(total=16) as pbar:
        table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']
        train, store, store_states, state_names, googletrend, weather, test = \
            [pd.read_csv(os.path.join(files_path, f'{fname}.csv'), low_memory=False) for fname in table_names]

        # Turn state Holidays to boolean
        train.StateHoliday = train.StateHoliday != '0'
        test.StateHoliday = test.StateHoliday != '0'

        # Join tables
        weather = join_df(weather, state_names, "file", "StateName")
        pbar.update(1)

        # Replace all instances of state name 'NI' to match the usage in the rest of the data: 'HB,NI'
        googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
        googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
        googletrend.loc[googletrend.State == 'NI', "State"] = 'HB,NI'
        pbar.update(1)

        # Extracts particular date fields from a complete datetime for the purpose of constructing categoricals
        date.get_datepart(weather, "Date", drop=False, inplace=True)
        date.get_datepart(googletrend, "Date", drop=False, inplace=True)
        date.get_datepart(train, "Date", drop=False, inplace=True)
        date.get_datepart(test, "Date", drop=False, inplace=True)

        # The Google trends data has a special category for the whole of the US
        trend_de = googletrend[googletrend.file == 'Rossmann_DE']
        pbar.update(1)

        # Outer join to a single dataframe
        store = join_df(store, store_states, "Store")
        joined = join_df(train, store, "Store")
        joined_test = join_df(test, store, "Store")
        joined = join_df(joined, googletrend, ["State", "Year", "Week"])
        joined_test = join_df(joined_test, googletrend, ["State", "Year", "Week"])
        joined = joined.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
        joined_test = joined_test.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
        joined = join_df(joined, weather, ["State", "Date"])
        joined_test = join_df(joined_test, weather, ["State", "Date"])
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
            df = df.sort_values(['Store', 'Date'], ascending=[True, False])
            get_elapsed(df, field, 'Before')
            field = 'StateHoliday'
            df = df.sort_values(['Store', 'Date'])
            get_elapsed(df, field, 'After')
            df = df.sort_values(['Store', 'Date'], ascending=[True, False])
            get_elapsed(df, field, 'Before')
            field = 'Promo'
            df = df.sort_values(['Store', 'Date'])
            get_elapsed(df, field, 'After')
            df = df.sort_values(['Store', 'Date'], ascending=[True, False])
            get_elapsed(df, field, 'Before')
            # Set the active index to Date
            df = df.set_index("Date")
            # Set null values from elapsed field calculations to 0
            columns = ['SchoolHoliday', 'StateHoliday', 'Promo']
            for o in ['Before', 'After']:
                for p in columns:
                    a = o + p
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
                joined = join_df(joined, df, ['Store', 'Date'])
            elif name == "test":
                joined_test = join_df(joined_test, df, ['Store', 'Date'])
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
    contin_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
                   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h',
                   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
                   'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']
    y = 'Sales'
    yl = np.log(train_df[y])
    train_df.drop(y, axis=1, inplace=True)
    train_df = train_df.set_index("Date")
    test_df = test_df.set_index("Date")
    # Get the categorical fields cardinality before turning them all to float32
    card_cat_features = {c: len(train_df[c].astype('category').cat.categories) + 1 for c in cat_vars}
    train_df = encoder.apply_encoding(train_df, contin_vars, cat_vars, do_scale=True)
    test_df = encoder.apply_encoding(test_df, contin_vars, cat_vars, do_scale=True)
    return train_df, test_df, yl, cat_vars, card_cat_features


def exp_rmspe(y_pred, targ):
    """
    Root-mean-squared percent error is the metric Kaggle used for this competition
    Args:
        y_pred (list): predicted labels
        targ (list): true labels

    Returns:
        The Root-mean-squared percent error
    """
    targ = np.exp(targ)
    pct_var = (targ - np.exp(y_pred)) / targ
    return math.sqrt((pct_var ** 2).mean())


def main():
    batch_size = 128
    epochs = 20
    output_path = "/tmp/rossman"

    preprocessed_train_path = os.path.join(output_path, 'joined.feather')
    preprocessed_test_path = os.path.join(output_path, 'joined_test.feather')
    WebFetcher.download_dataset("http://files.fast.ai/part2/lesson14/rossmann.tgz", output_path, True)
    print("Preprocessing...")
    if os.path.exists(preprocessed_train_path) and os.path.exists(preprocessed_test_path):
        train_df = pd.read_feather(preprocessed_train_path, nthreads=cpu_count())
        test_df = pd.read_feather(preprocessed_test_path, nthreads=cpu_count())
    else:
        train_df, test_df = prepare_data(output_path,
                                         preprocessed_train_path,
                                         preprocessed_test_path)

    train_df, test_df, yl, cat_vars, card_cat_features = create_features(train_df, test_df)
    val_idx = np.flatnonzero(
        (train_df.index <= datetime.datetime(2014, 9, 17)) & (train_df.index >= datetime.datetime(2014, 8, 1)))
    print("Preprocessing finished...")

    max_log_y = np.max(yl)
    y_range = (0, max_log_y * 1.2)
    shortcut = shortcuts.ColumnarShortcut.from_data_frame(train_df, val_idx, yl.astype(np.float32),
                                                          cat_vars, batch_size=batch_size, test_df=test_df)
    model = shortcut.get_model(card_cat_features, len(train_df.columns) - len(cat_vars),
                               0.04, 1, [1000, 500], [0.001, 0.01], y_range=y_range)
    learner = Learner(model)
    learner.train(optim.Adam, exp_rmspe, None, epochs, shortcut.get_train_loader, shortcut.get_val_loader)
    d = 0


if __name__ == "__main__":
    main()
