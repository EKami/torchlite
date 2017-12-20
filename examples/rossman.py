"""
  This file is inspired by the work of the third place winner of the Rossman
  competition on Kaggle as well as this notebook by fast.ai:
  https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb
"""
import os
import pandas as pd
import torchlight.structured.date as date
import numpy as np
import isoweek
from utils.fetcher import WebFetcher


def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None:
        right_on = left_on
    return left.merge(right, how='left', left_on=left_on,
                      right_on=right_on, suffixes=("", suffix))


def prepare_data(files_path, joined_path, joined_test_path):
    table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']
    train, store, store_states, state_names, googletrend, weather, test = \
        [pd.read_csv(os.path.join(files_path, f'{fname}.csv'), low_memory=False) for fname in table_names]

    # Turn state Holidays to boolean
    train.StateHoliday = train.StateHoliday != '0'
    test.StateHoliday = test.StateHoliday != '0'

    # Join tables
    weather = join_df(weather, state_names, "file", "StateName")

    # Replace all instances of state name 'NI' to match the usage in the rest of the data: 'HB,NI'
    googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
    googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
    googletrend.loc[googletrend.State == 'NI', "State"] = 'HB,NI'

    # Extracts particular date fields from a complete datetime for the purpose of constructing categoricals
    date.add_datepart(weather, "Date", drop=False, inplace=True)
    date.add_datepart(googletrend, "Date", drop=False, inplace=True)
    date.add_datepart(train, "Date", drop=False, inplace=True)
    date.add_datepart(test, "Date", drop=False, inplace=True)

    # The Google trends data has a special category for the whole of the US
    trend_de = googletrend[googletrend.file == 'Rossmann_DE']

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

    # Fill in missing values to avoid complications
    for df in (joined, joined_test):
        df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)
        df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)
        df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)
        df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)

    # Extract features "CompetitionOpenSince" and "CompetitionDaysOpen"
    for df in (joined, joined_test):
        df["CompetitionOpenSince"] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear,
                                                         month=df.CompetitionOpenSinceMonth, day=15))
        df["CompetitionDaysOpen"] = df.Date.subtract(df.CompetitionOpenSince).dt.days

    # Replace some erroneous / outlying data
    for df in (joined, joined_test):
        df.loc[df.CompetitionDaysOpen < 0, "CompetitionDaysOpen"] = 0
        df.loc[df.CompetitionOpenSinceYear < 1990, "CompetitionDaysOpen"] = 0

    # Add "CompetitionMonthsOpen" field, limiting the maximum to 2 years to limit number of unique categories.
    for df in (joined, joined_test):
        df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"] // 30
        df.loc[df.CompetitionMonthsOpen > 24, "CompetitionMonthsOpen"] = 24

    for df in (joined, joined_test):
        df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: isoweek.Week(
            x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1).astype(pd.datetime))
        df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days

    for df in (joined, joined_test):
        df.loc[df.Promo2Days < 0, "Promo2Days"] = 0
        df.loc[df.Promo2SinceYear < 1990, "Promo2Days"] = 0
        df["Promo2Weeks"] = df["Promo2Days"] // 7
        df.loc[df.Promo2Weeks < 0, "Promo2Weeks"] = 0
        df.loc[df.Promo2Weeks > 25, "Promo2Weeks"] = 25

    ## Durations

    joined.to_feather(joined_path)
    joined_test.to_feather(joined_test_path)
    print("Data saved to feather.")
    return joined, joined_test


def main():
    output_path = "/tmp/rossman"
    joined_path = os.path.join(output_path, 'joined.feather')
    joined_test_path = os.path.join(output_path, 'joined_test.feather')
    joined = None
    joined_test = None
    WebFetcher.download_dataset("http://files.fast.ai/part2/lesson14/rossmann.tgz", output_path, True)
    if not os.path.exists(joined_path) and os.path.exists(joined_test_path):
        joined, joined_test = prepare_data(output_path, joined_path, joined_test_path)
    else:
        joined = pd.read_feather(joined_path)
        joined_test = pd.read_feather(joined_test_path)

    d = 0


if __name__ == "__main__":
    main()
