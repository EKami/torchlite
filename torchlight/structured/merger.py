import os
import shutil
from dask.diagnostics import ProgressBar


def merge_datasets(on_df: list, from_df: list, merge_fnc: list, output_files: list,
                   replace_if_exist=False):
    """
    Merges pandas/dask dataframes from from_df to on_df. This allow
    for efficient merging of tables

    Args:
        on_df (list): List of main dataframes which will get metadata merged from from_df
        from_df (list):
        merge_fnc (list): List of functions for merging with signature
            (on_df, from_df) -> on_df. Must be the same size as from_df.
        output_files (list): List of output files. Will be saved into the Apache Parquet format.
            Must be of the same size as on_df
        replace_if_exist (bool): Replace if the files already exists
    Returns:

    """
    assert len(from_df) == len(merge_fnc), "from_df and merge_fnc length differs"
    assert len(output_files) == len(on_df), "on_df and output_files length differs"

    for df_opath in output_files:
        if os.path.exists(df_opath):
            if replace_if_exist:
                shutil.rmtree(df_opath)
            else:
                print("Datasets already merged")
                return

    for idf_main in range(len(on_df)):
        for df_second, df_second_fnc in zip(from_df, merge_fnc):
            df_main = on_df[idf_main]
            on_df[idf_main] = df_second_fnc(df_main, df_second)

    for main_df, df_ofile in zip(on_df, output_files):
        print(f"Processing {df_ofile}")
        with ProgressBar():
            main_df.to_parquet(df_ofile, engine='fastparquet')
