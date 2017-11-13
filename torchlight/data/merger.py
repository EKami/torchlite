import pandas as pd
from multiprocessing import cpu_count
from multiprocessing import Pool
import uuid
import time

g_from_df = None
g_merge_fnc_param = None
g_on_df_tranformer = None
g_output_file = None


def _evaluate_chunk(chunk):
    global g_from_df, g_merge_fnc_param, g_on_df_tranformer, g_output_file
    identifier = uuid.uuid4()
    output_file = g_output_file + '_' + str(identifier)
    chunk = g_on_df_tranformer(chunk)
    chunk = chunk.merge(g_from_df, **g_merge_fnc_param)
    chunk.to_csv(output_file, header=False)
    chunk_types = [(name, dtype) for name, dtype in zip(chunk.columns.tolist(), chunk.dtypes.tolist())]
    return output_file


def merge_df(from_df: pd.DataFrame, on_df: pd.io.parsers.TextFileReader,
             output_file: str, merge_fnc_param: dict,
             on_df_tranformer: callable = None,
             nthreads=cpu_count(), verbose=True) -> list:
    """
    This function is useful when you want to merge a little dataframe `from_df` into a big one `on_df`.
    It uses `on_df` (a DataFrame opened in chunks) and avoid using too much RAM.
    Merges from_df into on_df while loading into memory on_df up to its chunksize defined at its creation.
    The input dataframes rest untouched.
    Args:
        from_df (pd.DataFrame):
        on_df (pd.io.parsers.TextFileReader): A pandas DataFrame opened with a `chunksize` parameter
        output_file (str): The resulting output file
        merge_fnc_param (dict): The pandas.DataFrame.merge() function named parameters
        on_df_tranformer (callable): A function which operates at each read of a chunk and apply some
            preprocessing to the read chunk. Signature: func(pd.Dataframe) -> pd.Dataframe
        nthreads (int): Number of threads to use for merging
        verbose (bool): Display information messages
    Returns:
        list: A list of tuple each containing the type and name of each columns of the resulting output file
    """
    global g_from_df, g_merge_fnc_param, g_on_df_tranformer, g_output_file
    g_from_df = from_df
    g_merge_fnc_param = merge_fnc_param
    g_on_df_tranformer = on_df_tranformer
    g_output_file = output_file

    start_time = time.time()
    header_set = False
    chunk_types = None
    output_files = []
    with Pool(processes=nthreads) as pool:
        output_files.append(pool.map(_evaluate_chunk, on_df))

    del g_from_df, g_merge_fnc_param, g_on_df_tranformer, g_output_file
    if verbose:
        print("--- Merging finished in %s minutes ---" % round((time.time() - start_time) / 60, 2))

    return chunk_types
