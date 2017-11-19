import hashlib


def apply_encoding(dataframes, config_file, output_dir=None):
    """
    Apply
    Args:
        dataframes (list): List of dataframes on which to apply the encoding
        config_file (str): The config file containing the encoding configuration
        output_dir: (str, None): The output dir where the encoded files will
            be stored. If None is provided nothing is saved on disk. If a path is
            provided the encoded dataframes are saved on disk if this path does not
            contains them already. If it does them the existing files are opened and
            returned as DataFrames from this function.

    Returns:

    """
    # Check if the encoding has already been generated with
    # https://stackoverflow.com/questions/31567401/get-the-same-hash-value-for-a-pandas-dataframe-each-time
    pass