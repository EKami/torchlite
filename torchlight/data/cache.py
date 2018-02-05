"""
 A file to cache data for faster loading.
 E.g: Images turned to bcolz files
"""
import bcolz


def to_blosc_arrays(files, to_dir):
    """
    Turn a list of images to blosc and return the path
    to these images. Images stored as blosc on disk are read
    much faster than
    Args:
        files (list): A list of image files
        to_dir (str): The path to the stored blosc arrays
    Returns:
        list: A list of paths to the blosc images
    """
    # TODO use tqdm
    return files
