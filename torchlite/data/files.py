"""
 A file to cache data for faster loading.
 E.g: Images turned to bcolz files
"""
from pathlib import Path
import shutil
import bcolz
import numpy as np
from skimage import io
from tqdm import tqdm
import os


def to_blosc_arrays(files, to_dir):
    """
    Turn a list of images to blosc and return the path
    to these images. Images stored as blosc on disk are read
    much faster than standard image formats.

    If the cached files already exists they won't be cached again.
    Args:
        files (list): A list of image files
        to_dir (str): The path to the stored blosc arrays
    Returns:
        list: A list of paths to the blosc images
    """
    files_exists = True
    blosc_files = []
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)

    for file_path in files:
        _, file_name = os.path.split(file_path)
        img_blosc_path = os.path.join(to_dir, file_name)
        blosc_files.append(img_blosc_path)
        if not os.path.isdir(img_blosc_path):
            files_exists = False

    if files_exists:
        print("Cache files already generated")
    else:
        for file_path, img_blosc_path in tqdm(zip(files, blosc_files), desc="Caching files", total=len(files)):
            image = io.imread(file_path)
            img = bcolz.carray(image, rootdir=img_blosc_path, mode="w")
            img.flush()

    return blosc_files


def create_dir_if_not_exists(path):
    """
    If the given path does not exists create them recursively
    Args:
        path (str): The path to the directory

    Returns:
        Path: The path to the folder
    """
    os.makedirs(path, exist_ok=True)
    return Path(path)


def del_dir_if_exists(path):
    """
    If the given path does exist, the folder and its contents are deleted
    and recreated anew
    Args:
        path (str): The path to the directory

    Returns:
        Path: The path to the folder
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=False)
    return Path(path)


def get_files(path):
    """
    Returns the files from a folder
    Args:
        path (str): The directory to inspect

    Returns:
        list: A list of files
    """
    all_files = os.listdir(path)
    ret_files = [os.path.join(path, file) for file in all_files if os.path.isfile(os.path.join(path, file))]
    return ret_files


def get_file_names(file_paths, with_extension=True):
    """
    Given a path to multiple files return their name only in the same order
    Args:
        with_extension (bool): True to keep the file extension
        file_paths (list): A list of file paths

    Returns:
        list: A list of file names
    """
    ret = []
    for file in file_paths:
        if with_extension:
            name = Path(file).resolve().name
        else:
            name = Path(file).resolve().stem
        ret.append(name)
    return ret


def get_labels_from_folders(path, y_mapping=None):
    """
    Get labels from folder names as well as the absolute path to the files
    from the folders
    Args:
        path (str): The directory to inspect
        y_mapping (dict): If the labels were already mapped to an integer,
        give that mapping here in the form of {index: label}

    Returns:
        files (tuple): A tuple containing (file_path, label)
        y_mapping (dict): The mapping between the label and their index
    """
    y_all = [label for label in os.listdir(path) if os.path.isdir(os.path.join(path, label))]
    if not y_mapping:
        y_mapping = {v: int(k) for k, v in enumerate(y_all)}

    files = [[(os.path.join(path, label, file), y_mapping[label]) for file in
              os.listdir(os.path.join(path, label))] for label in y_all]
    files = np.array(files).reshape(-1, 2)
    return files, y_mapping
