"""
 A file to cache data for faster loading.
 E.g: Images turned to bcolz files
"""
import bcolz
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
