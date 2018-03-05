import urllib.request
import os
from kaggle_data.downloader import KaggleDataDownloader
from tqdm import tqdm


class KaggleDatasetFetcher:
    """
        A tool used to automatically download datasets from Kaggle
        TODO: Use https://github.com/Kaggle/kaggle-api
    """

    @staticmethod
    def download_dataset(competition_name: str, competition_files: list,
                         competition_files_ext: list, output_folder: str):
        """
            Downloads the dataset and return the input paths.
            Do not download again if the data is already present.
            You need to define $KAGGLE_USER and $KAGGLE_PASSWD in your environment
            and you must accept the competition rules beforehand.

            This downloader uses https://github.com/EKami/kaggle-data-downloader
            and assumes everything is properly installed.
        Args:
            competition_name (str): The name of the competition
            competition_files (list): List of files for the competition (in their uncompressed format)
            competition_files_ext (list): List of extensions for the competition files in the same order
            as competition_files. Ex: 'zip', '7z', 'xz'
            output_folder (str): Path to save the downloaded files

        Returns:
            tuple: (file_names, files_path)
        """
        assert len(competition_files) == len(competition_files_ext), \
            "Length of competition_files and competition_files_ext do not match"
        datasets_path = [output_folder + f for f in competition_files]

        is_dataset_present = True
        for file in datasets_path:
            if not os.path.exists(file):
                is_dataset_present = False

        if not is_dataset_present:
            # Put your Kaggle user name and password in a $KAGGLE_USER and $KAGGLE_PASSWD env vars respectively
            downloader = KaggleDataDownloader(os.getenv("KAGGLE_USER"), os.getenv("KAGGLE_PASSWD"), competition_name)

            zipfiles = [file + "." + ext for file, ext in zip(competition_files, competition_files_ext)]
            for file in zipfiles:
                downloader.download_dataset(file, output_folder)

            # Unzip the files
            zipdatasets_path = [output_folder + f for f in zipfiles]
            for path in zipdatasets_path:
                downloader.decompress(path, output_folder)
                os.remove(path)
        else:
            print("All datasets are present.")

        return competition_files, datasets_path


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


class WebFetcher:
    """
        A tool used to automatically download datasets from the web
    """

    @staticmethod
    def download_dataset(url: str, output_folder: str, decompress: bool):
        """
            Downloads the dataset and return the input paths.
            Do not download again if the data is already present.
            Args:
                url (str): Http link to the archive
                output_folder (str): Path to save the downloaded files
                decompress (bool): To uncompress the downloaded archive
            Returns:
                tuple: (file_name, file_path)
        """
        file_name = os.path.split(url)[-1]
        output_file_arch = os.path.join(output_folder, file_name)
        if not os.path.exists(output_file_arch):
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            print('Beginning file download...')
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc="Downloading {}".format(file_name)) as t:
                file, _ = urllib.request.urlretrieve(url, output_file_arch, reporthook=t.update_to)
            print("Unzipping file...")
            if decompress:
                KaggleDataDownloader.decompress(file, output_folder)
        else:
            print("File already exists.")

        return file_name, output_file_arch
