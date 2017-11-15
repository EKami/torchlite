import os
from kaggle_data.downloader import KaggleDataDownloader
from tqdm import tqdm
import pandas as pd


class KaggleDatasetFetcher:
    """
        A tool used to automatically download datasets from Kaggle
    """

    @staticmethod
    def download_dataset(competition_name: str, competition_files: list,
                         output_folder: str, to_feather=False):
        """
            Downloads the dataset and return the input paths.
            You need to define $KAGGLE_USER and $KAGGLE_PASSWD in your environment
            and you must accept the competition rules beforehand.
        Args:
            competition_name (str): The name of the competition
            competition_files (list): List of files for the competition
            output_folder (str): Path to save the downloaded files
            to_feather (bool): Transform files to the father format

        Returns:
            tuple: (file_names, files_path)
        """

        datasets_path = [output_folder + f for f in competition_files]
        feather_files = [f.replace(".csv", ".feather") for f in competition_files]
        feather_datasets_path = [output_folder + f for f in feather_files]

        is_dataset_present = True

        if to_feather:
            for file in feather_datasets_path:
                if not os.path.exists(file):
                    is_dataset_present = False
        else:
            for file in datasets_path:
                if not os.path.exists(file):
                    is_dataset_present = False

        if not is_dataset_present:
            # Put your Kaggle user name and password in a $KAGGLE_USER and $KAGGLE_PASSWD env vars respectively
            downloader = KaggleDataDownloader(os.getenv("KAGGLE_USER"), os.getenv("KAGGLE_PASSWD"), competition_name)

            zipfiles = [file + ".7z" for file in competition_files]
            for file in zipfiles:
                downloader.download_dataset(file, output_folder)

            # Unzip the files
            zipdatasets_path = [output_folder + f for f in zipfiles]
            for path in zipdatasets_path:
                downloader.decompress(path, output_folder)
                os.remove(path)
        else:
            print("All datasets are present.")

        if to_feather:
            for file, path in tqdm(zip(competition_files, datasets_path), total=len(datasets_path)):
                feather_path = path.replace(".csv", ".feather")
                if not os.path.exists(feather_path):
                    print(f"Moving {file} to feather...")
                    df = pd.read_csv(path)
                    df.to_feather(feather_path)
            competition_files = feather_files
            datasets_path = feather_datasets_path

        return competition_files, datasets_path
