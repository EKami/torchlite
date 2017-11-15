import os
from kaggle_data.downloader import KaggleDataDownloader


class KaggleDatasetFetcher:
    """
        A tool used to automatically download datasets from Kaggle
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
