"""
This file is the main file to run the rsna project which can be found @
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
"""
import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.api_client import ApiClient

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))


def retrieve_dataset():
    zip_files = ["stage_1_detailed_class_info.csv.zip", "stage_1_train_labels.csv.zip",
                 "stage_1_test_images.zip", "stage_1_train_images.zip"]
    out_dir = script_dir / "input" / "rsna-pneumonia-detection-challenge"
    if not out_dir.exists():
        os.mkdir(out_dir)
        api = KaggleApi(ApiClient())
        api.authenticate()
        api.competition_download_files("rsna-pneumonia-detection-challenge", out_dir, quiet=False)
        print("Extracting files...")
        for file in zip_files:
            zip_ref = zipfile.ZipFile(out_dir / file, 'r')
            zip_ref.extractall(out_dir)
            zip_ref.close()
            os.remove(out_dir / file)
        print("Dataset downloaded!")
    else:
        print("Dataset already present in input dir, skipping...")


def main():
    batch_size = 32

    retrieve_dataset()
    print("Done!")


if __name__ == "__main__":
    main()
