from configuration import *
import os
import zipfile
from tqdm import tqdm


def ensure_datasets(verbose=True):
    for name, folder, mask in zip(DATASET_NAMES, DATA_FOLDERS, DATASET_MASK):
        if mask != 1:
            continue

        folder_exists = os.path.exists(folder)
        if not folder_exists:
            if verbose:
                print(f"Will unzip {name} dataset to {folder}")
            with zipfile.ZipFile(folder + ".zip", "r") as zip_ref:
                for member in tqdm(zip_ref.infolist(), desc=f"Extracting {name}"):
                    zip_ref.extract(member, folder)
        elif verbose:
            print(f"No need to unzip {name} - it's already there")

if __name__ == "__main__":
    ensure_datasets()
