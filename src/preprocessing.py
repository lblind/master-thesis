"""
PREPROCESSING

Everything belong to the preprocessing part
"""

import os

import pandas as pd


def read_and_clean_csv(path_to_csv):
    if os.path.exists(path_to_csv) is False:
        raise ValueError(f"CSV file: {path_to_csv} not found.\n"
                         f"Please review your path definition and make sure the file exists.")
    df = pd.read_csv(path_to_csv)

    return df
