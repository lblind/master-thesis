"""
UTILS
-----
Auxiliary code snipplets that are used by various modules
"""
import pandas as pd
import os


def convert_excel_to_df(path_to_excel, sheet_name=None):
    """
    Reads the excel out of the given path and converts it to a
    pandas dataframe

    :param path_to_excel:
    :param sheet_name:
    :return:
    """
    if os.path.exists(path_to_excel) is False:
        raise ValueError("Cannot convert excel to df, as path:\n"
                         f"<{path_to_excel}> does not exist.\n"
                         f"Please revise the definition of your path.")

    if sheet_name is None:
        df = pd.read_excel(path_to_excel)
    else:
        df = pd.read_excel(path_to_excel, sheet_name=sheet_name)

    # Omit index column (Previously written)
    df.drop(columns="Unnamed: 0", inplace=True)

    return df
