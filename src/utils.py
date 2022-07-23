"""
UTILS
-----
Auxiliary code snippets that are used by various modules
"""
import numpy as np
import pandas as pd
import os


def convert_excel_to_df(path_to_excel, sheet_name=None, nan_char="-"):
    """
    Reads the excel out of the given path and converts it to a
    pandas dataframe

    :param path_to_excel:
    :param sheet_name: None or str
        Sheet to read
        If None, the first sheet will be read per default
        Otherwise, specify a sheet name
    :param nan_char:
    :return:
    """
    if os.path.exists(path_to_excel) is False:
        raise ValueError("Cannot convert excel to df, as path:\n"
                         f"<{path_to_excel}> does not exist.\n"
                         f"Please revise the definition of your path.")

    if sheet_name is None:
        df = pd.read_excel(path_to_excel)
    else:
        df = pd.read_excel(path_to_excel, sheet_name=sheet_name, na_values=nan_char)

    # Omit index column
    # no table formatting in Excel
    if "Unnamed: 0" in df:
        # Omit index column (Previously written)
        df.drop(columns="Unnamed: 0", inplace=True)
    # user defined naming
    if "Index" in df:
        # Omit index column (previously written/ renamed after formatting)
        df.drop(columns="Index", inplace=True)
    # Excel default naming (after table formatting)
    if "Column1" in df:
        # Omit index column (previously written/ renamed after formatting)
        df.drop(columns="Column", inplace=True)

    # replace all - with nan (in case na_values didn't work
    df.replace("-", np.nan, inplace=True)

    return df

def merge_dfs_left(df_left, df_right, on):
    """
    Left outer join

    :param df_left:
    :param df_right:
    :param on:
    :param how:
    :return:
    """
    # Step 1: drop all possible duplicates
    # df_left = df_left.drop_duplicates(keep="first")
    # df_right = df_right.drop_duplicates(keep="first")

    how = "left"
    no_rows_before_merge = df_left.shape[0]

    df_merged = df_left.merge(df_right, on=on, how=how)

    if no_rows_before_merge != df_merged.shape[0]:
        raise ValueError(f"Something went wrong in merge. Number of rows is not the"
                         f"same. (Before merge: {no_rows_before_merge}, After merge: {df_merged.shape[0]})")

    return df_merged


