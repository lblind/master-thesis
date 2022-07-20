"""
STATISTICS
----------
"""
import pandas as pd

import utils


def mean_column_per_group(gdf_final, column="AdjPrice", group="District"):
    """
    Calculates the mean per group (e.g. District, Region, etc.).
    Creates a new column and adds it to the dataframe

    :param column:
    :param gdf_final: geopandas.DataFrame
        Input dataframe
    :param group: str
        Column name of gdf_final on which to aggregate/ compute the means.
    :return: gdf_merged: geopandas.DataFrame
    """

    # calculate the means per group
    df_means = gdf_final.groupby(group).apply(lambda x: x[column].mean())

    # convert pandas series to dataframe
    df_means = pd.DataFrame({f"Mean{column}Per{group}": df_means})

    # merge means column to dataframe
    gdf_merged = utils.merge_dfs_left(df_left=gdf_final, df_right=df_means, on=group)

    return gdf_merged
