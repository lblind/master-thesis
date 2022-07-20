"""
STATISTICS
----------
"""
import pandas as pd

import utils

def mean_per_column(gdf_final, group="District"):
    """
    Calculates the mean per group (e.g. District, Region, etc.)

    :param group:
    :return:
    """
    print(gdf_final.head())
    print(gdf_final.District.unique())

    # calculate the means per group
    df_means = gdf_final.groupby(group).apply(lambda x : x["AdjPrice"].mean())

    # convert pandas series to dataframe
    df_means = pd.DataFrame(df_means)

    # merge means colun to dataframe
    gdf_merged = utils.merge_dfs_left(df_left=gdf_final,df_right= df_means, on=group)

    return gdf_merged