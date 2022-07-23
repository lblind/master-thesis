"""
STATISTICS
----------
"""
import os
import pandas as pd
import numpy as np

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

def summary_stats_prices_droughts(df_final, var_list_groups_by=None, excel_output_extension="-preproc-1",
                                  commodity=None, return_df_by_group_sheet="General"):
    """
    Summary statistics for missing values per market and
    commodity

    :param return_df_by_group_sheet: str
        if the dataframe should be returned directly (in order to save reading it afterwards from the excel)
        define which sheet / df to return afterwards.
    :param commodity: str
        if df_final is only a subset for a specific commoditiy, denote that such that
        it will be stored in the correct folder
    :param var_list_groups_by: list
    :param df_final: pd.DataFrame
    :param excel_output_extension: str
    :return:
    """

    # Set default values for var_list_group_by
    if var_list_groups_by is None:
        var_list_groups_by = ["Market", "Region", "Commodity", "Year"]

    # read number of overall entries/ rows
    no_overall_entries = df_final.shape[0]

    # Make sure that output directory exists
    if commodity is not None:
        output_path_stats = f"../output/{df_final.Country.unique()[0]}/summary-statistics/{commodity}"
    else:
        output_path_stats = f"../output/{df_final.Country.unique()[0]}/summary-statistics"
    if os.path.exists(output_path_stats) is False:
        os.makedirs(output_path_stats)

    # list_dfs_sum_stats = []
    dict_dfs_sum_stats = {}

    for group in var_list_groups_by:

        print(
            f"\n----------------------------------------------------------------------------------------------------\n"
            f"Missings per variable: <{group}>"
            f"\n----------------------------------------------------------------------------------------------------\n"
        )

        # List countaining the number of nan values per unique value in that variable
        na_values_list_price = []
        # List containing the share of nan values per unique value in that variable
        share_of_na_list_price = []
        # List containing the number of entries belonging to that unique value in the variable
        unique_value_size_list = []
        # List containing the share of that unique value in the overall number of possible values for that variable
        share_unique_value_country_list = []

        # List countaining the number of nan values per unique value in that variable
        na_values_list_drought = []
        # List containing the share of nan values per unique value in that variable
        share_of_na_list_drought = []
        # List containing the share of that unique value in the overall number of possible values for that variable
        share_unique_value_country_list_drought = []

        list_no_droughts = []

        # Share of droughts in that group
        share_of_droughts_list = []

        # Further distinction:
        # "Extremely dry (ED)", "Severely dry (SD)", "Moderately dry (MD)"
        share_of_droughts_list_ed = []
        share_of_droughts_list_sd = []
        share_of_droughts_list_md = []
        no_of_droughts_list_ed = []
        no_of_droughts_list_sd = []
        no_of_droughts_list_md = []

        format_number = "#"
        format_share = "%"
        format_missings = "nan"

        # SPEI DATA: A drought occured or not (not look at spei, but drought yes/ no)

        # Missings per unique value in that group
        for value in df_final[group].unique():
            df_group_value = df_final[df_final[group] == value]
            no_entries_group_value = df_group_value.shape[0]

            na_values_price = df_group_value.Price.isna().sum()
            share_of_na_price = na_values_price / no_entries_group_value
            print(
                f"\n[Var: {group}] Value <{value}>\n# missings (Price): {na_values_price}\nShare: {share_of_na_price}")

            # Overall statistics value per group
            unique_value_size_list.append(no_entries_group_value)
            share_unique_value_country_list.append(unique_value_size_list[-1] / no_overall_entries)

            # Statistics price
            na_values_list_price.append(na_values_price)
            share_of_na_list_price.append(share_of_na_price)

            # Statistics drought
            na_values_drought = df_group_value.Drought.isna().sum()
            share_of_na_drought = na_values_drought / no_entries_group_value

            # Calculate overall share of droughts in that group (including missings)
            df_group_value_drought = df_group_value[df_group_value["Drought"] == True]

            no_droughts_group_value = df_group_value_drought.shape[0]
            list_no_droughts.append(no_droughts_group_value)

            share_of_droughts = no_droughts_group_value / no_entries_group_value
            print(
                f"\n[Var: {group}] Value <{value}>\n# missings (Drought): {na_values_drought}\nShare: {share_of_na_drought}")

            # "Extremely dry (ED)", "Severely dry (SD)", "Moderately dry (MD)"
            no_of_extreme_droughts = \
                df_group_value_drought[df_group_value_drought["SpeiCat"] == "Extremely dry (ED)"].shape[0]
            share_of_extreme_droughts = 0 if no_droughts_group_value == 0 else no_of_extreme_droughts / no_droughts_group_value

            no_of_severe_droughts = \
                df_group_value_drought[df_group_value_drought["SpeiCat"] == "Severely dry (SD)"].shape[0]
            share_of_severe_droughts = 0 if no_droughts_group_value == 0 else no_of_severe_droughts / no_droughts_group_value

            no_of_moderate_droughts = \
                df_group_value_drought[df_group_value_drought["SpeiCat"] == "Moderately dry (MD)"].shape[0]
            share_of_moderate_droughts = 0 if no_droughts_group_value == 0 else no_of_severe_droughts / no_droughts_group_value

            na_values_list_drought.append(na_values_drought)
            share_of_na_list_drought.append(share_of_na_drought)
            share_of_droughts_list.append(share_of_droughts)

            share_of_droughts_list_ed.append(share_of_extreme_droughts)
            share_of_droughts_list_sd.append(share_of_severe_droughts)
            share_of_droughts_list_md.append(share_of_moderate_droughts)

            no_of_droughts_list_ed.append(no_of_extreme_droughts)
            no_of_droughts_list_sd.append(no_of_severe_droughts)
            no_of_droughts_list_md.append(no_of_moderate_droughts)

        df_sum_stats_group = pd.DataFrame({group: df_final[group].unique(),
                                           f"{format_number} overall entries": unique_value_size_list,
                                           f"{format_share} {group} / Country": share_unique_value_country_list,
                                           f"Price: {format_number} {format_missings}": na_values_list_price,
                                           f"Price: {format_share} {format_missings}": share_of_na_list_price,
                                           f"Drought: {format_number} {format_missings}": na_values_list_drought,
                                           f"Drought: {format_share} {format_missings}": share_of_na_list_drought,
                                           f"Drought: {format_share} Droughts": share_of_droughts_list,
                                           f"{format_share} Extreme (of Droughts)":
                                               share_of_droughts_list_ed,
                                           f"{format_share} Severe (of Droughts)":
                                               share_of_droughts_list_sd,
                                           f"{format_share} Moderate (of Droughts)":
                                               share_of_droughts_list_md,
                                           f"{format_number} Extreme Droughts": no_of_droughts_list_ed,
                                           f"{format_number} Severe Droughts": no_of_droughts_list_sd,
                                           f"{format_number} Moderate Droughts": no_of_droughts_list_md,
                                           }
                                          )
        # sort values
        df_sum_stats_group.sort_values(by=group)
        # append summary statistics for that group to overall list
        # list_dfs_sum_stats.append(df_sum_stats_group)

        # append summary statistics for that group to overall dictionary
        dict_dfs_sum_stats[group] = df_sum_stats_group

    # Create summary statistics for general statistics
    df_droughts = df_final[df_final["Drought"] == True]
    df_ed = df_droughts[df_droughts["SpeiCat"] == "Extremely dry (ED)"]
    df_sd = df_droughts[df_droughts["SpeiCat"] == "Severely dry (SD)"]
    df_md = df_droughts[df_droughts["SpeiCat"] == "Moderately dry (MD)"]

    # General data
    df_sum_stats_general = pd.DataFrame({
        f"{format_number} {format_missings}": [df_final.Price.isna().sum(), df_final.Spei.isna().sum()],
        f"{format_number} overall entries": [df_final.shape[0], df_final.shape[0]],
        f"{format_share} {format_missings}": [df_final.Price.isna().sum() / df_final.shape[0],
                                              df_final.Drought.isna().sum() / df_final.shape[0]],
        f"{format_number} Droughts": [np.nan, df_final[df_final["Drought"] == True].shape[0]],
        f"{format_share} Droughts": [np.nan, df_droughts.shape[0] / df_final.shape[0]],
        f"{format_share} Extreme (of Droughts)": [np.nan, df_ed.shape[0] / df_droughts.shape[0]],
        f"{format_share} Severe (of Droughts)": [np.nan, df_sd.shape[0] / df_droughts.shape[0]],
        f"{format_share} Moderate (of Droughts)": [np.nan, df_md.shape[0] / df_droughts.shape[0]],
        f"{format_number} Extreme Droughts": [np.nan, df_ed.shape[0]],
        f"{format_number} Severe Droughts": [np.nan, df_sd.shape[0]],
        f"{format_number} Moderate Droughts": [np.nan, df_md.shape[0]],
    }, ["Price", "Drought"])
    # df_sum_stats_general.set_index(["Price", "Drought"])

    # Write all dfs into one excel
    with pd.ExcelWriter(
            f"{output_path_stats}/{df_final.Country.unique()[0]}-sum-stats{excel_output_extension}.xlsx") as writer:
        df_sum_stats_general.to_excel(writer, sheet_name="General")

        for group in dict_dfs_sum_stats.keys():
            # take it and sort it by the group
            df_sum_stat = dict_dfs_sum_stats[group].sort_values(by=[group], ignore_index=True)
            df_sum_stat.to_excel(writer, sheet_name=group)

    # return a specific dataframe
    if return_df_by_group_sheet == "General":
        return df_sum_stats_general
    elif return_df_by_group_sheet in dict_dfs_sum_stats.keys():
        return dict_dfs_sum_stats[return_df_by_group_sheet].sort_values(by=[return_df_by_group_sheet], ignore_index=True)
    else:
        raise ValueError(f"Nothing will be returned, as "
                         f"{return_df_by_group_sheet} is neither "
                         f"General, nor part of the valid groups:"
                         f"{dict_dfs_sum_stats.keys()}.\n"
                         f"Plesae revise your definition.")