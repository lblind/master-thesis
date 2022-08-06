"""
ANALYSIS & STATISTICS
---------------------
Everything regarding further analysis and statistics (and its documentation)
E.g. some statistics are computed here and stored in Excel sheets.
"""
import os
import pandas as pd
import numpy as np

import utils
import preprocessing as preproc


def identify_spikes_per_commodity(df, spike_dev = 0, write_excel=True, state_preproc_step=4):
    """

    :param df:
    :param write_excel:
    :param state_preproc_step:
    :return:
    """
    # first: compute means per commodity
    mean_adj_price_per_commodity = df.groupby("Commodity")["AdjPrice"].mean()

    # store mean adj price in new column
    df["MeanAdjPrice"] = np.nan
    # create column to compute deviation from mean
    df["DevMean"] = np.nan
    # compute share of deviation from dev mean
    df["RelDevMean"] = np.nan
    # column to identify whether a spike has occured
    df["SpikeAdjPrice"] = np.nan

    for commodity in df.Commodity.unique():
        # create a view on the subset of the df that belongs to that commodity
        df_commodity = df[df.Commodity == commodity]

        # assign mean adjusted price per commodity
        df.loc[df.Commodity == commodity, "MeanAdjPrice"] = mean_adj_price_per_commodity[commodity]

        # compute deviation from mean per commodity
        df.loc[df.Commodity == commodity, "DevMean"] = df_commodity.AdjPrice - mean_adj_price_per_commodity[commodity]

        # create new view
        df_commodity = df[df.Commodity == commodity]

        # compute relative deviation from mean per commodity
        df.loc[df.Commodity == commodity, "RelDevMean"] = df_commodity.DevMean / mean_adj_price_per_commodity[commodity]

        # create new view
        df_commodity = df[df.Commodity == commodity]

        # identify spike -> if relative deviation > 10% or just at all deviatng
        # variant 1: just at all deviating from mean (dev > 0)
        df.loc[df.Commodity == commodity, "SpikeAdjPrice"] = df_commodity.DevMean > spike_dev * mean_adj_price_per_commodity[commodity]


    # make sure that spike is boolean and not object
    df.SpikeAdjPrice.astype("bool")

    if write_excel:

        country = df.Country.unique()[0]
        output_path = f"../output/{country}/spikes"
        if os.path.exists(output_path) is False:
            os.makedirs(output_path)

        df.to_excel(f"{output_path}/df-with-spikes-PREPROC-{state_preproc_step}.xlsx")

    return df



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


def sum_stats_range_of_variables(df, add_pad_months_time_span=0, preproc_step=1):
    """

    :param df:
    :return:
    """

    country = df.Country.unique()[0]

    df_commodities_dict = {}
    min_max_times_dict = {}

    for commodity in df.Commodity.unique():
        # Limit the dfs to specific subset
        df_commodity = df[df.Commodity == commodity]

        # call subset extraction function -> find union of time
        df_commodity, min_time, max_time = preproc.extract_df_subset_time_prices(df_commodity=df_commodity,
                                                                                 epsilon_month=add_pad_months_time_span)

        # call subset extraction function -> find intersection of time
        # (per region as interpolated by region afterwards)
        df_commodity, min_time, max_time = preproc.extract_intersec_subset_time_per_region_for_commodity(df_commodity=
                                                                                                         df_commodity)

        # store min and max time for further use/ documentation
        min_max_times_dict[commodity] = (min_time, max_time)

        # add result to dictionary
        df_commodities_dict[commodity] = df_commodity

    # Summary stats: write the subsets of time
    pd.DataFrame({
        "Commodity": min_max_times_dict.keys(),
        "TimeSpanMinY": [time_min.strftime("%Y") for time_min, time_max in min_max_times_dict.values()],
        "TimeSpanMinM": [time_min.strftime("%m") for time_min, time_max in min_max_times_dict.values()],
        "TimeSpanMaxY": [time_max.strftime("%Y") for time_min, time_max in min_max_times_dict.values()],
        "TimeSpanMaxM": [time_max.strftime("%m") for time_min, time_max in min_max_times_dict.values()],
        "Epsilon (Month)": [add_pad_months_time_span] * len(min_max_times_dict.keys())
    }).sort_values(by="Commodity").to_excel(
        f"../output/{country}/summary-statistics/preproc-STEP-{preproc_step}-time-spans-per-commodity.xlsx")


def sum_stats_prices(df, var_list_groups_by=None, excel_output_extension="-preproc-1",
                     commodity=None, return_df_by_group_sheet="General"):
    """
    Summary statistics for missing values in Prices AND droughts
    per: Commodity, Market, Region, Year

    :param return_df_by_group_sheet: str
        if the dataframe should be returned directly (in order to save reading it afterwards from the excel)
        define which sheet / df to return afterwards.
    :param commodity: str
        if df_final is only a subset for a specific commoditiy, denote that such that
        it will be stored in the correct folder
    :param var_list_groups_by: list
    :param df: pd.DataFrame
    :param excel_output_extension: str
    :return:
    """

    # Set default values for var_list_group_by
    if var_list_groups_by is None:
        var_list_groups_by = ["Market", "Region", "Commodity", "Year"]

    # read number of overall entries/ rows
    no_overall_entries = df.shape[0]

    # Make sure that output directory exists
    if commodity is not None:
        output_path_stats = f"../output/{df.Country.unique()[0]}/summary-statistics/{commodity}"
    else:
        output_path_stats = f"../output/{df.Country.unique()[0]}/summary-statistics"
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

        format_number = "#"
        format_share = "%"
        format_missings = "nan"

        # Missings per unique value in that group
        for value in df[group].unique():
            df_group_value = df[df[group] == value]
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

        df_sum_stats_group = pd.DataFrame({group: df[group].unique(),
                                           f"{format_number} overall entries": unique_value_size_list,
                                           f"{format_share} {group} / Country": share_unique_value_country_list,
                                           f"Price: {format_number} {format_missings}": na_values_list_price,
                                           f"Price: {format_share} {format_missings}": share_of_na_list_price,
                                           }
                                          )
        # sort values
        df_sum_stats_group.sort_values(by=group)
        # append summary statistics for that group to overall list
        # list_dfs_sum_stats.append(df_sum_stats_group)

        # append summary statistics for that group to overall dictionary
        dict_dfs_sum_stats[group] = df_sum_stats_group

    # General data
    df_sum_stats_general = pd.DataFrame({
        f"{format_number} {format_missings}": [df.Price.isna().sum()],
        f"{format_number} overall entries": [df.shape[0]],
        f"{format_share} {format_missings}": [df.Price.isna().sum() / df.shape[0]]
    }, ["Price"])

    # Write all dfs into one excel
    with pd.ExcelWriter(
            f"{output_path_stats}/{df.Country.unique()[0]}-sum-stats{excel_output_extension}.xlsx") as writer:
        df_sum_stats_general.to_excel(writer, sheet_name="General")

        for group in dict_dfs_sum_stats.keys():
            # take it and sort it by the group
            df_sum_stat = dict_dfs_sum_stats[group].sort_values(by=[group], ignore_index=True)
            df_sum_stat.to_excel(writer, sheet_name=group)

    # return a specific dataframe
    if return_df_by_group_sheet == "General":
        return df_sum_stats_general
    elif return_df_by_group_sheet in dict_dfs_sum_stats.keys():
        return dict_dfs_sum_stats[return_df_by_group_sheet].sort_values(by=[return_df_by_group_sheet],
                                                                        ignore_index=True)
    else:
        raise ValueError(f"Nothing will be returned, as "
                         f"{return_df_by_group_sheet} is neither "
                         f"General, nor part of the valid groups:"
                         f"{dict_dfs_sum_stats.keys()}.\n"
                         f"Plesae revise your definition.")


def sum_stats_prices_and_droughts(df, var_list_groups_by=None, excel_output_extension="-preproc-1",
                                  commodity=None, return_df_by_group_sheet="General"):
    """
    Summary statistics for missing values in Prices AND droughts
    per: Commodity, Market, Region, Year

    :param return_df_by_group_sheet: str
        if the dataframe should be returned directly (in order to save reading it afterwards from the excel)
        define which sheet / df to return afterwards.
    :param commodity: str
        if df_final is only a subset for a specific commoditiy, denote that such that
        it will be stored in the correct folder
    :param var_list_groups_by: list
    :param df: pd.DataFrame
    :param excel_output_extension: str
    :return:
    """

    # Set default values for var_list_group_by
    if var_list_groups_by is None:
        var_list_groups_by = ["Market", "Region", "Commodity", "Year"]

    # read number of overall entries/ rows
    no_overall_entries = df.shape[0]

    # Make sure that output directory exists
    if commodity is not None:
        output_path_stats = f"../output/{df.Country.unique()[0]}/summary-statistics/{commodity}"
    else:
        output_path_stats = f"../output/{df.Country.unique()[0]}/summary-statistics"
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
        for value in df[group].unique():
            df_group_value = df[df[group] == value]
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

        df_sum_stats_group = pd.DataFrame({group: df[group].unique(),
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
    df_droughts = df[df["Drought"] == True]
    df_ed = df_droughts[df_droughts["SpeiCat"] == "Extremely dry (ED)"]
    df_sd = df_droughts[df_droughts["SpeiCat"] == "Severely dry (SD)"]
    df_md = df_droughts[df_droughts["SpeiCat"] == "Moderately dry (MD)"]

    # General data
    df_sum_stats_general = pd.DataFrame({
        f"{format_number} {format_missings}": [df.Price.isna().sum(), df.Spei.isna().sum()],
        f"{format_number} overall entries": [df.shape[0], df.shape[0]],
        f"{format_share} {format_missings}": [df.Price.isna().sum() / df.shape[0],
                                              df.Drought.isna().sum() / df.shape[0]],
        f"{format_number} Droughts": [np.nan, df[df["Drought"] == True].shape[0]],
        f"{format_share} Droughts": [np.nan, df_droughts.shape[0] / df.shape[0]],
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
            f"{output_path_stats}/{df.Country.unique()[0]}-sum-stats{excel_output_extension}.xlsx") as writer:
        df_sum_stats_general.to_excel(writer, sheet_name="General")

        for group in dict_dfs_sum_stats.keys():
            # take it and sort it by the group
            df_sum_stat = dict_dfs_sum_stats[group].sort_values(by=[group], ignore_index=True)
            df_sum_stat.to_excel(writer, sheet_name=group)

    # return a specific dataframe
    if return_df_by_group_sheet == "General":
        return df_sum_stats_general
    elif return_df_by_group_sheet in dict_dfs_sum_stats.keys():
        return dict_dfs_sum_stats[return_df_by_group_sheet].sort_values(by=[return_df_by_group_sheet],
                                                                        ignore_index=True)
    else:
        raise ValueError(f"Nothing will be returned, as "
                         f"{return_df_by_group_sheet} is neither "
                         f"General, nor part of the valid groups:"
                         f"{dict_dfs_sum_stats.keys()}.\n"
                         f"Plesae revise your definition.")


def df_describe_excel(df, group_by_column=None, excel_extension="-final", column=None):
    """

    :param df:
    :return:
    """
    country = df.Country.unique()[0]
    sum_stats = df.describe()
    # Make sure that output directory exists

    output_path_stats = f"../output/{df.Country.unique()[0]}/summary-statistics/describe"
    if os.path.exists(output_path_stats) is False:
        os.makedirs(output_path_stats)

    # General describe
    sum_stats.to_excel(f"{output_path_stats}/{country}-describe-general{excel_extension}.xlsx")

    if group_by_column is not None:
        sum_stats_per_group = df.groupby(group_by_column).describe()

        if column is not None:
            sum_stats_per_group[column].to_excel(
                f"{output_path_stats}/{country}-describe-group-{group_by_column}{excel_extension}.xlsx",
                sheet_name=column)
        else:
            sum_stats_per_group.to_excel(
                f"{output_path_stats}/{country}-describe-group-{group_by_column}{excel_extension}.xlsx",
                sheet_name="All columns")
