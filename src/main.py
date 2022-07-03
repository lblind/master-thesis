"""
MAIN
---------
(Control)

Starting point of program execution
"""
import os

import preprocessing as preproc
import pandas as pd
import utils

if __name__ == "__main__":

    # CONFIGURATION
    # -----------------------------------
    # 1) set country
    country = "Malawi"
    # country = "Somalia"
    # country = "Tanzania"
    # country = "Kenya"

    # 2) set commmodities to drop
    dropped_commodities = ["Maize (white)", "Rice (imported)", "Sorghum (red)"]
    # dropped_commodities = None

    # TODO: Outcomment this block if dataset has not yet been created

    # print("\n# ------------------------------------------------------------------------------------------------------\n"
    #       "# PREPROC: PHASE 1 (MERGE DATA)"
    #       "\n# ------------------------------------------------------------------------------------------------------\n")
    #
    # # PART A) Get food price part of data
    # # -----------------------------------
    #
    # # 1. Read food prices, convert to excel and return as df
    # df_wfp = preproc.get_df_wfp_preprocessed_excel_per_country(country=country, dropped_commodities=dropped_commodities)
    # # df_wfp = preproc.get_df_wfp_preprocessed_excel_per_region(country=country, dropped_commodities=dropped_commodities)
    #
    # # 2. Read CSV containing market coordinates and merge to price data
    # df_wfp_with_coords = preproc.read_and_merge_wfp_market_coords(df_wfp=df_wfp, country=country)
    # # 3. Preparation for merge with Part B): Extract range of 3 main variables: time, longitude, latitude
    # slice_time, slice_lon, slice_lat = preproc.extract_time_lon_lat_slice(df_wfp_with_coords)
    #
    # print(f"Slice Time: {slice_time}\nSlice Lon: {slice_lon}\nSlice Lat: {slice_lat}")
    #
    # # PART B) Get climate part of data (SPEI)
    # # ---------------------------------------
    #
    # df_spei = preproc.read_climate_data(time_slice=slice_time, long_slice=slice_lon, lat_slice=slice_lat,
    #                                     country=country)
    #
    # # PART C) Merge Outcomes of Part A) and Part C) and store them as excel
    # # ---------------------------------------------------------------------
    # df_final = preproc.merge_food_price_and_climate_dfs(df_wfp_with_coords=df_wfp_with_coords, df_spei=df_spei)
    #
    # # Check that everything went okay in preprocessing/ merge part
    # if df_final.shape[0] != df_wfp.shape[0]:
    #     raise ValueError(f"Something went wrong in the preprocessing part.\n"
    #                      f"# rows of final/ merged df {df_final.shape[0]} should"
    #                      f" be the same as for the one of wfp {df_wfp.shape[0]}")
    #
    # df_final = preproc.classify_droughts(df_final)
    #
    # # Summary statistics categorical
    # counts = df_final["SpeiCat"].value_counts()
    # print(f"Counts:\n{counts}")
    #
    # counts_drought = df_final["Drought"].value_counts()
    # print(f"Counts Drought:\n{counts_drought}")
    # print(f"Share of droughts: {counts_drought[True]}/ {(counts_drought[True] + counts_drought[False])}")
    #
    # # Get the distinguished datasets
    # df_drought, df_no_drought = preproc.separate_df_drought_non_drought(df_final)
    #
    # # Check missings
    # preproc.summary_stats_prices_droughts(df_final=df_final)
    # # preproc.summary_stats_missings(df_final=df_wfp)
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # # TAKE CARE OF MISSING DATA -> SUBSET
    # # ------------------------------------------------------------------------------------------------------------------
    #
    # print("\n# ------------------------------------------------------------------------------------------------------\n"
    #       "# PREPROC: PHASE 2 (MISSING DATA: SUBSET CREATION)"
    #       "\n# ------------------------------------------------------------------------------------------------------\n")
    #
    # print("\n# ------------------------------------------------------------------------------------------------------\n"
    #       "# PREPROC: PHASE 2.1 (DROUGHTS)"
    #       "\n# ------------------------------------------------------------------------------------------------------\n")
    # # drop 2021, 2022 as no data for droughts (/SPEI) is available for those (even though WFP data is)
    # years_to_drop = [2021, 2022]
    # print(f"Dropping years: {years_to_drop}")
    # df_final = preproc.drop_years(df_final=df_final, years_list=years_to_drop)
    #
    # preproc.summary_stats_prices_droughts(df_final=df_final, excel_output_extension="-preproc-2")
    #
    # print("\n# ------------------------------------------------------------------------------------------------------\n"
    #       "# PREPROC: PHASE 2.2 (PRICES)"
    #       "\n# ------------------------------------------------------------------------------------------------------\n")
    #
    # # 65 (30%), 60 (27%, 30% also for central region)
    # cut_off_percentile = 60
    #
    # # Cut all regions with missing >= cut_off_percentile of missing values
    # df_final = preproc.drop_missing_decile_per_region_prices(
    #     path_excel_sum_stats=f"../output/{country}/summary-statistics/{country}-"
    #                          f"sum-stats-preproc-2.xlsx",
    #     df_final=df_final,
    #     cut_off_percentile=cut_off_percentile, excel_output_extension=f"-{cut_off_percentile}p")
    #
    # # Write sum stats
    # preproc.summary_stats_prices_droughts(df_final=df_final, excel_output_extension=f"-preproc-3-{cut_off_percentile}p")
    #
    # # EXTRAPOLATE REGIONAL PATTERNS
    # # doesn't extrapolate tails for interpolation method: cubic
    # df_final = preproc.extrapolate_prices_regional_patterns(df_final=df_final, interpolation_method="linear")
    #
    # # Write sum stats
    # preproc.summary_stats_prices_droughts(df_final=df_final, excel_output_extension=f"-preproc-4-{cut_off_percentile}p")
    #
    # preproc.write_preprocessing_results_to_excel(df_wfp=df_wfp, df_wfp_with_coords=df_wfp_with_coords,
    #                                              df_spei=df_spei, df_final=df_final, df_drought=df_drought,
    #                                              df_no_drought=df_no_drought)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# ANALYSIS"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    path_to_final_df = f"../output/{country}/{country}-final-dta.xlsx"
    df_final = utils.convert_excel_to_df(path_to_final_df)
