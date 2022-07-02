"""
MAIN
---------
(Control)

Starting point of program execution
"""
import os

import preprocessing as preproc
import pandas as pd


if __name__ == "__main__":

    # CONFIGURATION
    # -----------------------------------
    # 1) set country
    country = "Malawi"
    # country = "Somalia"
    # country = "Tanzania"
    # country = "Kenya"

    # 2) set commmodities to drop
    # dropped_commodities = ["Maize (white)", "Rice (imported)", "Sorghum (red)"]
    dropped_commodities = None

    # PART A) Get food price part of data
    # -----------------------------------

    # 1. Read food prices, convert to excel and return as df
    df_wfp = preproc.get_df_wfp_preprocessed_excel_per_country(country=country, dropped_commodities=dropped_commodities)
    # df_wfp = preproc.get_df_wfp_preprocessed_excel_per_region(country=country, dropped_commodities=dropped_commodities)

    # 2. Read CSV containing market coordinates and merge to price data
    df_wfp_with_coords = preproc.read_and_merge_wfp_market_coords(df_wfp=df_wfp, country=country)
    # 3. Preparation for merge with Part B): Extract range of 3 main variables: time, longitude, latitude
    slice_time, slice_lon, slice_lat = preproc.extract_time_lon_lat_slice(df_wfp_with_coords)

    print(f"Slice Time: {slice_time}\nSlice Lon: {slice_lon}\nSlice Lat: {slice_lat}")

    # PART B) Get climate part of data (SPEI)
    # ---------------------------------------

    df_spei = preproc.read_climate_data(time_slice=slice_time, long_slice=slice_lon, lat_slice=slice_lat, country=country)

    # PART C) Merge Outcomes of Part A) and Part C) and store them as excel
    # ---------------------------------------------------------------------
    df_final = preproc.merge_food_price_and_climate_dfs(df_wfp_with_coords=df_wfp_with_coords, df_spei=df_spei)

    # Check that everything went okay in preprocessing/ merge part
    if df_final.shape[0] != df_wfp.shape[0]:
        raise ValueError(f"Something went wrong in the preprocessing part.\n"
                         f"# rows of final/ merged df {df_final.shape[0]} should"
                         f" be the same as for the one of wfp {df_wfp.shape[0]}")

    df_final = preproc.classify_droughts(df_final)

    # Summary statistics categorical
    counts = df_final["SpeiCat"].value_counts()
    print(f"Counts:\n{counts}")

    counts_drought = df_final["Drought"].value_counts()
    print(f"Counts Drought:\n{counts_drought}")
    print(f"Share of droughts: {counts_drought[True]}/ {(counts_drought[True] + counts_drought[False])}")

    # Get the distinguished datasets
    df_drought, df_no_drought = preproc.separate_df_drought_non_drought(df_final)

    # Check missings
    preproc.summary_stats_prices_droughts(df_final=df_final)
    # preproc.summary_stats_missings(df_final=df_wfp)

    preproc.write_preprocessing_results_to_excel(df_wfp=df_wfp, df_wfp_with_coords=df_wfp_with_coords,
                                                 df_spei=df_spei, df_final=df_final, df_drought=df_drought,
                                                 df_no_drought=df_no_drought)















