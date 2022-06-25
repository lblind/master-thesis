"""
MAIN
(Control)

Starting point of program execution
"""

import preprocessing as preproc


if __name__ == "__main__":

    # PART A) Get food price part of data
    # -----------------------------------

    # 1. Read food prices, convert to excel and return as merged df (for all regions)
    df_wfp = preproc.get_df_wfp_preprocessed()
    # 2. Read CSV containing market coordinates and merge to price data
    df_wfp_with_coords = preproc.read_and_merge_wfp_market_coords(df_wfp)
    # 3. Preparation for merge with Part B): Extract range of 3 main variables: time, longitude, latitude
    slice_time, slice_lon, slice_lat = preproc.extract_time_lon_lat_slice(df_wfp_with_coords)

    print(f"Slice Time: {slice_time}\nSlice Lon: {slice_lon}\nSlice Lat: {slice_lat}")

    # PART B) Get climate part of data (SPEI)
    # ---------------------------------------

    df_spei = preproc.read_climate_data(time_slice=slice_time, long_slice=slice_lon, lat_slice=slice_lat)

    # PART C) Merge Outcomes of Part A) and Part C) and store them as excel
    # ---------------------------------------------------------------------
    df_final = preproc.merge_food_price_and_climate_dfs(df_wfp_with_coords=df_wfp_with_coords, df_spei=df_spei)

    df_final = preproc.classify_droughts(df_final)

    # Summary statistics categorical
    counts = df_final["SpeiCat"].value_counts()
    print(f"Counts:\n{counts}")

    counts_drought = df_final["Drought"].value_counts()
    print(f"Counts Drought:\n{counts_drought}")
    print(f"Share of droughts: {28671/ (28671 + 115943)}")

    # Store intermediate results and final output as excel
    df_wfp.to_excel("../output/df_wfp.xlsx", na_rep="-")
    df_wfp_with_coords.to_excel("../output/df_wfp_with_coords.xlsx", na_rep="-")
    df_spei.to_excel("../output/df_spei.xlsx", na_rep="-")
    df_final.to_excel("../output/final-dta.xlsx", na_rep="-")

    print(f"\n----------------------------------------------------------------------------------------------------\n"
          f"PREPROCESSING: DONE.\nSuccessfully merged different datasets (wfp, wfp coords, spei)\nand stored them"
          f" as excel workbooks in the output folder.\n"
          f"Summary statistics:\n"
          f"Number of entries: {df_final.shape[0]}\n"
          f"Number of nan/missing values Prices: {df_final.Price.isna().sum()} (Share: "
          f"{df_final.Price.isna().sum()/ df_final.shape[0]})\n"
          f"Number of nan/missing values SPEI: {df_final.Spei.isna().sum()} (Share: "
          f"{df_final.Spei.isna().sum()/ df_final.shape[0]})\n"
          f"----------------------------------------------------------------------------------------------------\n")








