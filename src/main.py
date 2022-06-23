"""
MAIN
(Control)

Starting point of program execution
"""
import datetime

import pandas as pd
import preprocessing as preproc


if __name__ == "__main__":

    # PART A) Get food price part of data

    # 1. Read food prices, convert to excel and return as merged df (for all regions)
    df_wfp = preproc.get_df_wfp_preprocessed()
    # 2. Read CSV containing market coordinates and merge to price data
    df_wfp_with_coords = preproc.read_and_merge_wfp_market_coords(df_wfp)
    # 3. Preparation for merge with Part B): Extract range of 3 main variables: time, longitude, latitude
    slice_time, slice_lon, slice_lat = preproc.extract_time_long_lat_slice(df_wfp_with_coords)

    # PART B) Get climate part of data (SPEI)

    df_spei = preproc.read_climate_data(time_slice=slice_time, long_slice=slice_lon, lat_slice=slice_lat)

    # # match df all to df climate
    # df_all_climate = pd.merge(df_all, df_spei, on=["long", "lat"], how="inner")

    # print(df_all_climate.columns)

    # print(df_all.columns)

    # print(df_markets_coord)







