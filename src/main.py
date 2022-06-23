"""
MAIN
(Control)

Starting point of program execution
"""
import datetime

import pandas as pd
import preprocessing as preproc


if __name__ == "__main__":

    # 1. Read food prices, convert to excel and return as dfs
    df_central, df_north, df_south = preproc.read_and_clean_csvs_food_prices()

    path_to_longs = "../input/food-price-dta/longs and lats/MWI_markets.csv"

    df_markets_coord = pd.read_csv(path_to_longs)
    df_markets_coord.rename(columns={'MarketName': 'Market'}, inplace=True)

    df_all = pd.merge(df_north, df_markets_coord, on="Market", how="inner")
    # print(df_all)

    # EXTRACT SLICES/ RANGES OF VARIABLES
    range_time, range_long_market, range_lat_market = preproc.extract_time_long_lat_slice(df_all)

    #
    # # df_all.to_excel("../output/food_prices_long_lat.xlsx")

    path_climate = "../input/climate-dta/spei01.nc"
    df_spei = preproc.read_climate_data(path_climate, time_slice=range_time,
                                        long_slice=range_long_market, lat_slice=range_lat_market)
    # # match df all to df climate
    # df_all_climate = pd.merge(df_all, df_spei, on=["long", "lat"], how="inner")

    # print(df_all_climate.columns)

    # print(df_all.columns)

    # print(df_markets_coord)







