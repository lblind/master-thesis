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
    print(df_all)

    # EXTRACT SLICES/ RANGES OF VARIABLES

    # TIME
    max_year = df_all.loc[df_all["Year"].idxmax()]
    max_month = df_all.loc[df_all["Month"].idxmax()]
    max_day = 31 # set manually, as unit of WFP data = MONTH

    max_date = datetime.datetime(max_year, max_month, max_day)

    min_year = df_all.loc[df_all["Year"].idxmin()]
    min_month = df_all.loc[df_all["Month"].idxmin()]
    min_day = 1  # set manually, as unit of WFP data = MONTH

    min_date = datetime.datetime(min_year, min_month, min_day)

    range_time = slice(min_date, max_date)

    max_long_market = df_all.loc[df_all["MarketLongitude"].idxmax()]
    min_long_market = df_all.loc[df_all["MarketLongitude"].idxmin()]

    range_long_market = slice(min_long_market, max_long_market)

    max_lat_market = df_all.loc[df_all["MarketLatitude"].idxmax()]
    min_lat_market = df_all.loc[df_all["MarketLatitude"].idxmin()]

    range_lat_market = slice(min_lat_market, max_lat_market)



    #
    # # df_all.to_excel("../output/food_prices_long_lat.xlsx")

    path_climate = "../input/climate-dta/spei01.nc"
    df_spei = preproc.read_climate_data(path_climate, )
    # # match df all to df climate
    # df_all_climate = pd.merge(df_all, df_spei, on=["long", "lat"], how="inner")

    # print(df_all_climate.columns)

    # print(df_all.columns)

    # print(df_markets_coord)







