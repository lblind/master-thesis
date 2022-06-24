"""
PREPROCESSING

Everything belonging to the preprocessing part
"""

import os
import pandas as pd
import xarray as xr
import netCDF4
import numpy as np
import datetime

from geopy.distance import great_circle

import math


def get_df_wfp_preprocessed(path_to_dir_wfp_csvs_per_region="../input/food-price-dta/csv"):
    """

    :param path_to_dir_wfp_csvs_per_region:
    :return:
    """
    if os.path.exists(path_to_dir_wfp_csvs_per_region) is False:
        raise ValueError(f"Directory containing csvs <{path_to_dir_wfp_csvs_per_region}> not found.\n"
                         f"Please review your path definition and make sure the directory exists.")

    central_path_to_csv_all = f"{path_to_dir_wfp_csvs_per_region}" \
                              f"/Central_All_Commodities_WFP_2022Jun14_Malawi_FoodPricesData.csv"
    northern_path_to_csv_all = f"{path_to_dir_wfp_csvs_per_region}" \
                               f"/Northern_All_Commodities_WFP_2022Jun14_Malawi_FoodPricesData.csv"
    southern_path_to_csv_all = f"{path_to_dir_wfp_csvs_per_region}" \
                               f"/Southern_All_Commodities_WFP_2022Jun14_Malawi_FoodPricesData.csv"

    df_central = pd.read_csv(central_path_to_csv_all)
    df_northern = pd.read_csv(northern_path_to_csv_all)
    df_southern = pd.read_csv(southern_path_to_csv_all)

    # southern: omit prices for: Maize (white) - retail, Rice (imported) - retail, Sorghum (red) - retail
    # as no data available for other regions

    print(df_southern.Commodity.unique())
    dropped_commodities = ["Maize (white) - retail", "Rice (imported) - retail", "Sorghum (red) - retail"]
    # df_southern = df_southern[df_southern.Commodity.isin(dropped_commodities) is False]
    print(df_southern.Commodity.unique())

    # mark dfs corresponding to their region and combine them as one df
    df_central["Region"] = "Central"
    df_northern["Region"] = "North"
    df_southern["Region"] = "South"

    df_merged_all_regions = pd.concat([df_central, df_northern, df_southern], ignore_index=True)
    print(df_merged_all_regions)

    # Some summary statistics
    df_regions_list = [df_central, df_northern, df_southern]

    for i, df in enumerate(df_regions_list):
        print(f"#------------------------------------------------------------------------\n"
              f"\nSummary statistic"
              f"# [{i}]\n"
              f"Size of Market: {df.Market.unique().size}\n"
              f"\n#------------------------------------------------------------------------\n")
        # print(df)
        # print(df.columns)
        # for column in df.columns:
        #     print(df[column].unique())
        #     # print(df["Commodity"].unique())
        #     # print(df["Year"].unique())
        #     # print(df["Market"].unique())

    print("Overall size of markets", len(df_merged_all_regions["Market"].unique()))
    return df_merged_all_regions


def read_and_merge_wfp_market_coords(df_wfp, path_to_csv_wfp_coords_markets="../input/food-price-dta/"
                                                                            "longs and lats/MWI_markets.csv"):
    """

    :param df_wfp:
    :param path_to_csv_wfp_coords_markets:

    :return:
    """
    # Read csv
    df_wfp_coords_markets = pd.read_csv(path_to_csv_wfp_coords_markets)
    # Rename column for merge
    df_wfp_coords_markets.rename(columns={'MarketName': 'Market'}, inplace=True)

    # Merge Food Price data with provided coordinates of markets
    df_wfp_coords = pd.merge(df_wfp, df_wfp_coords_markets, on="Market", how="inner")

    # df_wfp_coords_north = pd.merge(df_wfp_north, df_wfp_coords_markets, on="Market", how="inner")
    # df_wfp_coords_south = pd.merge(df_wfp_central, df_wfp_coords_markets, on="Market", how="inner")

    return df_wfp_coords


def extract_time_lon_lat_slice(df_wfp_coords):
    """

    :param df_wfp_coords:
    :return:
    """
    # EXTRACT SLICES/ RANGES OF VARIABLES

    # TIME
    max_year = df_wfp_coords.loc[df_wfp_coords["Year"].idxmax(), "Year"]
    max_month = df_wfp_coords.loc[df_wfp_coords["Month"].idxmax(), "Month"]
    max_day = 31  # set manually, as unit of WFP data = MONTH

    max_date = datetime.datetime(max_year, max_month, max_day)
    print("\nMax date\n", max_date)

    min_year = df_wfp_coords.loc[df_wfp_coords["Year"].idxmin(), "Year"]
    min_month = df_wfp_coords.loc[df_wfp_coords["Month"].idxmin(), "Month"]
    min_day = 1  # set manually, as unit of WFP data = MONTH

    min_date = datetime.datetime(min_year, min_month, min_day)
    print("\nMin date\n", min_date)

    range_time = slice(min_date, max_date)

    max_long_market = df_wfp_coords.loc[df_wfp_coords["MarketLongitude"].idxmax(), "MarketLongitude"]
    min_long_market = df_wfp_coords.loc[df_wfp_coords["MarketLongitude"].idxmin(), "MarketLongitude"]

    range_lon_market = slice(min_long_market, max_long_market)

    max_lat_market = df_wfp_coords.loc[df_wfp_coords["MarketLatitude"].idxmax(), "MarketLatitude"]
    min_lat_market = df_wfp_coords.loc[df_wfp_coords["MarketLatitude"].idxmin(), "MarketLatitude"]

    range_lat_market = slice(min_lat_market, max_lat_market)

    return range_time, range_lon_market, range_lat_market


def read_climate_data(time_slice, long_slice, lat_slice, path_to_netcdf="../input/climate-dta/spei01.nc"):
    """
    Reads the netcdf and converts it into a pandas df

    :param path_to_netcdf:
    :return:


    References
    ----------
    SPEI Database:

    """
    ds = xr.open_dataset(path_to_netcdf)
    # print(ds)

    # Just keep data that is relevant (time range, market lats, market lons)
    # ds = ds.sel(time=slice(start_time, end_time))
    ds = ds.sel(time=time_slice)
    ds = ds.sel(lon=long_slice)
    ds = ds.sel(lat=lat_slice)

    print(ds)
    print("\nCOORDINATES:\n", ds.coords)
    print("\nDIMENSIONS:\n", ds.dims)
    # Access METADATA
    for i, var in enumerate(ds.variables.values()):
        print(f"\nVARIABLE {i}:\n", var)

    # rint("First row:", ds[0, :, :])

    # extract the data per column
    lon_arr = ds.variables["lon"][:]
    lat_arr = ds.variables["lat"][:]
    time_arr = ds.variables["time"][:]
    spei_arr = ds.variables["spei"][:]

    df = ds.to_dataframe()
    df.to_excel("../output/climate_data.xlsx")

    df_excel = pd.read_excel("../output/climate_data.xlsx")

    print(df["spei"])
    # print(df.columns)
    print(df_excel.columns)

    # Propagate nan values (for lon, lat)
    df_excel[["lon", "lat"]] = df_excel[["lon", "lat"]].fillna(method="ffill")

    df_excel["Year"] = df_excel["time"]
    df_excel["Year"] = df_excel["Year"].apply(lambda x: x.year)

    df_excel["Month"] = df_excel["time"]
    df_excel["Month"] = df_excel["Month"].apply(lambda x: x.month)

    df_excel["Day"] = df_excel["time"]
    df_excel["Day"] = df_excel["Day"].apply(lambda x: x.day)

    df_excel.to_excel("../output/climate_data_preprocessed.xlsx")

    print(df_excel["time"])

    return df_excel


def closest_point_great_circles(point, neighbors):
    """

    :param point:
    :param neighbours:
    :return:
    """
    pass


def determine_closest_points_for_markets(df_spei, df_wfp_with_coords, algorithm="great circles"):
    """

    References
    ----------
    https://www.timvink.nl/closest-coordinates/

    :return:
    """

    # First: Create list of coordinates
    df_spei["tuple_lon_lat_spei"] = list(zip(df_spei["lat_spei"], df_spei["lon_spei"]))
    df_wfp_with_coords["tuple_lon_lat_markets"] = list(zip(df_wfp_with_coords["MarketLatitude"],
                                                           df_wfp_with_coords["MarketLongitude"]))
    unique_spei_coords = df_spei["tuple_lon_lat_spei"].unique() # 90
    unique_market_coords = df_wfp_with_coords["tuple_lon_lat_markets"].unique() # 121
    print(unique_market_coords, "\n", len(unique_market_coords))
    print(unique_spei_coords, "\n", len(unique_spei_coords))

    # create two new columns
    df_wfp_with_coords["lon_spei_nn"] = np.nan
    df_wfp_with_coords["lat_spei_nn"] = np.nan
    df_wfp_with_coords["distance_nn"] = np.nan

    for market_point in unique_market_coords:
        market_lat = market_point[0]
        market_lon = market_point[1]

        # print("Lat:", market_lat, "Lon:", market_lon)

        # iterate over all available spei points
        for i, spei_point in enumerate(unique_spei_coords):
            if i == 0:
                # assume min distance = with first spei coordinate
                min_distance = great_circle(market_point, spei_point)

                # remember coordinates of spei market
                min_spei_coords = spei_point

            else:
                curr_distance = great_circle(market_point, spei_point)
                if curr_distance < min_distance:
                    min_distance = curr_distance
                    min_spei_coords = spei_point

        # print(f"Min Distance: {min_distance}\nMin Spei: {min_spei_coords}")
        # assign information to min distance for market
        df_wfp_with_coords.loc[(df_wfp_with_coords["MarketLatitude"] ==
                               market_lat) & (df_wfp_with_coords["MarketLongitude"] == market_lon), ["distance_nn"]] \
            = min_distance
        df_wfp_with_coords.loc[(df_wfp_with_coords["MarketLatitude"] ==
                               market_lat) & (df_wfp_with_coords["MarketLongitude"] == market_lon), ["lat_spei_nn"]] \
            = min_spei_coords[0]
        df_wfp_with_coords.loc[(df_wfp_with_coords["MarketLatitude"] ==
                               market_lat) & (df_wfp_with_coords["MarketLongitude"] == market_lon), ["lon_spei_nn"]] \
            = min_spei_coords[1]

    # return df with information on min distance
    return df_wfp_with_coords





def merge_food_price_and_climate_dfs(df_wfp_with_coords, df_spei):
    """
    Merge dataframe of climate data with extended food price data

    :param df_wfp_with_coords:
    :param df_spei:

    :return:
    """
    print("Columns WFP:\n",df_wfp_with_coords.columns)
    print("Columns SPEI:\n", df_spei.columns)

    # Rename columns for merge
    df_spei.rename(columns={'lat': 'lat_spei'}, inplace=True)
    df_spei.rename(columns={'lon': 'lon_spei'}, inplace=True)

    print("Columns SPEI:\n", df_spei.columns)

    # add information on closest spei measure points for markets
    df_wfp_with_coords = determine_closest_points_for_markets(df_spei=df_spei, df_wfp_with_coords=df_wfp_with_coords)


    # df_spei.rename(columns={'lat': 'MarketLatitude'}, inplace=True)
    # df_spei.rename(columns={'lon': 'MarketLongitude'}, inplace=True)

    df_spei.rename(columns={'lat_spei': 'lat_spei_nn'}, inplace=True)
    df_spei.rename(columns={'lon_spei': 'lon_spei_nn'}, inplace=True)

    # Merge Food Price data with provided coordinates of markets (nearest neighbor market to
    df_final = pd.merge(df_wfp_with_coords, df_spei, on=["Year", "Month", "lat_spei_nn", "lon_spei_nn"],
                             how="inner")
    #
    df_final.to_excel("../output/final-dta.xlsx")
    #
    return df_final


