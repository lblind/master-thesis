"""
PREPROCESSING
-------------

Everything belonging to the preprocessing part
"""
import glob
import os
import pandas as pd
import xarray as xr
import netCDF4
import numpy as np
import datetime

from geopy.distance import great_circle

import math


def get_df_wfp_preprocessed_one_excel(country, dropped_commodities=None):
    """
    Reads the provided csv for the country

    :param country:
    :param dropped_commodities:

    :return:
    """
    # if no commodities is given -> default value:
    if dropped_commodities is None:
        dropped_commodities = ["Maize (white)", "Rice (imported)", "Sorghum (red)"]

    path_to_dir_wfp_csvs_per_region = f"../input/{country}/food-price-dta/csv-prices"
    if os.path.exists(path_to_dir_wfp_csvs_per_region) is False:
        raise ValueError(f"Directory containing csvs <{path_to_dir_wfp_csvs_per_region}> not found.\n"
                         f"Please review your path definition and make sure the directory exists.")

    # iterate over all csvs in that folder
    for file in glob.glob(f"{path_to_dir_wfp_csvs_per_region}/*.csv"):
        df_country = pd.read_csv(file)

        # Extract region name / rename column
        df_country.rename(columns={"Admin 1" : "Region"}, inplace=True)

    # summary stats region
    for region_name in df_country.Region.unique():
        # print unique commodities
        print(f"\nRegion: {region_name}\nNo. of Markets: {df_country.Market.unique().size}\n"
              f"Commodities before omission:\n{df_country.Commodity.unique()}")

    # drop commodities (southern)
    df_country = df_country[~df_country.Commodity.isin(dropped_commodities)]
    print("Unique Commodities after omission:\n", df_country.Commodity.unique())

    print(f"Overall number of markets entire country ({country})", len(df_country["Market"].unique()))
    return df_country


def get_df_wfp_preprocessed_excel_per_region(country, dropped_commodities=None):
    """
    Reads the different csvs per region

    :param country:
    :param dropped_commodities:

    :return:
    """
    # if no commodities is given -> default value:
    if dropped_commodities is None:
        dropped_commodities = ["Maize (white)", "Rice (imported)", "Sorghum (red)"]

    path_to_dir_wfp_csvs_per_region = f"../input/{country}/food-price-dta/csv-prices"
    if os.path.exists(path_to_dir_wfp_csvs_per_region) is False:
        raise ValueError(f"Directory containing csvs <{path_to_dir_wfp_csvs_per_region}> not found.\n"
                         f"Please review your path definition and make sure the directory exists.")

    df_regions_list = []
    # iterate over all csvs in that folder
    for file in glob.glob(f"{path_to_dir_wfp_csvs_per_region}/*.csv"):
        # Extract region name
        csv_file_name = os.path.basename(file)
        region_name = csv_file_name.split("_")[0]

        df_region = pd.read_csv(file)

        # extract region name
        df_region["Region"] = region_name
        df_regions_list.append(df_region)

        # print unique commodities
        print(f"\nRegion: {region_name}\nNo. of Markets: {df_region.Market.unique().size}\n"
              f"Commodities before ommission:\n{df_region.Commodity.unique()}")

        # drop commodities (southern)
        df_region = df_region[~df_region.Commodity.isin(dropped_commodities)]
        print("Unique Commodities after omission:\n", df_region.Commodity.unique())

    df_merged_all_regions = pd.concat(df_regions_list, ignore_index=True)
    print(df_merged_all_regions)

    print(f"Overall number of markets entire country ({country})", len(df_merged_all_regions["Market"].unique()))
    return df_merged_all_regions


def read_and_merge_wfp_market_coords(df_wfp, country):
    """
    Match the wfp market coordinates (lats, lons) to the wfp price data

    :param country:
    :param df_wfp:
    :param path_to_csv_wfp_coords_markets:

    :return:
    """

    path_to_csv_wfp_coords_markets = f"../input/{country}/food-price-dta/csv-lons-and-lats/MWI_markets.csv"

    # Read csv-prices
    df_wfp_coords_markets = pd.read_csv(path_to_csv_wfp_coords_markets)
    # Rename column for merge
    df_wfp_coords_markets.rename(columns={'MarketName': 'Market'}, inplace=True)

    # Merge Food Price data with provided coordinates of markets
    df_wfp_coords = pd.merge(df_wfp, df_wfp_coords_markets, on="Market", how="left")

    # df_wfp_coords_north = pd.merge(df_wfp_north, df_wfp_coords_markets, on="Market", how="inner")
    # df_wfp_coords_south = pd.merge(df_wfp_central, df_wfp_coords_markets, on="Market", how="inner")

    return df_wfp_coords


def extract_time_lon_lat_slice(df_wfp_coords):
    """
    Extract the slices that belong to the wfp data

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


def read_climate_data(time_slice, long_slice, lat_slice, country, path_to_netcdf="../input/Global/climate-dta/spei01.nc"):
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

    # extract the data per column
    lon_arr = ds.variables["lon"][:]
    lat_arr = ds.variables["lat"][:]
    time_arr = ds.variables["time"][:]
    spei_arr = ds.variables["spei"][:]

    # Create dir if non existent
    output_path = f"../output/{country}/intermediate-results"

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    # store relevant slice in dataframe
    df = ds.to_dataframe()
    df.to_excel(f"{output_path}/{country}-tmp-spei-sliced.xlsx")

    # read slice again (proper format)
    df_excel = pd.read_excel(f"{output_path}/{country}-tmp-spei-sliced.xlsx")

    # delete temporary excel file again
    os.remove(f"{output_path}/{country}-tmp-spei-sliced.xlsx")

    # print(df["spei"])
    # # print(df.columns)
    # print(df_excel.columns)

    # Propagate nan values (for lon, lat)
    df_excel[["lon", "lat"]] = df_excel[["lon", "lat"]].fillna(method="ffill")

    df_excel["Year"] = df_excel["time"]
    df_excel["Year"] = df_excel["Year"].apply(lambda x: x.year)

    df_excel["Month"] = df_excel["time"]
    df_excel["Month"] = df_excel["Month"].apply(lambda x: x.month)

    df_excel["Day"] = df_excel["time"]
    df_excel["Day"] = df_excel["Day"].apply(lambda x: x.day)

    # store results as excel
    # df_excel.to_excel(f"{output_path}/{country}-spei-sliced-preprocessed.xlsx")

    return df_excel


def determine_closest_points_for_markets(df_spei, df_wfp_with_coords):
    """
    For each market, determine the closest spei measure point (algorithm = great circles)

    References
    ----------
    https://www.timvink.nl/closest-coordinates/

    :return:
    """

    # First: Create list of coordinates
    df_spei["tuple_lat_lon_spei"] = list(zip(df_spei["lat_spei"], df_spei["lon_spei"]))
    df_wfp_with_coords["tuple_lat_lon_markets"] = list(zip(df_wfp_with_coords["MarketLatitude"],
                                                           df_wfp_with_coords["MarketLongitude"]))
    unique_spei_coords = df_spei["tuple_lat_lon_spei"].unique()  # 90
    unique_market_coords = df_wfp_with_coords["tuple_lat_lon_markets"].unique()  # 121
    # print(unique_market_coords, "\n", len(unique_market_coords))
    # print(unique_spei_coords, "\n", len(unique_spei_coords))

    # create two new columns
    df_wfp_with_coords["lat_spei_nn"] = np.nan
    df_wfp_with_coords["lon_spei_nn"] = np.nan
    df_wfp_with_coords["distance_nn"] = np.nan

    for market_point in unique_market_coords:
        market_lat = market_point[0]
        market_lon = market_point[1]

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

    For that, a nearest neighbor matching technique for each market to the next spei measure point is used
    (metric: great circle distance/ haversine distance)


    :param df_wfp_with_coords:
    :param df_spei:

    :return: df_final: pd.DataFrame
        Final Dataset containing both wfp prices, market coordinates and spei indicators
    """
    # print("Columns WFP:\n",df_wfp_with_coords.columns)
    # print("Columns SPEI:\n", df_spei.columns)

    # Rename columns for merge
    df_spei.rename(columns={'lat': 'lat_spei'}, inplace=True)
    df_spei.rename(columns={'lon': 'lon_spei'}, inplace=True)

    # add information on closest spei measure points for markets
    df_wfp_with_coords = determine_closest_points_for_markets(df_spei=df_spei, df_wfp_with_coords=df_wfp_with_coords)

    df_spei.rename(columns={'lat_spei': 'lat_spei_nn'}, inplace=True)
    df_spei.rename(columns={'lon_spei': 'lon_spei_nn'}, inplace=True)

    # Merge Food Price data with provided coordinates of markets (nearest neighbor market to
    # JOIN SPEI: (lat_spei, lon_spei) ON WFP nearest neighbour (lat_spei, lon_spei)
    df_final = pd.merge(df_wfp_with_coords, df_spei, on=["Year", "Month", "lat_spei_nn", "lon_spei_nn"],
                        how="left")

    # BEAUTIFY DF

    # Some renaming
    # Mark new columns with * prefix, camel case
    df_final.rename(columns={"Data Source": "DataSourceWFP",
                             "lon_spei_nn": "*LonSpeiNN",
                             "lat_spei_nn": "*LatSpeiNN",
                             "distance_nn": "*DistanceNN",
                             "time": "TimeSpei",
                             "spei": "Spei",
                             "Day": "*DaySpei",
                             "Region": "Region",
                             "tuple_lat_lon_markets": "*TupleLatLonMarkets",
                             "tuple_lat_lon_spei": "*TupleLatLonSpei",
                             "Price Type": "PriceType"
                             }, inplace=True
                    )

    set_cols_before_reordering = set(df_final.columns.tolist())

    # Some reordering
    df_final = df_final.reindex(columns=["Country",
                                         "Region",
                                         "Market",
                                         "MarketID",

                                         "Year",
                                         "Month",
                                         "Commodity",
                                         "Unit",
                                         "Currency",
                                         "Price",
                                         "Spei",

                                         "*DistanceNN",
                                         "MarketCreateDate",
                                         "PriceType",
                                         "DataSourceWFP",
                                         "Adm0Code",
                                         "Adm1Code",
                                         "Adm2Code",
                                         "MarketLatitude",
                                         "MarketLongitude",
                                         "*TupleLatLonMarkets",
                                         "*LatSpeiNN",
                                         "*LonSpeiNN",
                                         "*TupleLatLonSpei",
                                         "TimeSpei",
                                         "*DaySpei"
                                         ])

    set_cols_after_reordering = set(df_final.columns.tolist())

    diff_cols = set_cols_before_reordering.difference(set_cols_after_reordering)

    if len(diff_cols) != 0:
        raise ValueError(f"Error in reordering the columns of the final df.\nPossibly omitted columns: {diff_cols}")

    return df_final


def classify_droughts(df_final):
    """
    For each point given, classify whether a drought has occurred or not

    :param df_final:
    :return:

    References
    ----------
    https://www.researchgate.net/figure/SPEI-drought-index-categories_tbl1_283244485
    """
    # bins = [-np.inf, -2, -1.5, -1, -0.99, 0.99, 1.49, 1.99]
    bins = [-np.inf, -2, -1.5, -1, 0.99, 1.49, 1.99, np.inf]
    category_names = ["Extremely dry (ED)", "Severely dry (SD)", "Moderately dry (MD)",
                      "Near normal (NN)",
                      "Moderately wet (MW)", "Very wet (VW)", "Extremely wet (EW)"]

    # Spei categories
    df_final["SpeiCat"] = pd.cut(x=df_final["Spei"], bins=bins, labels=category_names, right=True)

    # Simple boolean flag
    df_final["Drought"] = df_final["Spei"] <= -1

    return df_final


def separate_df_drought_non_drought(df_final_classified):
    """
    Separate the already classified dataframe (drought/ non-drought)
    into two datasets (drought/ non-drought)

    :param df_final_classified: pd.DataFrame
    :return: df_drought, df_no_drought: pd.DataFrame(s)
    """
    df_drought = df_final_classified[df_final_classified["Drought"]]
    df_no_drought = df_final_classified[~df_final_classified["Drought"]]

    return df_drought, df_no_drought


def summary_stats_missings(df_final):
    """
    Summary statistics for missing values per market and
    commodity

    :param df_final:
    :return:
    """

    print(f"\n----------------------------------------------------------------------------------------------------\n"
          f"Missings per MARKET"
          f"\n----------------------------------------------------------------------------------------------------\n"
          )

    output_path_stats = f"../output/{df_final.Country.unique()[0]}/summary-statistics"
    if os.path.exists(output_path_stats) is False:
        os.makedirs(output_path_stats)

    na_values_list = []
    share_of_na_list = []
    market_size_list = []
    # Missings per Market

    for market in df_final["Market"].unique():
        # print(market)
        na_values = df_final[df_final.Market == market].Price.isna().sum()
        share_of_na = na_values / df_final[df_final.Market == market].shape[0]
        print(f"\nMarket: {market}\n# missings: {na_values}\nShare: {share_of_na}")
        # df_sum_stats_market = pd.concat([df_sum_stats_market, ])

        na_values_list.append(na_values)
        share_of_na_list.append(share_of_na)
        market_size_list.append(df_final[df_final.Market == market].shape[0])

    df_sum_stats_market = pd.DataFrame({"Market" : df_final["Market"].unique(),
                                        "No. missings" : na_values_list,
                                        "No. overall entries" : market_size_list,
                                        "Share missings" : share_of_na_list},
                                       )

    print(f"\n----------------------------------------------------------------------------------------------------\n"
          f"Missings per COMMODITY"
          f"\n----------------------------------------------------------------------------------------------------\n"
          )

    na_values_list = []
    share_of_na_list = []
    commodities_size_list = []
    # Missings per Commodity
    for commodity in df_final["Commodity"].unique():
        na_values = df_final[df_final.Commodity == commodity].Price.isna().sum()
        share_of_na = na_values / df_final[df_final.Commodity == commodity].shape[0]
        print(f"\nCommodity: {commodity}\n# missings: {na_values}\nShare: {share_of_na}")
        # df_sum_stats_market = pd.concat([df_sum_stats_market, ])

        commodities_size_list.append(df_final[df_final.Commodity == commodity].shape[0])

        na_values_list.append(na_values)
        share_of_na_list.append(share_of_na)

    df_sum_stats_commodity = pd.DataFrame({"Commodity" : df_final["Commodity"].unique(),
                                        "No. missings" : na_values_list,
                                        "No. overall entries" : commodities_size_list,
                                        "Share missings" : share_of_na_list},
                                       )

    print(f"\n----------------------------------------------------------------------------------------------------\n"
          f"Missings per REGION"
          f"\n----------------------------------------------------------------------------------------------------\n"
          )
    # Missings per Region
    na_values_list = []
    share_of_na_list = []
    region_size_list = []
    for region in df_final["Region"].unique():
        na_values = df_final[df_final["Region"] == region].Price.isna().sum()
        share_of_na = na_values / df_final[df_final["Region"] == region].shape[0]
        print(f"\nRegion: {region}\n# missings: {na_values}\nShare: {share_of_na}")
        # df_sum_stats_market = pd.concat([df_sum_stats_market, ])

        na_values_list.append(na_values)
        share_of_na_list.append(share_of_na)
        region_size_list.append(df_final[df_final["Region"] == region].shape[0])

    df_sum_stats_region = pd.DataFrame({"Region" : df_final["Region"].unique(),
                                        "No. missings" : na_values_list,
                                        "No. overall entries" : region_size_list,
                                        "Share missings" : share_of_na_list},
                                       )

    # General data:
    df_sum_stats_general = pd.DataFrame({
        "No. missings" : [df_final.Price.isna().sum()],
        "No. overall entries": [df_final.shape[0]],
        "Share missings" : df_final.Price.isna().sum()/ df_final.shape[0]
    })

    # Write all dfs into one excel
    with pd.ExcelWriter(f"{output_path_stats}/{df_final.Country.unique()[0]}-missing-values.xlsx") as writer:
        df_sum_stats_general.to_excel(writer, sheet_name="General")
        df_sum_stats_market.to_excel(writer, sheet_name="Markets")
        df_sum_stats_commodity.to_excel(writer, sheet_name="Commodity")
        df_sum_stats_region.to_excel(writer, sheet_name="Region")


def write_preprocessing_results_to_excel(df_wfp, df_wfp_with_coords, df_spei, df_final, df_drought, df_no_drought):
    """

    :param df_wfp:
    :param df_wfp_with_coords:
    :param df_spei:
    :param df_final:
    :param df_drought:
    :param df_no_drought:
    :return:
    """
    # read country out of dataset
    country = df_wfp.Country.unique()[0]

    # Store intermediate results and final output as excel
    output_path_final = f"../output/{country}"
    output_path_intermediate = f"{output_path_final}/intermediate-results"
    if os.path.exists(output_path_intermediate) is False:
        os.makedirs(output_path_intermediate)

    df_wfp.to_excel(f"{output_path_intermediate}/df_wfp.xlsx", na_rep="-")
    df_wfp_with_coords.to_excel(f"{output_path_intermediate}/df_wfp_with_coords.xlsx", na_rep="-")
    df_spei.to_excel(f"{output_path_intermediate}/df_spei.xlsx", na_rep="-")

    df_drought.to_excel(f"{output_path_final}/{country}-drought.xlsx", na_rep="-")
    df_no_drought.to_excel(f"{output_path_final}/{country}-no-drought.xlsx", na_rep="-")
    df_final.to_excel(f"{output_path_final}/{country}-final-dta.xlsx", na_rep="-")

    print(f"Df drought shape: {df_drought.shape}\ndf_no_drought: {df_no_drought.shape}")

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




