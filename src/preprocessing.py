"""
PREPROCESSING
-------------

Everything belonging to the preprocessing part
"""
import glob
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import netCDF4
import numpy as np
import datetime
import utils

from geopy.distance import great_circle

import scipy

import math


def get_df_wfp_preprocessed_excel_country_method(country, dropped_commodities=None):
    """
    Reads the provided csv for the country

    :param country:
    :param dropped_commodities:

    :return:
    """
    # # if no commodities is given -> default value:
    # if dropped_commodities is None:
    #     dropped_commodities = ["Maize (white)", "Rice (imported)", "Sorghum (red)"]

    path_to_dir_wfp_csvs_per_region = f"../input/{country}/food-price-dta/csv-prices"
    if os.path.exists(path_to_dir_wfp_csvs_per_region) is False:
        raise ValueError(f"Directory containing csvs <{path_to_dir_wfp_csvs_per_region}> not found.\n"
                         f"Please review your path definition and make sure the directory exists.")

    # iterate over all csvs in that folder
    for file in glob.glob(f"{path_to_dir_wfp_csvs_per_region}/*.csv"):
        df_country = pd.read_csv(file)

        # Extract region name / rename column
        df_country.rename(columns={"Admin 1": "Region"}, inplace=True)

    # summary stats region
    for region_name in df_country.Region.unique():
        # print unique commodities
        print(f"\nRegion: {region_name}\nNo. of Markets: {df_country.Market.unique().size}\n"
              f"Commodities before omission:\n{df_country.Commodity.unique()}")

    if dropped_commodities is not None:
        # drop commodities
        df_country = df_country[~df_country.Commodity.isin(dropped_commodities)]
        print(f"Unique Commodities after omission of:\n{dropped_commodities}\n", df_country.Commodity.unique())

        print(f"Overall number of markets entire country ({country})", len(df_country["Market"].unique()))
    return df_country


def get_df_wfp_preprocessed_excel_region_method(country, dropped_commodities=None):
    """
    Reads the different csvs per region

    Difference to :link get_df_wfp_preprocessed_excel_country_method:
        Difference in propagating missing values (doesn't drop missing values)

    :param country:
    :param dropped_commodities:

    :return:
    """
    # # if no commodities is given -> default value:
    # if dropped_commodities is None:
    #     dropped_commodities = ["Maize (white)", "Rice (imported)", "Sorghum (red)"]

    path_to_dir_wfp_csvs_per_region = f"../input/{country}/food-price-dta/csv-prices/Regions"
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

        # print unique commodities
        print(f"\nRegion: {region_name}\nNo. of Markets: {df_region.Market.unique().size}\n"
              f"Commodities before omission:\n{df_region.Commodity.unique()}")

        if dropped_commodities is not None:
            print("Dropping: ")
            # drop commodities
            df_region = df_region[~df_region.Commodity.isin(dropped_commodities)]
            print("Unique Commodities after omission:\n", df_region.Commodity.unique())

        # append dataframe to list
        df_regions_list.append(df_region)

    df_merged_all_regions = pd.concat(df_regions_list, ignore_index=True)
    print(df_merged_all_regions)

    print(f"Overall number of markets entire country ({country})", len(df_merged_all_regions["Market"].unique()))
    return df_merged_all_regions


def check_markets_per_commodity_time(df_wfp):
    """
    Check for each commodity, that the same amount of entries per month/ year
    exist. If not, fill them up with missing values.

    :param df_wfp:
    :return:
    """
    country = df_wfp.Country.unique()[0]

    dfs_common_markets_per_commodity = []
    dfs_freq_markets_per_commodity = []
    max_frequencies_per_commodity = []

    for commodity in df_wfp.Commodity.unique():
        df_wfp_commodity = df_wfp[df_wfp.Commodity == commodity]
        markets_commodity = df_wfp_commodity.Market.unique()

        dict_freq_market = {}
        # create a dictionary for each market (counting in how many delta ts/ months it is present)
        dict_freq_market = dict(zip(markets_commodity, [0] * len(markets_commodity)))

        common_elements = np.array([])
        for year in df_wfp.Year.unique():
            df_wfp_year = df_wfp_commodity[df_wfp_commodity.Year == year]
            for month in df_wfp.Month.unique():
                df_wfp_year_month = df_wfp_year[df_wfp_year.Month == month]

                markets_month = df_wfp_year_month.Market.unique()

                # count all values that occured
                for market in markets_month:
                    dict_freq_market[market] += 1

                if np.array_equal(markets_commodity, markets_month) is False:
                    # find intersection
                    common_elements = markets_month[np.in1d(markets_month, common_elements)]
                    common_elements = markets_commodity[np.in1d(markets_commodity, common_elements)]
                    # raise ValueError(f"[{year}, {month}] does not contain all markets for commodity <{commodity}>\n"
                    #                  f"Should be ({len(markets_commodity)}): {markets_commodity})\n"
                    #                  f"Is ({len(markets_month)}): {markets_month}")

        max_frequency_commodity = len(df_wfp.Year.unique()) * len(df_wfp.Month.unique())
        dfs_common_markets_per_commodity.append(pd.DataFrame(common_elements))
        dfs_freq_markets_per_commodity.append(pd.DataFrame(dict_freq_market.values(), index=dict_freq_market.keys()))

    # Write all dfs into one excel
    with pd.ExcelWriter(
            f"../output/{country}/summary-statistics/intersection-markets-per-commodity.xlsx") as writer:

        if len(dfs_freq_markets_per_commodity) != len(df_wfp.Commodity.unique()) or \
                len(dfs_common_markets_per_commodity) != len(df_wfp.Commodity.unique()):
            raise ValueError(f"For each commodity, exactly one corresponding dataframe needs to be stored\n"
                             f"(Commodites: {len(df_wfp.Commodity.unique())})\n"
                             f"Frequencies per market: {len(dfs_freq_markets_per_commodity)}\n"
                             f"Common markets: {len(dfs_common_markets_per_commodity)}")

        for i, commodity in enumerate(df_wfp.Commodity.unique()):
            df_comm_markets = dfs_common_markets_per_commodity[i]
            df_freq_markets = dfs_freq_markets_per_commodity[i]

            df_comm_markets.to_excel(writer, sheet_name=f"{commodity} - Common Markets")
            df_freq_markets.to_excel(writer, sheet_name=f"{commodity} - Frequency Markets")

    print("CHECKED FOR MISSINGS IN COMBINATIONS")
    return df_wfp


def adjust_food_prices(country, df_wfp, data_source="WFP"):
    """
    Adjust food prices to one common price level (most recent one)
    If WFP: Food inflation

    Preprocessing Step input csv WFP:
    - Deleted last comma in the header row (was too much), otherwise new column
    - Removed "" for the 2 following columns in header (time, value(percent), otherwise they are not detected
      as separate columns

    - Replaced all "" with blank (via Excel) - otherwise separation is not properly detected

    :param inflation_data:
    :return:

    References
    ----------
    Inflation/ Deflation: https://towardsdatascience.com/adjusting-prices-for-inflation-in-pandas-daaaa782cd89

    """

    path_to_inflation_dir = f"../input/{country}/inflation-dta/{data_source}"

    if os.path.exists(path_to_inflation_dir) is False:
        raise ValueError(f"Deflation data not defined. Please put the downloaded inflation csv of the chosen source"
                         f" into the following"
                         f"folder: {path_to_inflation_dir}.\n(Source: {data_source})")

    for i, filename in enumerate(glob.glob(path_to_inflation_dir + "/*.csv")):
        # don't read last row into excel
        inflation_df = pd.read_csv(filename, sep=",", header=0, skipfooter=1, parse_dates=[1],
                                   engine="python")

        print(f"Read inflation dir from csv {filename}.")

        # extract year and month for inflation data
        inflation_df["Year"] = inflation_df.Time.dt.year
        inflation_df["Month"] = inflation_df.Time.dt.month

        # extract inflation data
        only_inflation_df = inflation_df[inflation_df.Indicator == "Inflation"]
        # Drop indicator column as no use for it anymore
        only_inflation_df = only_inflation_df.drop(columns=["Indicator"])
        # Rename columns
        only_inflation_df = only_inflation_df.rename(columns={"Value (percent)": "Inflation [%]",
                                                              "Time": "TimeInflation"})

        # only keep food inflation (not inflation data)
        food_inflation_df = inflation_df[inflation_df.Indicator == "Food Inflation"]
        # Drop indicator column as no use for it anymore
        food_inflation_df = food_inflation_df.drop(columns=["Indicator"])

        # write base year
        base_year = food_inflation_df["Year"].iloc[-1]
        base_month = food_inflation_df["Month"].iloc[-1]

        food_inflation_df["PriceAdjBaseDate(Y,M)"] = f"({base_year}, {base_month})"


        # Create an index multiplier (based on last entry in dataset (-1) -> most current as base year)
        food_inflation_df["FoodInflationMult"] = food_inflation_df["Value (percent)"].iloc[-1] / food_inflation_df[
            "Value (percent)"]

        # Rename columns
        food_inflation_df = food_inflation_df.rename(columns={"Value (percent)": "FoodInflation [%]",
                                                              "Time": "TimeFoodInflation"})

        # write it as excel
        food_inflation_df.to_excel(f"{path_to_inflation_dir}/{country}-food-inflation-dta.xlsx")

        # merge df wfp to inflation data
        df_wfp = df_wfp.merge(food_inflation_df, on=["Year", "Month"], how="left")

        # merge inflation data to it
        df_wfp = df_wfp.merge(only_inflation_df, on=["Year", "Month"], how="left")

        # make sure that prices are float, raise error if invalid types are encountered
        df_wfp["Price"] = pd.to_numeric(df_wfp["Price"], errors="raise")

        # Adjust prices to real terms / price levels in most recent year (2022, 4 in our case)
        df_wfp["AdjPrice"] = df_wfp["Price"] * df_wfp["FoodInflationMult"]

        df_wfp.to_excel(f"{path_to_inflation_dir}/{country}-inflation-merged.xlsx")

        if i > 0:
            warnings.warn(f"More than one csv detected in dir for inflation: {path_to_inflation_dir}.\n"
                          f"({os.listdir(path_to_inflation_dir)})\n"
                          f" Only the first"
                          f"one will be considered.")
            break

    return df_wfp


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


def read_climate_data(time_slice, long_slice, lat_slice, country,
                      path_to_netcdf="../input/Global/climate-dta/spei01.nc"):
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
                                         "Inflation [%]",
                                         "FoodInflation [%]",
                                         "FoodInflationMult",
                                         "PriceAdjBaseDate(Y,M)",
                                         "AdjPrice",
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
                                         "*DaySpei",

                                         "TimeFoodInflation",
                                         "TimeInflation",


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
    # df_final["Drought"] = df_final["Spei"] <= -1
    df_final["Drought"] = df_final["SpeiCat"].isin(["Extremely dry (ED)", "Severely dry (SD)", "Moderately dry (MD)"])

    # Replace values with nan if Spei/ SpeiCat is nan
    df_final.loc[df_final["SpeiCat"].isna(), "Drought"] = np.nan

    return df_final


def separate_df_drought_non_drought(df_final_classified):
    """
    Separate the already classified dataframe (drought/ non-drought)
    into two datasets (drought/ non-drought)
    (ignores nan values)

    :param df_final_classified: pd.DataFrame
    :return: df_drought, df_no_drought: pd.DataFrame(s)
    """
    # # drop rows that are nan in drought
    # df_without_nan = df_final_classified.dropna(subset=["Drought"])
    #
    # df_drought = df_without_nan[df_without_nan["Drought"]]
    # df_no_drought = df_without_nan[~df_without_nan["Drought"]]
    df_drought = df_final_classified[df_final_classified["Drought"] == True]
    df_no_drought = df_final_classified[df_final_classified["Drought"] == False]

    return df_drought, df_no_drought


def summary_stats_prices_droughts(df_final, var_list_groups_by=None, excel_output_extension="-preproc-1"):
    """
    Summary statistics for missing values per market and
    commodity

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
    output_path_stats = f"../output/{df_final.Country.unique()[0]}/summary-statistics"
    if os.path.exists(output_path_stats) is False:
        os.makedirs(output_path_stats)

    list_dfs_sum_stats = []

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
        list_dfs_sum_stats.append(df_sum_stats_group)

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

        for i, df_sum_stat in enumerate(list_dfs_sum_stats):
            group = var_list_groups_by[i]
            df_sum_stat.to_excel(writer, sheet_name=group)


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
          f"Basic Summary statistics:\n"
          f"Number of entries: {df_final.shape[0]}\n"
          f"Number of nan/missing values Prices: {df_final.Price.isna().sum()} (Share: "
          f"{df_final.Price.isna().sum() / df_final.shape[0]})\n"
          f"Number of nan/missing values Drought: {df_final.Drought.isna().sum()} (Share: "
          f"{df_final.Drought.isna().sum() / df_final.shape[0]})\n"
          f"----------------------------------------------------------------------------------------------------\n")


def drop_years(df_final, years_list):
    """
    Drops all data for specific years

    (e.g. 2021, 2022 as no data for SPEI for these years (even though data on food prices)

    :param df_final:
    :param years_list:
    :return:
    """
    # Just keep years for
    return df_final[~df_final["Year"].isin(years_list)]


def drop_missing_decile_per_region_prices(path_excel_sum_stats, df_final, cut_off_percentile=90,
                                          excel_output_extension=""):
    """
    Drops a certain decile of markets

    :param df_sum_stats:
    :return:
    """
    country = df_final.Country.unique()[0]

    # read summary stats
    df_sum_stats_all_markets = pd.read_excel(path_excel_sum_stats, sheet_name="Market")

    # Drop index column
    df_sum_stats_all_markets.drop(columns="Unnamed: 0", inplace=True)

    markets_to_cut_per_region_dict = {}
    cut_offs_per_region_dict = {}

    markets_to_cut_list = []

    # for all regions...
    for region in df_final.Region.unique():
        # 1) Extract markets belonging to that region
        df_final_region = df_final[df_final["Region"] == region]
        unique_markets_region = df_final_region.Market.unique()

        # 2) Extract sum stats for these markets
        df_sum_stats_markets_region = df_sum_stats_all_markets[
            df_sum_stats_all_markets.Market.isin(unique_markets_region)]

        # 3) Extract list of missings for that market
        df_col_missings_prices = df_sum_stats_markets_region["Price: % nan"]
        if df_col_missings_prices.shape[0] == 0:
            raise ValueError("SOMETHING IS WRONG HERE. NO ROWS FOUND FOR MISSING VALUES PRICES.")

        # 4) find the 9th decile:
        cut_off_decile_value = np.percentile(df_col_missings_prices, cut_off_percentile)

        # 5) Find all the markets that belong to that cut-off decile (& their shares)
        markets_to_cut_df_region = df_sum_stats_markets_region[df_col_missings_prices >= cut_off_decile_value]
        # markets_to_cut_df_region = markets_to_cut_df_region[["Market", "Price: % nan"]]

        # 6) Store dataframe & cutoff for that region (Market names + share of missings)
        markets_to_cut_per_region_dict[f"{region}"] = markets_to_cut_df_region
        cut_offs_per_region_dict[f"{region}"] = cut_off_decile_value

        # 7) Store markets to cut in a list
        markets_to_cut_list.extend(markets_to_cut_df_region.Market.tolist())

    # 8) Only keep those that don't belong to the cut markets
    df_reduced = df_final[~df_final["Market"].isin(markets_to_cut_list)]

    # 9) Create statistics -> Store the dropped markets & their shares of missings
    output_dir = f"../output/{country}/summary-statistics/preproc-3-dropped-markets"
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    # Write all dfs into one excel
    with pd.ExcelWriter(
            f"{output_dir}/{country}-dropped-markets{excel_output_extension}.xlsx") as writer:

        # Write cut-offs as excel
        df_cut_offs = pd.DataFrame({"Cut-off": cut_offs_per_region_dict.values()},
                                   cut_offs_per_region_dict.keys())
        df_cut_offs.to_excel(writer, sheet_name=f"Cut-offs (percentile {cut_off_percentile}")

        # Write all dropped markets to excel
        # iterate over all dropped regions
        for region, df_region in markets_to_cut_per_region_dict.items():
            df_region.to_excel(writer, sheet_name=region)

    # returned reduced dataframe
    return df_reduced


def extrapolate_prices_regional_patterns(df_final, interpolation_method="linear"):
    """
    Extrapolates missing values in Prices based on regional patterns

    :param df_final: pd.DataFrame
    :param interpolation_method: str
        Interpolation method used to extrapolate the missings. For more information, cf. References
    :return:

    References
    ----------
    Interpolation method:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html

    Plot Interpolation:
    https://www.geeksforgeeks.org/scipy-interpolation/#:~:text=Interpolation%20is%20a%20technique%20of%20constructing%20data%20points,in%20many%20ways%20some%20of%20them%20are%20%3A
    """
    # Make sure that prices are sorted chronologically
    print("Sorting dataframe chronologically")
    df_final = df_final.sort_values(by=["TimeSpei", "Market"], ignore_index=True)

    country = df_final.Country.unique()[0]

    if len(df_final.Currency.unique()) != 1:
        raise ValueError(f"More than one currency found in df: {df_final.Currency.unique()}.\n"
                         f"Cannot visualize prices without one common currency.\n"
                         f"Please standardize your prices to one currency.")
    currency = df_final.Currency.unique()[0]

    dfs_extrapolated_per_region = {}

    alpha = 0.5
    for region in df_final.Region.unique():
        print(f"Extrapolating Prices for region: <{region}> (Method: {interpolation_method})")

        # extract dataframe for that region
        df_region = df_final[df_final.Region == region]

        # make a scatter plot (only for the inflation-adjusted prices)
        # plt.scatter(df_region.TimeSpei, df_region.Price, label="Original points",
        #             alpha=alpha)
        plt.scatter(df_region.TimeSpei, df_region["AdjPrice"], label="Original points",
                    alpha=alpha)

        # Plot and color markets separately
        # for market in df_region.Market.unique():
        #     df_region_market = df_region[df_region.Market == market]
        #     plt.scatter(df_region_market.TimeSpei, df_region_market.Price, label=market)

        plt.suptitle("(Inflation-Adjusted) Price Distribution")
        plt.title(f"Region: '{region}'")
        plt.xlabel("Time")
        plt.ylabel(f"Price [{currency}]")

        # store output and make sure it exists
        output_dir = f"../output/{country}/plots"
        if os.path.exists(output_dir) is False:
            os.makedirs(output_dir)
        plt.savefig(f"{output_dir}/{region}-scatter-adj-prices.png")
        # plt.show()

        # Extrapolate
        if interpolation_method == "spline":
            # Extrapolate (nominal) Prices
            df_region_new = df_region.assign(Price=df_region.loc[:, "Price"].interpolate(method="spline", order=2))
            # Extrapolate inflation-adjusted prices
            df_region_new = df_region_new.assign(AdjPrice=df_region_new.loc[:, "AdjPrice"].interpolate(method="spline", order=2))
        else:
            # Extrapolate (nominal) prices
            df_region_new = df_region.assign(Price=df_region.loc[:, "Price"].interpolate(method=interpolation_method))
            # Extrapolate inflation-adjusted prices
            df_region_new = df_region_new.assign(AdjPrice=df_region_new.loc[:, "AdjPrice"].interpolate(method=interpolation_method))

        # Plot post interpolation (/ extrapolated points)
        # plt.scatter(df_region_new.TimeSpei[df_region.Price.isna()], df_region_new.Price[df_region.Price.isna()],
        #             color="green", label="Extrapolated points", marker="*", alpha=alpha)

        plt.scatter(df_region_new.TimeSpei[df_region.Price.isna()], df_region_new.AdjPrice[df_region.Price.isna()],
                    color="green", label="Extrapolated points", marker="*", alpha=alpha)
        plt.legend()
        plt.savefig(f"{output_dir}/{region}-scatter-prices-extrapolated.png")
        plt.show()

        # append extrapolated dataframe to dictionary
        dfs_extrapolated_per_region[region] = df_region_new

    # Merge extrapolated dataframes per region
    df_merged_all_regions = pd.concat(dfs_extrapolated_per_region.values(), ignore_index=True)

    return df_merged_all_regions
