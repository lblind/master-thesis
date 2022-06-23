"""
PREPROCESSING

Everything belong to the preprocessing part
"""

import os
import pandas as pd
import xarray as xr
import netCDF4
import numpy as np


def read_and_clean_csvs_food_prices(path_to_dir_csvs="../input/food-price-dta/csv"):
    """

    :param path_to_dir_csvs:
    :return:
    """
    if os.path.exists(path_to_dir_csvs) is False:
        raise ValueError(f"Directory containing csvs <{path_to_dir_csvs}> not found.\n"
                         f"Please review your path definition and make sure the directory exists.")

    central_path_to_csv_all = f"{path_to_dir_csvs}/Central_All_Commodities_WFP_2022Jun14_Malawi_FoodPricesData.csv"
    northern_path_to_csv_all = f"{path_to_dir_csvs}/Northern_All_Commodities_WFP_2022Jun14_Malawi_FoodPricesData.csv"
    southern_path_to_csv_all = f"{path_to_dir_csvs}/Southern_All_Commodities_WFP_2022Jun14_Malawi_FoodPricesData.csv"

    df_central = pd.read_csv(central_path_to_csv_all)
    df_northern = pd.read_csv(northern_path_to_csv_all)
    df_southern = pd.read_csv(southern_path_to_csv_all)

    # southern: omit prices for: Maize (white) - retail, Rice (imported) - retail, Sorghum (red) - retail
    # as no data available for other regions

    print(df_southern.Commodity.unique())
    dropped_commodities = ["Maize (white) - retail", "Rice (imported) - retail", "Sorghum (red) - retail"]
    # df_southern = df_southern[df_southern.Commodity.isin(dropped_commodities) is False]
    print(df_southern.Commodity.unique())

    print(46 + 20 + 57)

    dfs = [df_central, df_northern, df_southern]

    for i, df in enumerate(dfs):
        # print(f"#------------------------------------------------------------------------\n"
        #       f"# [{i}]\n"
        #       f"{df.Market.unique().size}"
        #       f"#------------------------------------------------------------------------\n")
        #     print(df)
        #     print(df.columns)
        #     for column in df.columns:
        #         print(df[column].unique())
        #     # print(df["Commodity"].unique())
        #     # print(df["Year"].unique())
        #     # print(df["Market"].unique())
        pass

    return dfs


def read_climate_data(path_to_netcdf, time_slice, long_slice, lat_slice):
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

    time_column = df_excel["time"]

    df_excel["Year"] = df_excel["time"]
    df_excel["Year"] = df_excel["Year"].apply(lambda x : x.year)

    df_excel["Month"] = df_excel["time"]
    df_excel["Month"] = df_excel["Month"].apply(lambda x: x.month)

    df_excel["Day"] = df_excel["time"]
    df_excel["Day"] = df_excel["Day"].apply(lambda x: x.day)

    df_excel.to_excel("../output/climate_data_preprocessed.xlsx")

    print(df_excel["time"])

    return df_excel

    # TODO: NEXT STEPS
    # 1. Iterate over time
    # ...if middle of month (15)/ only consider that (as only one match value):
    # extract long, lat, spei, year, month
    # add row to dataframe
    # return dataframe

    df_climate = pd.DataFrame()

    # next: loop over remaining time
    # extract spei, long and lat

    # print("\nKeys\n")
    # print(ds.keys())
    #
    # data = netCDF4.Dataset(path_to_netcdf)
    #
    # # Extract variables
    # print("Variables")
    # columns = ["long", "lat", "spei", "time",]
    # # print(data.variables["lon"])
    #
    # print("Dimensions", ds.dims)
    # print("Variables", ds.variables)
    #

    #
    # print(f"LONG\n{long}, {len(long)}")
    # print(f"LAT\n{lat}, {len(lat)}")
    # print(f"Time\n{time}")
    #
    # # make df
    # # df_new = pd.DataFrame(data=[long, lat, spei], index=time, columns=columns[:-1])
    # # print("\nDFNEW\n", df_new.columns, "\n", df_new)
    #
    # path_logs_dir = "../input/climate-dta/logs"
    #
    # if os.path.exists(path_logs_dir) is False:
    #     os.makedirs(path_logs_dir)
    #
    # # store head of cdf as excel
    # # ds.head().to_dataframe().to_excel(f"{path_logs_dir}/header_cdf.xlsx")
    #
    # df = ds.to_dataframe()
    #
    # # drop values with no spei available
    # df=df.dropna()
    #
    # # drop everything that doesn't match the WFP climate base (-2020)
    #
    #
    # print(df)
    # print(df.shape)
    # # print(df.lon)
    #
    # # df.iloc[:50, :].to_excel("../input/climate-dta/subsection_spei.xlsx")
    #
    #
    #
    # #
    # # print(df.columns)
    # # print(df)
    # #
    # # print(f"------------------------------\n"
    # #       f"After reindexing\n"
    # #       f"------------------------------\n")
    # #
    # # # REINDEX dataframe
    # # df.reindex(columns=columns)
    # # print(df.columns)
    # # print(df)
    #
    # # print first column
    # # print(df.iloc[:, 0])
    #
    # # print first row
    # # print(df.iloc[0])
    #
    # return df



