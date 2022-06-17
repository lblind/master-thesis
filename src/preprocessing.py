"""
PREPROCESSING

Everything belong to the preprocessing part
"""

import os
import pandas as pd
import xarray as xr
import netCDF4


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


def read_climate_data(path_to_netcdf):
    """
    Reads the netcdf and converts it into a pandas df

    :param path_to_netcdf:
    :return:
    """
    ds = xr.open_dataset(path_to_netcdf)
    print("\nKeys\n")
    print(ds.keys())

    data = netCDF4.Dataset(path_to_netcdf)

    # Extract variables
    print("Variables")
    columns = ["long", "lat", "spei", "time",]
    # print(data.variables["lon"])

    print("Dimensions", ds.dims)
    print("Variables", ds.variables)

    # extract the data per column
    long = data.variables["lon"][:]
    lat = data.variables["lat"][:]
    time = data.variables["time"][:]
    spei = data.variables["spei"][:]

    print(f"LONG\n{long}, {len(long)}")
    print(f"LAT\n{lat}, {len(lat)}")

    # make df
    # df_new = pd.DataFrame(data=[long, lat, spei], index=time, columns=columns[:-1])
    # print("\nDFNEW\n", df_new.columns, "\n", df_new)

    path_logs_dir = "../input/climate-dta/logs"

    if os.path.exists(path_logs_dir) is False:
        os.makedirs(path_logs_dir)

    # store head of cdf as excel
    # ds.head().to_dataframe().to_excel(f"{path_logs_dir}/header_cdf.xlsx")

    df = ds.to_dataframe()

    # drop values with no spei available
    df=df.dropna()

    # drop everything that doesn't match the WFP climate base (-2020)


    print(df)
    print(df.shape)
    # print(df.lon)

    # df.iloc[:50, :].to_excel("../input/climate-dta/subsection_spei.xlsx")



    #
    # print(df.columns)
    # print(df)
    #
    # print(f"------------------------------\n"
    #       f"After reindexing\n"
    #       f"------------------------------\n")
    #
    # # REINDEX dataframe
    # df.reindex(columns=columns)
    # print(df.columns)
    # print(df)

    # print first column
    # print(df.iloc[:, 0])

    # print first row
    # print(df.iloc[0])

    return df



