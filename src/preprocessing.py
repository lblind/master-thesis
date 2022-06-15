"""
PREPROCESSING

Everything belong to the preprocessing part
"""

import os
import pandas as pd
import xarray as xr


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
    df_southern = df_southern[df_southern.Commodity.isin(dropped_commodities) is False]
    print(df_southern.Commodity.unique())


    dfs = [df_central, df_northern, df_southern]

    # for i, df in enumerate(dfs):
    #     print(f"#------------------------------------------------------------------------\n"
    #           f"# [{i}]\n"
    #           f"#------------------------------------------------------------------------\n")
    #     print(df)
    #     print(df.columns)
    #     for column in df.columns:
    #         print(df[column].unique())
    #     # print(df["Commodity"].unique())
    #     # print(df["Year"].unique())
    #     # print(df["Market"].unique())

    return dfs



def read_climate_data(path_to_netcdf):
    ds = xr.open_dataset(path_to_netcdf)
    df = ds.to_dataframe()

    return df



