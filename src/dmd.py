"""
Dynamic Mode Decomposition
"""

import pydmd
from pydmd import DMD

import pandas as pd
import os
import numpy as np


def get_snapshot_matrix_x_for_commodity(df_commodity, write_excel=True):
    """
    Arranges the final dataset in a way that it fits the Dynamic Mode Decomposition (DMD), i.e.:
    X:
    - one vector per delta t (i.e. month) containing the prices for all investigated markets
    - put these vectors alongside each other

    X':
    - shift X one delta t (i.e. month)

    :param df_final:
    :return:
    """
    country = df_commodity.Country.unique()[0]
    commodity = df_commodity.Commodity.unique()[0]

    list_prices_per_month = []
    snapshot_matrix_x_for_commodity = pd.DataFrame()

    # iterate over all dates
    for year in df_commodity.Year.unique():
        df_commodity_year = df_commodity[df_commodity.Year == year]
        for month in df_commodity.Month.unique():
            # extract all prices for that time
            # vec_prices_commodity_per_month = df_commodity[df_commodity.TimeSpei == time]["Price"]
            vec_prices_commodity_per_month = np.array(df_commodity_year[df_commodity_year.Month == month]["Price"])

            print(commodity, len(vec_prices_commodity_per_month))

            # add vector to matrix
            snapshot_matrix_x_for_commodity[f"{year}, {month}"] = vec_prices_commodity_per_month
            list_prices_per_month.append(vec_prices_commodity_per_month)
    print(f"{commodity}\n{len(list_prices_per_month)}")
    if write_excel:
        dir_output = f"../output/{country}/intermediate-results/DMD"

        if os.path.exists(dir_output) is False:
            os.makedirs(dir_output)

        snapshot_matrix_x_for_commodity.to_excel(f"{dir_output}/X-{commodity}.xlsx")

    return snapshot_matrix_x_for_commodity


def dmd_per_commodity(df_final, write_excels=True):
    """

    :param df_final:
    :return:
    """
    country = df_final.Country.unique()[0]
    list_xs_per_commodity = []
    for commodity in df_final.Commodity.unique():
        # Extract the part of the dataframe that belongs to that commodity
        df_final_per_commodity = df_final[df_final.Commodity == commodity]

        if write_excels:
            output_dir = f"../output/{country}/intermediate-results/commodities"
            if os.path.exists(output_dir) is False:
                os.makedirs(output_dir)
            df_final_per_commodity.to_excel(f"{output_dir}/{commodity}-df_final.xlsx")

        # print(df_final_per_commodity)

        x = get_snapshot_matrix_x_for_commodity(df_final_per_commodity, write_excel=write_excels)



