"""
Dynamic Mode Decomposition
--------------------------
Functions that are used to perform the Dynamic Mode Decomposition (DMD).
"""

import pydmd
from matplotlib import pyplot as plt
from pydmd import DMD

import pandas as pd
import os
import numpy as np
import utils


def get_snapshot_matrix_x_for_commodity(df_commodity, time_span_min, time_span_max, write_excel=True):
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

    first_year, first_month_first_year = time_span_min
    last_year, last_month_last_year = time_span_max

    print(df_commodity.Market.unique(), len(df_commodity.Market.unique()))

    # iterate over the given time range of that commodity
    # +1, as python interval right edge = exclusive
    for year in range(first_year, last_year + 1):
        df_commodity_year = df_commodity[df_commodity.Year == year]
        for month in range(1, 12 + 1):
            # skip months for first year
            if year == first_year:
                if month < first_month_first_year:
                    print(f"Skipping month: {month} as first year ({first_year}).")
                    continue
            # skip months last year
            elif year == last_year:
                if month > last_month_last_year:
                    print(f"Breaking month: {month} as last year ({last_year})")
                    break

            # extract all prices for that time
            # vec_prices_commodity_per_month = df_commodity[df_commodity.TimeSpei == time]["Price"]
            vec_prices_commodity_per_month = np.array(df_commodity_year[df_commodity_year.Month == month]["AdjPrice"])


            print(f" [{commodity}] ({year}, {month})", commodity, len(vec_prices_commodity_per_month),
                  vec_prices_commodity_per_month.shape)
            # print(df_commodity_year[df_commodity_year.Month == month]["Market"])
            # print(vec_prices_commodity_per_month)

            # add vector to matrix
            # print(f"{year}, {month}")
            snapshot_matrix_x_for_commodity[f"{year}, {month}"] = vec_prices_commodity_per_month
            list_prices_per_month.append(vec_prices_commodity_per_month)
    # print(f"{commodity}\n{len(list_prices_per_month)}")
    if write_excel:
        dir_output = f"../output/{country}/intermediate-results/DMD"

        if os.path.exists(dir_output) is False:
            os.makedirs(dir_output)

        snapshot_matrix_x_for_commodity.to_excel(f"{dir_output}/X-{commodity}.xlsx")

    return snapshot_matrix_x_for_commodity


def dmd_algorithm(df_snapshots, country, commodity, svd_rank=0, exact=True):
    """

    :param commodity:
    :param df_snapshots:
    :param svd_rank:
        if set to 0, it will be automatically detected
    :param exact:
    :param opt:
    :return:
    """

    output_dir = f"../output/{country}/dmd"
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    # define the operator/ parameters you want to use for the algorithm
    dmd = DMD(svd_rank=svd_rank, exact=exact)

    # convert df to numpy
    np_snapshots = df_snapshots.to_numpy()

    # train the data
    dmd.fit(np_snapshots)

    # extract results (modes + frequency)
    modes = dmd.modes
    frequency = dmd.frequency
    reconstructed_data = dmd.reconstructed_data
    eigs = dmd.eigs

    print(f"Frequency ({frequency.shape})\n", frequency)
    print(f"Eigs ({eigs.shape})\n", eigs)
    print(f"Modes ({modes.shape})\n", modes)

    print(f"Original data ({df_snapshots.shape})")
    print(f"Reconstructed data ({reconstructed_data.shape})\n", reconstructed_data)

    # dmd.save(f"../output/{country}/dmd/{country}-dmd-output-{commodity}.xlsx")

    # plot the results
    # dmd.plot_modes_2D(figsize=(12, 5))

    # show the results
    # plt.show()

    # return the trained dmd object
    return dmd


def dmd_per_commodity(df_final, write_excels=True):
    """

    :param df_final:
    :return:
    """
    country = df_final.Country.unique()[0]
    dict_xs_per_commodity = {}

    # create dir for dmd if not yet existent
    output_dir_dmd = f"../output/{country}/dmd"
    if os.path.exists(output_dir_dmd) is False:
        os.makedirs(output_dir_dmd)

    df_time_spans = utils.convert_excel_to_df(f"../output/{country}/summary-statistics/time-spans-per-commodity.xlsx")

    for commodity in df_final.Commodity.unique():

        # --------------------------------------------------------------------------------------------------------------
        # STEP 1: Arrange data in a way needed (as snapshot matrices)
        # --------------------------------------------------------------------------------------------------------------

        # Extract the part of the dataframe that belongs to that commodity
        # df_final_per_commodity = df_final[df_final.Commodity == commodity]

        df_final_per_commodity = utils.convert_excel_to_df(f"../output/{country}/{country}-final-dta.xlsx",
                                                           sheet_name=commodity)

        df_time_span_commodity = df_time_spans[df_time_spans.Commodity == commodity]

        # read time spans for commodity
        time_span_min = (int(df_time_span_commodity["TimeSpanMinY"]), int(df_time_span_commodity["TimeSpanMinM"]))
        time_span_max = (int(df_time_span_commodity["TimeSpanMaxY"]), int(df_time_span_commodity["TimeSpanMaxM"]))

        # construct the snapshot matrix per commodity
        x_snapshot_matrix = get_snapshot_matrix_x_for_commodity(df_final_per_commodity, time_span_min=time_span_min,
                                                time_span_max=time_span_max, write_excel=write_excels)
        # append snapshot matrix to dict
        dict_xs_per_commodity[commodity] = x_snapshot_matrix

        # --------------------------------------------------------------------------------------------------------------
        # STEP 2: Do the actual DMD
        # --------------------------------------------------------------------------------------------------------------

        dmd_algorithm(x_snapshot_matrix, country=country, commodity=commodity)

    if write_excels:
        # Write all dfs into one excel
        with pd.ExcelWriter(
                f"{output_dir_dmd}/{country}-snapshot-matrices-per-commodity.xlsx") as writer:
            for commodity in df_final.Commodity.unique():
                dict_xs_per_commodity[commodity].to_excel(writer, sheet_name=commodity, na_rep="-")
