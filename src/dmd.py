"""
Dynamic Mode Decomposition
--------------------------
Functions that are used to perform the Dynamic Mode Decomposition (DMD).
"""


import pydmd
from matplotlib import pyplot as plt
from pydmd import DMD
from pydmd import MrDMD

import pandas as pd
import os
import numpy as np
import utils
import visualization


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


def save_dmd_results(dmd, country, commodity, excel_output_extension=""):
    """

    :param dmd:
    :return:
    """
    print(f"Saving results of DMD...")
    results_dict = {}
    results_dict["eigs"] = pd.DataFrame(dmd.eigs)
    results_dict["reconstructed_data"] = pd.DataFrame(dmd.reconstructed_data)
    results_dict["modes"] = pd.DataFrame(dmd.modes)
    results_dict["frequency"] = pd.DataFrame(dmd.frequency)
    results_dict["dynamics"] = pd.DataFrame(dmd.dynamics)

    results_dict["svd_rank"] = pd.DataFrame([dmd.svd_rank])
    results_dict["growth_rate"] = pd.DataFrame([dmd.growth_rate])
    results_dict["snapshots"] = pd.DataFrame(dmd.snapshots)
    results_dict["amplitudes"] = pd.DataFrame(dmd.amplitudes)


    print(f"Frequency ({dmd.frequency.shape})\n", dmd.frequency)
    print(f"Eigs ({dmd.eigs.shape})\n", dmd.eigs)
    print(f"Modes ({dmd.modes.shape})\n", dmd.modes)

    print(f"Original data ({dmd.snapshots.shape})")
    print(f"Reconstructed data ({dmd.snapshots.shape})\n", dmd.reconstructed_data)

    # make sure that directory exists
    output_path = f"../output/{country}/dmd/{commodity}"
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    # Write all dfs into one excel
    with pd.ExcelWriter(f"{output_path}/dmd-results{excel_output_extension}.xlsx") as writer:
        for group in results_dict.keys():
            # extract relevant subgroup
            df_sum_stat = results_dict[group]
            df_sum_stat.to_excel(writer, sheet_name=group)

    print(f"Saving results of DMD successful.")


def dmd_algorithm(df_snapshots, country, commodity, svd_rank=0, exact=True, mr_dmd=False):
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

    if mr_dmd:
        dmd = MrDMD(dmd)

    # convert df to numpy
    np_snapshots = df_snapshots.to_numpy()

    # train the data
    dmd.fit(np_snapshots)


    # save the dmd outputs as excels
    save_dmd_results(dmd, country, commodity)

    # return the trained dmd object
    return dmd


def dmd_per_commodity(df_final, write_excels=True, svd_rank=1, mr_dmd=False, transpose=True):
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

        # svd_rank = 2
        if transpose:
            dmd = dmd_algorithm(x_snapshot_matrix.T, country=country, commodity=commodity, svd_rank=svd_rank, mr_dmd=mr_dmd)
            png_appendix = "T"
        else:
            dmd = dmd_algorithm(x_snapshot_matrix, country=country, commodity=commodity, svd_rank=svd_rank, mr_dmd=mr_dmd)
            png_appendix = ""


        # visualize what you have found
        if mr_dmd:
            visualization.plot_dmd_results(dmd, country, algorithm="mrDMD", transposed=transpose)
        else:
            visualization.plot_dmd_results(dmd, country, algorithm="base", transposed=transpose)

    if write_excels:
        # Write all dfs into one excel
        with pd.ExcelWriter(
                f"{output_dir_dmd}/{country}-snapshot-matrices-per-commodity.xlsx") as writer:
            for commodity in df_final.Commodity.unique():
                dict_xs_per_commodity[commodity].to_excel(writer, sheet_name=commodity, na_rep="-")
