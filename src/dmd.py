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


def dmd_own_implementation(S):
    """
    Own implementation of Dynamic Mode Decomposition (DMD)
    This is just an auxiliary function to check the validity
    of the PyDMD implementation.

    :param S: pandas.DataFrame
        Snapshot matrix
    :return: W, V: panda.DataFrame
        DMD eigenvalues and DMD modes
    """
    X = S.to_numpy()[:, :-1]
    X_prime = S.to_numpy()[:, 1:]

    X_pinv = np.linalg.pinv(X)
    A = X_prime.dot(X_pinv)

    # Compute eigendecomposition
    # W = eigenvalues
    # V = Eigenvectors
    W, V = np.linalg.eig(A)

    W_matrix = np.diag(W)

    x = np.linspace(1, S.shape[0], S.shape[0])

    reconstructed_data = V @ W_matrix @ V.T
    reconstructed_data = np.empty(S.shape)
    x_k = S[:, 0]
    reconstructed_data[:, 0] = x_k
    for k in range(1, S.shape[1]):
        reconstructed_data[:, k] = A.dot(x_k)
        x_k = reconstructed_data[:, k]

    plt.imshow(reconstructed_data.real)
    # plt.plot(x, V[:, 0])
    plt.title("Own implementation first DMD mode")
    plt.show()

    return W, V


def get_snapshot_matrix_x_for_commodity(df_commodity, time_span_min, time_span_max, write_excel=True):
    """
    Arranges the final dataset in a way that it fits the Dynamic Mode Decomposition (DMD), i.e.:
    X:
    - one vector per delta t (i.e. month) containing the prices for all investigated markets
    - put these vectors alongside each other

    X':
    - shift X one delta t (i.e. month)

    :param write_excel: boolean
        Whether to write the results to an Excel sheet or not
    :param df_commodity: pandas.DataFrame
        Dataset containing only the data for one commodity
    :param time_span_max: datetime
        Date of the last entry for this commodity
    :param time_span_min:  datetime
        Date of the first entry for this commodity

    :return: S: pandas.DataFrame
        Snapshot matrix S
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


def compute_abs_error(dmd, country, commodity, rank, algorithm, transposed=False):
    """
    Computes the absolute error resulting out of the reconstruction
    and plots the corresponding error matrix

    :param dmd: pyDMD baseobject
        DMD object containing the set configuration
    :param country: str
        Country for which the DMD has been computed
    :param commodity: str
        Commodity for which the DMD has been computed
    :param rank: int
        Rank that has been set for the reconstruction
    :param algorithm: str
        Algorithm that has been used for the PyDMD DMD implementation (Default: base)
    :param transposed: boolean
        Whether or not the snapshot matrix has been transposed or not

    :return:
    """
    # create dir for dmd if not yet existent
    output_dir_dmd = f"../output/{country}/dmd"
    if os.path.exists(output_dir_dmd) is False:
        os.makedirs(output_dir_dmd)
    abs_error = np.abs(dmd.snapshots - dmd.reconstructed_data.real)

    visualization.plot_abs_error_matrix(abs_error=abs_error, country=country, rank=rank,
                                        algorithm=algorithm, transposed=transposed, commodity=commodity)

    return abs_error


def save_dmd_results(dmd, country, commodity, excel_output_extension="", transposed=False, algorithm="base", rank=0):
    """
    Function that saves the results of the trained PyDMD dmd object

    :param rank: int
        Which rank should be set in the SVD which underlies the DMD
    :param algorithm: str
        Algorithm/ type of object that should be used for the PyDMD implementation
    :param excel_output_extension: str
        Postfix that should be appended to the regular excel name
    :param commodity: str
        Name of the commodity for which the DMD has been computed
    :param dmd: PyDMD object
        Trained PyDMD object
    :param country: str
        Country for which the DMD has been computed
    :param transposed: boolean
        Whether or not the Snapshot matrix has been transposed

    :return: Nothing
        The extracted results in an Excel workbook which will be stored in the output folder.
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

    abs_error = compute_abs_error(dmd=dmd, country=country, commodity=commodity,
                                  rank=dmd.svd_rank, algorithm=algorithm,
                                  transposed=transposed)

    stats_abs_error = pd.DataFrame({
        "Mean": [abs_error.flatten().mean()],
        "Min": [abs_error.flatten().min()],
        "Median": [np.median(abs_error.flatten())],
        "Max": [abs_error.flatten().max()],
        "Std. dev": [abs_error.flatten().std()],
    })

    results_dict["abs_error_df"] = pd.DataFrame(abs_error)
    results_dict["abs_error_stat"] = stats_abs_error

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
    with pd.ExcelWriter(f"{output_path}/dmd-results{excel_output_extension}-r{rank}-T-{transposed}.xlsx") as writer:
        for group in results_dict.keys():
            # extract relevant subgroup
            df_sum_stat = results_dict[group]
            df_sum_stat.to_excel(writer, sheet_name=group)

    print(f"Saving results of DMD successful.")


def dmd_algorithm(df_snapshots, country, commodity, svd_rank=0, exact=True, mr_dmd=False, transposed=False):
    """
    Actual DMD algorithm containing all the steps required in the procedure
    1. Convert the snapshots to numpy
    2. Fit the PyDMD object based on those
    3. Store the results

    :param transposed: boolean
        Whether or not the snapshot matrix has been tranposed
    :param mr_dmd: boolean
        Whether or not the Multiresolution DMD should be computed (DEfault False -> Base DMD)
    :param country: str
        Country for which the DMD has been computed
    :param commodity: str
        Commodity for which the DMD has been computed
    :param df_snapshots: pandas.DataFrame
        Snapshot matrix as a dataframe
    :param svd_rank: int
        Rank of the SVD underlying the DMD
        if set to 0, it will be automatically detected
    :param exact: boolean
        Whether the DMD modes should be computed exactly (True) or projected (False)

    :return: pyDMD
        The trained PyDMD object
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
    save_dmd_results(dmd, country, commodity, rank=svd_rank, transposed=transposed)

    # return the trained dmd object
    return dmd


def dmd_per_commodity(df_final, write_excels=True, svd_rank=0.95, mr_dmd=False, transpose=True,
                      own_implementation=False, exact_modes=True):
    """
    Calls the DMD procedure for each unique commodity detected in the final dataset

    :param exact_modes: boolean
        Whether the exact (True) or projected (False) DMD modes should be computed
    :param own_implementation: boolean
        Whether my own (True) or the PyDMD (False) implementation shoould be used
    :param transpose: boolean
        Whether or not the snapshot matrix should be tranposed
    :param mr_dmd: boolean
        Whether or not the Multiresolution DMD should (True) or should not (False) be used (False -> Base DMD)
    :param svd_rank: int
        Rank of the SVD underlying the DMD
    :param write_excels: boolean
        Whether or not the results should be stored in Excel workbooks
    :param df_final: pandas.DataFrame
        Final preprocessed Dataframe

    :return: Nothing
        All results will be stored in Excel workbooks if write_excels is set to True
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

        if own_implementation:
            dmd_own_implementation(x_snapshot_matrix)
        else:
            # svd_rank = 2
            if transpose:
                dmd = dmd_algorithm(x_snapshot_matrix.T, country=country, commodity=commodity, svd_rank=svd_rank,
                                    mr_dmd=mr_dmd, transposed=transpose, exact=exact_modes)
                png_appendix = "T"
            else:
                dmd = dmd_algorithm(x_snapshot_matrix, country=country, commodity=commodity, svd_rank=svd_rank,
                                    mr_dmd=mr_dmd, transposed=transpose, exact=exact_modes)

            # visualize what you have found
            if mr_dmd:
                visualization.plot_dmd_results(dmd, country, algorithm="mrDMD", transposed=transpose,
                                               commodity=commodity, svd_rank=svd_rank)
            else:
                visualization.plot_dmd_results(dmd, country, algorithm="base", transposed=transpose,
                                               commodity=commodity, svd_rank=svd_rank)

    if write_excels:
        # Write all dfs into one excel
        with pd.ExcelWriter(
                f"{output_dir_dmd}/{country}-snapshot-matrices-per-commodity.xlsx") as writer:
            for commodity in df_final.Commodity.unique():
                dict_xs_per_commodity[commodity].to_excel(writer, sheet_name=commodity, na_rep="-")


