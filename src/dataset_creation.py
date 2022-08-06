"""
CREATION OF DATASET
-------------------

"""
import warnings

import pandas as pd

import preprocessing as preproc
import analysis as stats
import visualization


# cut off markets: 0.5, cut off commodities: 0.8
# Focus commodities: cut off: 0.4, 0.8 -> still to high

# overall missings below 30%: cut_off_markets = 0.4, cut_off_commodities = 0.75
# a bit better: cut_off_marekts = 0.35, cut_off_commodities = 0.75 (17 Markets)
# solves the issue: 0.75 (commodities), 0.3 (markets) (10 Markets)

# TODO: different cut offs per commodity for markets (and compute share of missings PER COMMODITY)
# TODO: Maybe loop over create dataset function and extract only subset per commodity
def phase_a_preprocess_wfp_dataset(country, dropped_commodities,
                                   add_pad_months_time_span=0,
                                   cut_off_commodities=0.6,
                                   cut_off_markets=0.5,
                                   limit_consec_interpol=14
                                   ):
    """
    Read raw WFP database and do the following steps:

    - STEP 1 -> Download raw WFP data (per region & merge into one df)
    - STEP 2 -> Merge Inflation data + Adjust food prices to one common inflation level (most recent)
    - STEP 3 -> Reduce % missings -> Subset Time: Extract relevant time slice per commodity
                                                (first non-nan entry -> last non-nan entry)
    - STEP 4 -> Reduce % missings -> Subset Commodity: Cut off commodities with certain share of missings
    - STEP 5 -> Reduce % missings -> Subset Market:
    - STEP 6 -> Inter-/Extrapolate missing data

    :param country:
    :param dropped_commodities:
    :param add_pad_months_time_span:
    :param cut_off_commodities:
    :param cut_off_markets:
    :param limit_consec_interpol:
    :return:
    """
    preproc_step = 0
    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 1 (Read WFP dataset (merged per region)"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    preproc_step += 1
    df_wfp = preproc.get_df_wfp_preprocessed_excel_region_method(country=country,
                                                                 dropped_commodities=dropped_commodities)
    # write the raw output to an Excel workbook
    df_wfp.to_excel(f"../output/{country}/intermediate-results/df_wfp_STEP{preproc_step}.xlsx")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 2 (Adjust prices to common food inflation level -> most recent)"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    preproc_step += 1
    # Adjust food prices to one common price level
    df_wfp = preproc.adjust_food_prices(country=country, df_wfp=df_wfp, data_source_inflation="WFP")

    # write the raw output to an Excel workbook
    df_wfp.to_excel(f"../output/{country}/intermediate-results/df_wfp_STEP{preproc_step}.xlsx")

    # Write sum stats (result Step 2/1)
    stats.sum_stats_prices(df=df_wfp, excel_output_extension="-preproc-STEP1-2-df_wfp")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 3 (Missings -> Subset of Time: Extract relevant slice per commodity)"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    preproc_step += 1

    # extract only the relevant subsets of time
    df_wfp = preproc.extract_df_subset_time_prices_all_commodities(df=df_wfp,
                                                                   add_pad_months_time_span=add_pad_months_time_span)

    # write the raw output to an Excel workbook
    df_wfp.to_excel(f"../output/{country}/intermediate-results/df_wfp_STEP{preproc_step}.xlsx")

    # TODO: additional subset time component restricting time to smallest time overlap per region? (as extrapolated in
    # regional patterns?) -> alternatively: just extrapolate over dataset, regardless of regional patterns

    # write summary statistics (share of missing values within commodity dataset)
    # Check missings (result of step 3)
    df_commodity_stats = stats.sum_stats_prices(df=df_wfp, return_df_by_group_sheet="Commodity",
                                                excel_output_extension=f"-preproc-STEP{preproc_step}-subset-time")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 4 (Cut extreme outliers (manually) -> Rice)"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    # Visualization of price distribution before cut offs
    visualization.boxplot_adj_prices(df_wfp, png_appendix=f"-preproc-STEP{preproc_step}-before-cut")

    preproc_step += 1

    # replace extreme outliers with nan (only 1 market)
    df_wfp = preproc.replace_extreme_outliers_with_nan(df_wfp)

    # write the raw output to an Excel workbook
    df_wfp.to_excel(f"../output/{country}/intermediate-results/df_wfp_STEP{preproc_step}.xlsx")

    df_commodity_stats = stats.sum_stats_prices(df=df_wfp, return_df_by_group_sheet="Commodity",
                                                excel_output_extension=f"-preproc-STEP{preproc_step}-cut-extreme-outliers")

    # Visualization of price distribution before cut offs
    visualization.boxplot_adj_prices(df_wfp, png_appendix=f"-preproc-STEP{preproc_step}-after-cut")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 5 (Missings -> Subset of Commodity: Drop too sparse commodities)"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    preproc_step += 1

    # Drop all commodities that have missing data > certain cut-off (too sparse)

    df_wfp = preproc.cut_too_sparse_values_in_column(df=df_wfp,
                                                     column="Commodity",
                                                     df_sum_stats_column=df_commodity_stats,
                                                     preproc_step_no=preproc_step,
                                                     cut_off=cut_off_commodities
                                                     )

    # write summary statistics (share of missing values within market dataset)
    # check missings (result of step 5)
    df_market_stats = stats.sum_stats_prices(df=df_wfp, return_df_by_group_sheet="Market",
                                             excel_output_extension=f"-preproc-STEP{preproc_step}-subset-commodity")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 6 (Missings -> Subset of Market: Drop too sparse markets)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    preproc_step += 1
    # Drop all markets that have missing data > certain cut-off (too sparse)

    # print(excel_path_stats)
    df_wfp = preproc.cut_too_sparse_values_in_column(df=df_wfp,
                                                     column="Market",
                                                     df_sum_stats_column=df_market_stats,
                                                     preproc_step_no=preproc_step,
                                                     cut_off=cut_off_markets,
                                                     )

    # Write sum stats (result Step 5)
    stats.sum_stats_prices(df=df_wfp, excel_output_extension=f"-preproc-STEP{preproc_step}-subset-market")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 7 - Per commodity: (Interpolation of missing data)"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    preproc_step += 1

    # Define inter-/ extrapolation method
    interpolation_method = "linear"
    # interpolation_method = "cubicspline"
    # interpolation_method = "quadratic"
    order = None
    interpolation_method = "spline"
    # order = 2
    order = 1
    # extrapolate?
    extrapolate = True

    # Per Commodity: EXTRAPOLATE REGIONAL PATTERNS
    # doesn't extrapolate tails for interpolation method: cubic
    df_wfp = preproc.extrapolate_prices_per_commodity_regional_patterns(df_wfp=df_wfp,
                                                                        interpolation_method=interpolation_method,
                                                                        intrapolation_limit=limit_consec_interpol,
                                                                        order=order,
                                                                        extrapolate=extrapolate)

    # TODO In case the implementtion changes, currently already an error is thrown
    # in the interpolation method if that is the case
    if df_wfp.Price.isna().sum() != 0:
        warnings.warn(f"Number of prices should be 0 after extrapolation, but is: {df_wfp.Price.isna().sum()}")

    # write final sum stats
    stats.sum_stats_prices(df=df_wfp, excel_output_extension=f"-preproc-STEP{preproc_step}-interpolation-PHASE-A")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: COMPLETED (WFP dataset preprocessed -> no missings anymore)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    return df_wfp


def phase_b_merge_wfp_with_spei_dataset(country, df_wfp_preproc, write_results_to_excel=True):
    """
    Merge the reduced WFP Price dataset with
    - the dataset containing the market coordinates
    - the SPEI dataset (based on nearest neighbor for each market)

    And classify the different SPEI categories

    Concrete procedure:
    STEP 8  -> Merge WFP Prices with WFP Market coordinates
    STEP 9  -> Extract relevant subset of SPEI dataset (range time, lon, lat)
    STEP 10 -> Merge climate data SPEI with market coordinates (nearest neighbour)
    STEP 11 -> Subset of time: drop years for which there is no SPEI data (2021, 2022)
    STEP 12 -> Classification of droughts and derivation of SPEI categories


    :param country:
    :param df_wfp_preproc:
    :return:
    """
    # preproc steps of phase a
    preproc_step = 7

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE B: STEP 8 - Merge df_wfp with market coordinates dataset"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    preproc_step += 1
    # 2. Read CSV containing market coordinates and merge to price data
    df_wfp_with_coords = preproc.read_and_merge_wfp_market_coords(df_wfp=df_wfp_preproc, country=country)
    # 3. Preparation for merge with Part B): Extract range of 3 main variables: time, longitude, latitude

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE B: STEP 9 - Extraction of range of variables for: time, lon, lat & read relevant slice"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    preproc_step += 1

    slice_time, slice_lon, slice_lat = preproc.extract_time_lon_lat_slice(df_wfp_with_coords)

    # read relevant subset of entire (global) SPEI database
    df_spei = preproc.read_climate_data(time_slice=slice_time, long_slice=slice_lon, lat_slice=slice_lat,
                                        country=country)

    visualization.line_plot_spei(df_spei=df_spei, country=country, show=True)
    visualization.line_plot_mean_min_max_spei_per_time(df_spei=df_spei, country=country, show=True)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE B: STEP 10 - Merge climate data (SPEI) with market coordinates"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    preproc_step += 1

    df_wfp_and_spei = preproc.merge_food_price_and_climate_dfs(df_wfp_with_coords=df_wfp_with_coords, df_spei=df_spei)

    stats.sum_stats_prices(df=df_wfp_and_spei,
                           excel_output_extension=f"-preproc-STEP{preproc_step}-Merged-SPEI")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE B: STEP 11 - Subset Time: Drop years for which there is no SPEI data (2021, 2022)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    preproc_step += 1

    # TODO maybe do this smarter and check already in merge function above if max SPEI date < max WFP date -> omit
    # everything above that (or: maybe do an inner join instead of a left outer join instead?)
    # drop 2021, 2022 as no data for droughts (/SPEI) is available for those (even though WFP data is)
    years_to_drop = [2021, 2022]
    print(f"# Dropping years (as no SPEI data available): {years_to_drop}")
    df_wfp_and_spei = preproc.drop_years(df=df_wfp_and_spei, years_list=years_to_drop)

    stats.sum_stats_prices(df=df_wfp_and_spei,
                           excel_output_extension=f"-preproc-STEP{preproc_step}-Merged-no-2021-22")

    visualization.line_plot_spei_per_region(df_wfp_and_spei=df_wfp_and_spei, show=True)

    visualization.line_plot_mean_min_max_spei_per_time(df_spei=df_wfp_and_spei, time="TimeWFP", country=country, show=True)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE B: STEP 12 - Classification of droughts and into SPEI categories"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    preproc_step += 1

    # add SPEI categories
    df_final = preproc.classify_droughts(df_wfp_and_spei)

    # Plot Histogram for SPEI categories
    visualization.plot_hist(df_final, "SpeiCat", orientation="horizontal", bins=7,
                            png_appendix=f"-preproc-STEP{preproc_step}")

    # Summary statistics categorical
    counts = df_final["SpeiCat"].value_counts()
    print(f"Counts:\n{counts}")

    counts_drought = df_final["Drought"].value_counts()
    print(f"Counts Drought:\n{counts_drought}")
    print(f"Share of droughts: {counts_drought[True]}/ {(counts_drought[True] + counts_drought[False])}")

    stats.sum_stats_prices_and_droughts(df=df_wfp_and_spei,
                                        excel_output_extension=f"-preproc-STEP{preproc_step}-FINAL-DATASET")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE B: COMPLETED (WFP dataset merged with SPEI dataset + Drought classification)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - WRITING RESULTS TO EXCEL."
          "\n# ------------------------------------------------------------------------------------------------------\n")
    if write_results_to_excel:
        dict_df_final_per_commodity = {}
        for commodity in df_final.Commodity.unique():
            df_final_commodity = df_final[df_final.Commodity == commodity]
            dict_df_final_per_commodity[commodity] = df_final_commodity
        df_drought, df_no_drought = preproc.separate_df_drought_non_drought(df_final)

        preproc.write_preprocessing_results_to_excel(df_wfp=df_wfp_preproc, df_wfp_with_coords=df_wfp_with_coords,
                                                     df_spei=df_spei, dict_df_final_per_commodity=
                                                     dict_df_final_per_commodity,
                                                     df_drought=df_drought,
                                                     df_no_drought=df_no_drought)

    return df_final


def create_dataset(country, dropped_commodities, write_results_to_excel=True):
    """
    Creates the dataset in two phases
    PHASE A: Take care of WFP dataset and reduce the share of missing data
    PHASE B: Merge WFP dataset with Market coordinates + SPEI dataset.

    :param country:
    :param dropped_commodities:
    :return:
    """
    df_wfp_preproc = phase_a_preprocess_wfp_dataset(country=country,
                                                    dropped_commodities=dropped_commodities)
    df_final = phase_b_merge_wfp_with_spei_dataset(country=country, df_wfp_preproc=df_wfp_preproc,
                                                   write_results_to_excel=write_results_to_excel)

    return df_final
