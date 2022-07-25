"""
CREATION OF DATASET
-------------------

"""
import pandas as pd

import preprocessing as preproc
import statistics_snippets as stats
import visualization


def phase_a_preprocess_wfp_dataset(country, dropped_commodities,
                                   extrapolated_months=0,
                                   cut_off_commodities=0.9,
                                   cut_off_markets=0.9
                                   ):
    """
    Read raw WFP database and do the following steps:

    - STEP 1: Download raw WFP data (per region & merge into one df)
    - STEP 2: Merge Inflation data + Adjust food prices to one common inflation level (most recent)
    - STEP 3: Reduce % missings -> Subset Time: Extract relevant time slice per commodity
                                                (first non-nan entry -> last non-nan entry)
    - STEP 4: Reduce % missings -> Subset Commodity: Cut off commodities with certain share of missings
    - STEP 5: Reduce % missings -> Subset Market:
    - STEP 6: Inter-/Extrapolate missing data

    :param country:
    :param dropped_commodities:
    :param extrapolated_months:
    :param cut_off_commodities:
    :param cut_off_markets:
    :return:
    """
    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 1 (Read WFP dataset (merged per region)"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    df_wfp = preproc.get_df_wfp_preprocessed_excel_region_method(country=country,
                                                                 dropped_commodities=dropped_commodities)
    # write the raw output to an Excel workbook
    df_wfp.to_excel(f"../output/{country}/intermediate-results/df_wfp.xlsx")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 2 (Adjust prices to common food inflation level -> most recent)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # Adjust food prices to one common price level
    df_wfp = preproc.adjust_food_prices(country=country, df_wfp=df_wfp, data_source_inflation="WFP")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 3 (Missings -> Subset of Time: Extract relevant slice per commodity)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # extract only the relevant subsets of time
    eps_extrapolate_months = 0
    df_wfp = preproc.extract_df_subset_time_prices_all_commodities(df=df_wfp,
                                                                   epsilon_month_extrapolation=eps_extrapolate_months)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 4 (Missings -> Subset of Commodity: Drop too sparse commodities)"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    # write summary statistics (share of missing values within commodity dataset)
    # Check missings
    df_commodity_stats = stats.sum_stats_prices(df=df_wfp, return_df_by_group_sheet="Commodity",
                                                excel_output_extension="-preproc-STEP4")

    # drop commodities that are too sparse
    cut_off_percent_commodity = 90
    excel_path_stats = f"../output/{country}/summary-statistics/{country}-sum-stats-preproc-STEP4.xlsx"
    df_wfp = preproc.drop_commodities_too_sparse(df=df_wfp,
                                                 df_sum_stats_commodities=df_commodity_stats,
                                                 cut_off_percent=cut_off_percent_commodity,
                                                 excel_to_append_dropped_commodities=excel_path_stats,
                                                 preproc_step_no=4
                                                 )

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 5 (Missings -> Subset of Market: Drop too sparse markets)"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    # write summary statistics (share of missing values within market dataset)
    # check missings
    df_market_stats = stats.sum_stats_prices(df=df_wfp, return_df_by_group_sheet="Market",
                                             excel_output_extension="-preproc-STEP5")

    # Drop all markets that have missing data > certain cut-off
    cut_off_percent_market = 90

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC - PHASE A: STEP 6 (Interpolation of missing data)"
          "\n# ------------------------------------------------------------------------------------------------------\n")


def phase_b_merge_wfp_with_spei_dataset(df_wfp):
    """

    :param df_wfp:
    :return:
    """
    pass


def create_dataset(country, dropped_commodities):
    """

    :param country:
    :param dropped_commodities:
    :return:
    """
    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: PHASE 1 (MERGE DATA)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # PART A) Get food price part of data
    # -----------------------------------

    # 1. Read food prices, convert to excel and return as df
    # Country method: Doesn't propagate missing values for all combinations of markets, time, commodity
    # (doesn't write them in the dataset)
    # df_wfp = preproc.get_df_wfp_preprocessed_excel_country_method(country=country,
    #                                                              dropped_commodities=dropped_commodities)

    # Region method: Does propagate missing values
    # propagates them for all combinations of time, commodity and markets and writes them to the dataset
    df_wfp = preproc.get_df_wfp_preprocessed_excel_region_method(country=country,
                                                                 dropped_commodities=dropped_commodities)
    df_wfp.to_excel(f"../output/{country}/intermediate-results/df_wfp.xlsx")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Check that all combinations are filled with nans (if no data given)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # check validity of extracted dataset (check that for all combinations of markets, commodity & time
    # food prices have been populated)
    # df_wfp = preproc.check_markets_per_commodity_time(df_wfp=df_wfp)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Adjust food prices to inflation (last recent (food) inflation level)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # Adjust food prices to one common price level
    df_wfp = preproc.adjust_food_prices(country=country, df_wfp=df_wfp, data_source_inflation="WFP")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Add market lon, lats to dataset"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # 2. Read CSV containing market coordinates and merge to price data
    df_wfp_with_coords = preproc.read_and_merge_wfp_market_coords(df_wfp=df_wfp, country=country)
    # 3. Preparation for merge with Part B): Extract range of 3 main variables: time, longitude, latitude
    slice_time, slice_lon, slice_lat = preproc.extract_time_lon_lat_slice(df_wfp_with_coords)

    print(f"Slice Time: {slice_time}\nSlice Lon: {slice_lon}\nSlice Lat: {slice_lat}")

    # PART B) Get climate part of data (SPEI)
    # ---------------------------------------

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Read climate data (SPEI)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # STEP 4) read and merge climate data & only keep non-nan values / omit those where no measurement has been possible
    df_spei = preproc.read_climate_data(time_slice=slice_time, long_slice=slice_lon, lat_slice=slice_lat,
                                        country=country)

    # PART C) Merge Outcomes of Part A) and Part C) and store them as excel
    # ---------------------------------------------------------------------
    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Merge climate data to food data"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    df_final = preproc.merge_food_price_and_climate_dfs(df_wfp_with_coords=df_wfp_with_coords, df_spei=df_spei)

    # Check that everything went okay in preprocessing/ merge part
    if df_final.shape[0] != df_wfp.shape[0]:
        raise ValueError(f"Something went wrong in the preprocessing part.\n"
                         f"# rows of final/ merged df {df_final.shape[0]} should"
                         f" be the same as for the one of wfp {df_wfp.shape[0]}")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Drop years where no SPEI data is available (2021, 2022) -> even though WFP data is"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    # TODO maybe do this smarter and check already in merge function above if max SPEI date < max WFP date -> omit
    # everything above that (or: maybe do an inner join instead of a left outer join instead?)
    # drop 2021, 2022 as no data for droughts (/SPEI) is available for those (even though WFP data is)
    years_to_drop = [2021, 2022]
    print(f"# Dropping years (as no SPEI data available): {years_to_drop}")
    df_final = preproc.drop_years(df_final=df_final, years_list=years_to_drop)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Classify droughts & SPEI categories"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # add SPEI categories
    df_final = preproc.classify_droughts(df_final)

    # Plot Histogram for SPEI categories
    visualization.plot_hist(df_final, "SpeiCat", orientation="horizontal", bins=7, png_appendix="-preproc-STEP7")

    # Summary statistics categorical
    counts = df_final["SpeiCat"].value_counts()
    print(f"Counts:\n{counts}")

    counts_drought = df_final["Drought"].value_counts()
    print(f"Counts Drought:\n{counts_drought}")
    print(f"Share of droughts: {counts_drought[True]}/ {(counts_drought[True] + counts_drought[False])}")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Write Summary Statistics 1 (for missings in prices & drought indicators/ SPEI)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # Check missings
    stats.sum_stats_prices_and_droughts(df=df_final)
    # preproc.summary_stats_missings(df_final=df_wfp)

    # ------------------------------------------------------------------------------------------------------------------
    # TAKE CARE OF MISSING DATA -> SUBSET
    # ------------------------------------------------------------------------------------------------------------------

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: PHASE 2 (MISSING DATA: SUBSET CREATION)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Write summary statistics 2 (General)"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    df_sum_stats_commodities = stats.sum_stats_prices_and_droughts(df=df_final,
                                                                   excel_output_extension="-preproc-2",
                                                                   return_df_by_group_sheet="Commodity")
    print("Sum stats commodity\n", df_sum_stats_commodities)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Drop commodities with missings > 90%"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    cut_off_percent_commodities = 90
    df_final = preproc.drop_commodities_too_sparse(df=df_final, df_sum_stats_commodities=df_sum_stats_commodities,
                                                   cut_off_percent=cut_off_percent_commodities,
                                                   excel_to_append_dropped_commodities=
                                                   f"../output/{country}/summary-statistics/"
                                                   f"{country}-sum-stats-preproc-2.xlsx")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: PHASE 2.2 (PRICES)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Continue preprocessing PER COMMODITY
    # ------------------------------------------------------------------------------------------------------------------

    # 65 (30%), 60 (27%, 30% also for central region)
    # without populated missings: 60
    cut_off_percentile = 10

    df_commodities_dict = {}
    min_max_times_dict = {}

    # how many months do you want to look left/right after the first and last data entry. +
    # epsilon_month_extrapolation = 3
    epsilon_month_extrapolation = 0
    # how many entries are you willing to consecutively inter-/ extrapolate afterwards
    epsilon_entries_interpolation = 24

    for commodity in df_final.Commodity.unique():
        # Extract df for commodity
        df_final_commodity = df_final[df_final.Commodity == commodity]

        # Limit the dfs to specific subset
        df_final_commodity, min_time, max_time = preproc.extract_df_subset_time_prices(df_commodity=df_final_commodity,
                                                                                       epsilon_month=epsilon_month_extrapolation)
        # store min and max time for further use/ documentation
        min_max_times_dict[commodity] = (min_time, max_time)

        # calculate summary statistics PER commodity
        print(
            "\n# ----------------------------------------------------------------------------------------------------\n"
            f"# [{commodity}] PREPROC: Write summary statistics 2 (Per Commodity)"
            "\n# ---------------------------------------------------------------------------------------------------\n")
        stats.sum_stats_prices_and_droughts(df=df_final_commodity,
                                            excel_output_extension=f"-preproc-2-{commodity}",
                                            commodity=commodity)

        print(
            "\n# ---------------------------------------------------------------------------------------------------\n"
            f"# [{commodity}] Subset Markets/ region -> PREPROC: Cut off missing values after certain percentile"
            "\n# ---------------------------------------------------------------------------------------------------\n")

        # Cut all regions with missing >= cut_off_percentile of missing values
        df_final_commodity = preproc.drop_missing_percentile_per_region_prices(
            path_excel_sum_stats=f"../output/{country}/summary-statistics/{commodity}/{country}-"
                                 f"sum-stats-preproc-2-{commodity}.xlsx",
            df_final=df_final_commodity,
            cut_off_percentile=cut_off_percentile, excel_output_extension=f"-{cut_off_percentile}p-{commodity}")

        print(
            "\n# ---------------------------------------------------------------------------------------------------\n"
            f"# [{commodity}] PREPROC: Write summary statistics 3"
            "\n# ---------------------------------------------------------------------------------------------------\n")

        # Write sum stats
        stats.sum_stats_prices_and_droughts(df=df_final_commodity, excel_output_extension=
        f"-preproc-3-{cut_off_percentile}p-{commodity}", commodity=commodity)

        print(
            "\n# ----------------------------------------------------------------------------------------------------\n"
            f"# [{commodity}] PREPROC: PHASE 2.2 (PRICES) - EXTRAPOLATION"
            "\n# ----------------------------------------------------------------------------------------------------\n")

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

        # EXTRAPOLATE REGIONAL PATTERNS
        # doesn't extrapolate tails for interpolation method: cubic
        df_final_commodity = preproc.extrapolate_prices_regional_patterns(df_final=df_final_commodity,
                                                                          interpolation_method=interpolation_method,
                                                                          intrapolation_limit=epsilon_entries_interpolation,
                                                                          order=order,
                                                                          extrapolate=extrapolate)

        print(
            "\n# ----------------------------------------------------------------------------------------------------\n"
            f"# [{commodity}] PREPROC: Write summary statistics 4"
            "\n# ----------------------------------------------------------------------------------------------------\n")

        # Write sum stats
        df_sum_stats_market = stats.sum_stats_prices_and_droughts(df=df_final_commodity,
                                                                  excel_output_extension=f"-preproc-4"
                                                                                         f"-{cut_off_percentile}p"
                                                                                         f"-{commodity}"
                                                                                         f"-eps-{epsilon_entries_interpolation}",
                                                                  commodity=commodity,
                                                                  return_df_by_group_sheet="Market")

        print(
            "\n# ----------------------------------------------------------------------------------------------------\n"
            f"# [{commodity}] PREPROC: Drop markets with missings beyond interpolation range"
            "\n# ----------------------------------------------------------------------------------------------------\n")

        # TODO: check!
        # DROP ALL MARKETS THAT STILL HAVE MISSING VALUES / missing values beyond interpolation range.
        df_final_commodity = preproc.drop_markets_missing_beyond_interp_range(df_final_commodity=df_final_commodity,
                                                                              df_sum_stats_market=df_sum_stats_market,
                                                                              interpolation_limit
                                                                              =epsilon_entries_interpolation)

        # add result to dictionary
        df_commodities_dict[commodity] = df_final_commodity

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Combining all results as excel"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # TODO: move part below (separation of datasets below + write one sheet per commodity)
    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Separate datasets droughts/ no droughts"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # TODO separate drought and no-drought after df_final_all is computed (or separate them
    # within function: write_preprocessing results to excel
    # Get the distinguished datasets
    df_drought, df_no_drought = preproc.separate_df_drought_non_drought(df_final)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Writing Results as Excel"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    df_final_all = preproc.write_preprocessing_results_to_excel(df_wfp=df_wfp, df_wfp_with_coords=df_wfp_with_coords,
                                                                df_spei=df_spei,
                                                                dict_df_final_per_commodity=df_commodities_dict,
                                                                df_drought=df_drought,
                                                                df_no_drought=df_no_drought)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Writing Sum Stats (General) as Excel"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # Write sum stats for general thing
    stats.sum_stats_prices_and_droughts(df=df_final_all,
                                        excel_output_extension=f"-preproc-4"
                                                               f"-{cut_off_percentile}p"
                                                               f"-eps-{epsilon_entries_interpolation}")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# PREPROC: Writing Time Spans as Excel"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # write the subsets of time
    pd.DataFrame({
        "Commodity": min_max_times_dict.keys(),
        "TimeSpanMinY": [time_min.strftime("%Y") for time_min, time_max in min_max_times_dict.values()],
        "TimeSpanMinM": [time_min.strftime("%m") for time_min, time_max in min_max_times_dict.values()],
        "TimeSpanMaxY": [time_max.strftime("%Y") for time_min, time_max in min_max_times_dict.values()],
        "TimeSpanMaxM": [time_max.strftime("%m") for time_min, time_max in min_max_times_dict.values()],
        "Epsilon (Month)": [epsilon_month_extrapolation] * len(min_max_times_dict.keys())
    }).to_excel(f"../output/{country}/summary-statistics/time-spans-per-commodity.xlsx")

    return df_final_all
