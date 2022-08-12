"""
MAIN
---------
(Control)

Starting point of program execution
"""
import os

import preprocessing as preproc
import pandas as pd
import utils
import visualization
import dataset_creation
import analysis as stats

import dmd

if __name__ == "__main__":

    # CONFIGURATION
    # -----------------------------------
    # 1) set country
    country = "Malawi"
    # country = "Somalia"
    # country = "Tanzania"
    # country = "Kenya"

    # 2) set commmodities to drop
    # Drop because not available in all regions (only south)
    dropped_commodities = ["Maize (white)", "Rice (imported)", "Sorghum (red)"]

    # Drop nothing
    # dropped_commodities = None

    # # TODO: Outcomment this line if dataset hat not yet been created
    df_final = dataset_creation.create_dataset(country=country, dropped_commodities=dropped_commodities,
                                               write_results_to_excel=True)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# ANALYSIS"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    path_to_final_df = f"../output/{country}/{country}-final-dta.xlsx"
    # TODO outcomment this line if dataset has already been created
    df_final = utils.convert_excel_to_df(path_to_final_df)

    # stats.df_describe_excel(df_final, group_by_column="Commodity")

    path_to_df_wfp = f"../output/{country}/intermediate-results/df_wfp_STEP4.xlsx"
    df_wfp = utils.convert_excel_to_df(path_to_df_wfp)

    df_with_drought = utils.convert_excel_to_df(f"../output/{country}/intermediate-results/df_wfp_w_drought_STEP4.xlsx")
    df_with_drought = utils.merge_drought_to_df_wfp(df_wfp)

    df_with_drought.to_excel(f"../output/{country}/intermediate-results/df_wfp_w_drought_STEP4.xlsx")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# ANALYSIS - Statistics"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # create stats just by commodity
    stats.df_describe_excel(df_wfp, group_by_column="Commodity", excel_extension="-STEP4", column="AdjPrice")
    # ... and by commodity & region
    stats.df_describe_excel(df_wfp, group_by_column=["Commodity", "Region"], excel_extension="-STEP4", column="AdjPrice")

    # add spei dataset & classify droughts
    df_wfp_drought = utils.merge_drought_to_df_wfp(df_wfp)
    # stats by commodity & drought
    stats.df_describe_excel(df_wfp_drought, group_by_column=["Commodity", "Drought"], excel_extension="-STEP4", column="AdjPrice")
    # stats by commodity, drought & region
    stats.df_describe_excel(df_wfp_drought, group_by_column=["Commodity", "Drought", "Region"], excel_extension="-STEP4", column="AdjPrice")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# ANALYSIS - Peak"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    df_with_peak = stats.identify_spikes_per_commodity(df_wfp)
    visualization.scatter_spikes_per_commodity(df_with_peak)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# ANALYSIS - Correlations"
          "\n# ------------------------------------------------------------------------------------------------------\n")


    print(df_with_drought.columns)
    df_corr = stats.compute_correlations(df_with_drought)

    visualization.plot_correlation_matrix(df_wfp)




    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# VISUALIZATION"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# VISUALIZATION - Explorative analysis"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# VISUALIZATION - Scatter plots"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    visualization.scatter_adj_prices_per_region_one_fig(df_wfp=df_wfp)
    visualization.scatter_adj_prices_all_commodities_droughts(df_wfp=df_wfp)
    visualization.scatter_adj_price_per_region_drought_one_fig(df_wfp=df_wfp)

    # visualization.plot_hist_for_all_commodities(df_wfp)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# VISUALIZATION - Boxplots"
          "\n# ------------------------------------------------------------------------------------------------------\n")
    visualization.box_plot_for_all_commodities_by_group(df_wfp)
    visualization.box_plot_for_all_commodities_by_group(df_wfp, by="Drought")
    visualization.box_plot_for_all_commodities_by_group(df_wfp, by="Region")
    visualization.box_plot_for_all_commodities_by_group(df_wfp, by=["Drought", "Region"])


    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# VISUALIZATION - Maps"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    visualization.plot_malawi_regions_adm1(df_with_drought, scatter_markets=False)
    visualization.plot_malawi_districts_adm2(df_with_drought, plot_markets=True)

    visualization.plot_malawi(df_final=df_final)

    # plot districts
    visualization.plot_malawi_districts_adm2(df_final)

    visualization.plot_country_adm2_prices_for_year_month(df_final, 2018, 8, "Maize")
    visualization.plot_country_adm2_price_spei(df_final, 2018, 8, "Maize")
    visualization.plot_missings(df_final, "Price")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# DYNAMIC MODE DECOMPOSITION (DMD)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    dmd.dmd_per_commodity(df_final, exact_modes=False, transpose=False)