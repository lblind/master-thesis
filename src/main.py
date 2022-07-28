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
import statistics_snippets as stats

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

    # TODO: Outcomment this line if dataset hat not yet been created
    df_final = dataset_creation.create_dataset(country=country, dropped_commodities=dropped_commodities,
                                                write_results_to_excel=False)

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# ANALYSIS"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    path_to_final_df = f"../output/{country}/{country}-final-dta.xlsx"
    # TODO outcomment this line if dataset has already been created
    # df_final = utils.convert_excel_to_df(path_to_final_df)
    # visualization.plot_hist(df_final, "SpeiCat", orientation="horizontal", bins=7, png_appendix="-preproc-STEP6")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# VISUALIZATION"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # visualization.plot_malawi(df_final=df_final)
    # visualization.plot_malawi_regions(df_final)

    # plot districts
    # visualization.plot_malawi_districts_adm2(df_final)

    # visualization.plot_country_adm2_prices_for_year_month(df_final, 2018, 8, "Maize")
    # visualization.plot_country_adm2_price_spei(df_final, 2018, 8, "Maize")
    # visualization.plot_missings(df_final, "Price")

    print("\n# ------------------------------------------------------------------------------------------------------\n"
          "# DYNAMIC MODE DECOMPOSITION (DMD)"
          "\n# ------------------------------------------------------------------------------------------------------\n")

    # dmd.dmd_per_commodity(df_final)