"""
MAIN
(Control)

Starting point of program execution
"""

import preprocessing as preproc


if __name__ == "__main__":
    print("Hello World")

    central_path_to_csv_all = "../input/food-price-dta/Central_All_Commodities_WFP_2022Jun14_Malawi_FoodPricesData.csv"
    northern_path_to_csv_all = "../input/food-price-dta/Northern_All_Commodities_WFP_2022Jun14_Malawi_FoodPricesData.csv"
    southern_path_to_csv_all = "../input/food-price-dta/Southern_All_Commodities_WFP_2022Jun14_Malawi_FoodPricesData.csv"

    df_central = preproc.read_and_clean_csv(central_path_to_csv_all)
    df_northern = preproc.read_and_clean_csv(northern_path_to_csv_all)
    df_southern = preproc.read_and_clean_csv(southern_path_to_csv_all)

    dfs = [df_central, df_northern, df_southern]

    for i, df in enumerate(dfs):
        print(f"#------------------------------------------------------------------------\n"
              f"# [{i}]\n"
              f"#------------------------------------------------------------------------\n")
        print(df)
        print(df.columns)
        for column in df.columns:
            print(df[column].unique())
        # print(df["Commodity"].unique())
        # print(df["Year"].unique())
        # print(df["Market"].unique())

