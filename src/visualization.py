"""
VISUALIZATION
-------------

"""

# TODO: color regions
# TODO: visualize prices (e.g. color scale, kdeplot)
# from mpl_toolkits.basemap import Basemap
import os

import geoplot as gplt
import geopandas as gpd
import geoplot.crs as gcrs
import imageio
import pandas as pd
import pathlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mapclassify as mc
import numpy as np
import folium
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plotly.express as px

import missingno as msgo
import statistics_snippets as stats


def plot_missings(df_final, column):
    """
    Plot number and percent of missing values

    :param df_final:
    :param column:
    :return:
    """
    msgo.bar(df_final)

    plt.title("Missing values")
    plt.xlabel("Variable")
    plt.ylabel("Percent")
    plt.show()


def plot_malawi_regions_adm1(df_final):
    """
    Plot malawi and color the specific regions

    :param df_final:
    :return:
    """
    country = df_final.Country.unique()[0]
    output_path_maps = f"../output/{country}/plots/maps"
    if os.path.exists(output_path_maps) is False:
        os.makedirs(output_path_maps)

    malawi_adm2 = gpd.read_file("../input/Malawi/maps/WFPGeoNode/mwi_bnd_admin2/mwi_bnd_admin2.shp")
    malawi_adm1 = gpd.read_file("../input/Malawi/maps/WFPGeoNode/mwi_bnd_admin1/mwi_bnd_admin1.shp")

    malawi_adm1.rename(columns={"NAME_1": "Region"}, inplace=True)
    malawi_adm2.rename(columns={"NAME_2": "Market"}, inplace=True)

    print(malawi_adm1.Region.unique())

    fig, ax = plt.subplots(1, 1)
    # divider = make_axes_locatable(ax)

    # cax = divider.append_axes("right", size="5%", pad=0.8)
    # lower right
    # malawi_adm1.plot(column="Region", ax=ax, legend=True, legend_kwds={"loc": "upper left",
    #
    #                                                                    "bbox_to_anchor" : (0.8, 0.1)})
    # lower left
    # malawi_adm1.plot(column="Region", ax=ax, legend=True, legend_kwds={"loc": "upper right",
    #                                                                    "bbox_to_anchor": (0.8, 0.1)})
    # jet, rainbow too colorful
    # doesn't work/ white: terrain, Purples, greens, Blues
    possible_cmaps = ["magma", "viridis", "Blues", "plasma", "jet",
                      "rainbow", "turbo", "cubehelix", "terrain", "tab20b", "seismic", "Blues_r",
                      "bone", "winter", "summer", "Purples", "Greens_r", "viridis_r", "YlGn",
                      "YlGn_r", "YlGnBu_r", "YlGnBu", "Spectral_r", "Purples_r",
                      "PuBuGn_r", "PiYG_r", "BrBG_r", "BrBG", "Dark2_r"]
    # 4, winter (not enough contrast), summer (nice, but creates wrong image)
    # magma, plasma, turbo
    cmap = possible_cmaps[-1]
    # cmap = "summer" +"_r"
    cmap = "summer"

    # ax.set_prop_cycle(color=cmap[1:])
    malawi_adm1.plot(column="Region", ax=ax, legend=True, legend_kwds={"loc": "lower left",
                                                                       "bbox_to_anchor": (0.6, 0.8)},
                     cmap=cmap)

    # default
    # malawi_adm1.plot(column="Region", ax=ax, legend=True, legend_kwds={"loc": "lower left",
    #                                                                    "bbox_to_anchor": (0.6, 0.8)})

    # plt.tight_layout()

    # convert regular dataframe to geopandas df
    gdf_final_markets = gpd.GeoDataFrame(
        df_final, geometry=gpd.points_from_xy(df_final.MarketLongitude, df_final.MarketLatitude)
    )

    plt.scatter(df_final.MarketLongitude, df_final.MarketLatitude, c="darkblue", edgecolor="orange")

    # gplt.pointplot(gdf_final_markets, ax=ax)
    # gdf_final_markets.plot(kind="scatter", ax=ax)
    # plt.grid()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.suptitle("Malawi - Regions")
    # plt.title("Malawi - Regions", loc="left")

    plt.savefig(f"{output_path_maps}/{country}-Regions.png")

    plt.show()


def histogram(df, column, commodity, year, month, bins=20, save_fig=False):
    """

    :param df:
    :param column:
    :param bins:
    :return:
    """
    country = df.Country.unique()[0]
    output_path_maps = f"../output/{country}/plots/histograms"
    if os.path.exists(output_path_maps) is False:
        os.makedirs(output_path_maps)

    df[column].hist(bins=bins)
    plt.xlabel("SPEI")
    plt.ylabel("Number of markets")
    plt.suptitle(f"Histogram {column} - {commodity}")
    plt.title(f"{year}, {month}")
    plt.show()

    if save_fig:
        plt.savefig(output_path_maps + f"/{commodity}-{year}-{month}-hist.png")


def plot_country_adm2_prices_for_year_month(df_final, year, month, commodity=None,
                                            path_shape_file="../input/Malawi/maps/WFPGeoNode/mwi_bnd_admin2/mwi_bnd_admin2.shp"):
    """
    Plots
    :param df_final:
    :param year:
    :param month:
    :param commodity:
    :return:
    """
    # Make sure that output dir exists
    country = df_final.Country.unique()[0]
    output_path_maps = f"../output/{country}/plots/maps"
    if os.path.exists(output_path_maps) is False:
        os.makedirs(output_path_maps)

    # just extract subset, year
    df_final = df_final[df_final.Year == year]

    # just extract subset, month
    df_final = df_final[df_final.Month == month]

    # plot only subset of commodity
    if commodity is not None:
        if commodity in df_final.Commodity.unique():
            df_final = df_final[df_final.Commodity == commodity]
        else:
            raise ValueError(f"Commodity {commodity} not valid. Please choose "
                             f"one of the following commodities: {df_final.Commodity.unique()}")

    # Read shape file
    malawi_adm2 = gpd.read_file(path_shape_file)
    malawi_adm2.rename(columns={"NAME_2": "District"}, inplace=True)
    fig, ax = plt.subplots(1, 1)

    # rename geometry column
    malawi_adm2 = malawi_adm2.rename_geometry("geometry_adm2")

    # Set current coordinate reference system
    crs_adm2 = malawi_adm2.crs

    # convert regular dataframe to geopandas df
    gdf_final_markets = gpd.GeoDataFrame(
        df_final, geometry=gpd.points_from_xy(df_final.MarketLongitude, df_final.MarketLatitude)
    )
    # rename the geometry column
    gdf_final_markets = gdf_final_markets.rename_geometry("geometry_markets")
    print(gdf_final_markets.columns)



    # set geometry columns explicitly
    # gdf_final_markets = gdf_final_markets.set_geometry("geometry_markets")
    # copy the column to a new one
    # gdf_final_markets["geometry_markets"] = gdf_final_markets.geometry

    gdf_final_markets.crs = crs_adm2

    cmap = "summer"

    # TODO: keep both geometry columns

    # spatial join: find the fitting admin 2 for each market
    # as the geometries of the right df should be sustained (District Polygons not Market Points)
    join_method = "right"
    # gdf_markets_with_admin2 = gpd.sjoin(gdf_final_markets.to_crs(crs=crs_adm2), malawi_adm2, how="inner",
    #                                     predicate="intersects")
    gdf_markets_with_admin2 = gpd.sjoin(malawi_adm2, gdf_final_markets.to_crs(crs=crs_adm2),  how="inner",
                                        predicate="intersects")

    print(gdf_markets_with_admin2.columns)

    # TODO: not necessary (created mean column is not used)
    # gdf_merged = stats.mean_column_per_group(gdf_markets_with_admin2, group="District", column="AdjPrice")

    gdf_merged = gdf_markets_with_admin2

    # ------------------------------------------------------------------------------------------------------------------
    # Scatterplot - Prices
    # ------------------------------------------------------------------------------------------------------------------
    color_array = ["darkred" if row.Drought else "darkblue" for idx, row in gdf_merged.iterrows()]

    # TODO maybe change that to a pointplot
    scale_factor_percent = 25
    adj_prices_scaled = gdf_merged.AdjPrice * (scale_factor_percent / 100)
    sc = plt.scatter(gdf_merged.MarketLongitude, gdf_merged.MarketLatitude, c=color_array, edgecolor="orange",
                     s=adj_prices_scaled, alpha=0.7, zorder=2, label=adj_prices_scaled)

    currency = df_final.Currency.unique()[0]
    legend_prices = ax.legend(*sc.legend_elements("sizes", num=6), loc="lower left", bbox_to_anchor=(-1.3, 0.35),
                              title=f"Price [{scale_factor_percent}% {currency}]")


    print("Adj prices scaled\n", adj_prices_scaled.unique())
    print(f"Adj prices scaled min: {np.min(adj_prices_scaled)}, max: {np.max(adj_prices_scaled)}")

    # ------------------------------------------------------------------------------------------------------------------
    # MAP Malawi
    # ------------------------------------------------------------------------------------------------------------------



    # to plot the lines between the districts: edgecolor="darkgreen"
    # 1) Plot Malawi
    # malawi_adm2.plot(column="District", ax=ax, legend=True, legend_kwds={"loc": "lower left",
    #                                                                      "bbox_to_anchor": (1.1, -0.105),
    #                                                                      "fontsize": "x-small",
    #                                                                      "title": "District"},
    #                  cmap=cmap, edgecolor="darkgreen")

    gdf_merged.plot(column="District", ax=ax, legend=True, legend_kwds={"loc": "lower left",
                                                                         "bbox_to_anchor": (1.1, -0.105),
                                                                         "fontsize": "x-small",
                                                                         "title": "District"},
                     cmap=cmap, edgecolor="darkgreen")

    # manually add legend for prices back
    ax.add_artist(legend_prices)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{year}, {month}")

    if commodity is not None:
        plt.suptitle(f"Malawi - {commodity}")
        plt.savefig(f"{output_path_maps}/{country}-Districts-Adm2-Prices-{year}-{month}-{commodity}.png")
    else:
        plt.suptitle("Malawi - Markets")
        plt.savefig(f"{output_path_maps}/{country}-Districts-Adm2-Prices-{year}-{month}.png")

    plt.show()


def plot_prices_and_spei_adm2(df_final):
    """
    Plot relative prices and color districts based on their mean spei value
    :param df_final:
    :return:
    """
    country = df_final.Country.unique()[0]
    output_path_maps = f"../output/{country}/plots/maps"
    if os.path.exists(output_path_maps) is False:
        os.makedirs(output_path_maps)

    malawi_adm2 = gpd.read_file("../input/Malawi/maps/WFPGeoNode/mwi_bnd_admin2/mwi_bnd_admin2.shp")
    malawi_adm2.rename(columns={"NAME_2": "District"}, inplace=True)
    fig, ax = plt.subplots(1, 1)

    crs_adm2 = malawi_adm2.crs

    # convert regular dataframe to geopandas df
    gdf_final_markets = gpd.GeoDataFrame(
        df_final, geometry=gpd.points_from_xy(df_final.MarketLongitude, df_final.MarketLatitude)
    )

    gdf_final_markets.crs = crs_adm2

    print(gdf_final_markets.columns)
    print(malawi_adm2.columns)
    cmap = "summer"

    # spatial join: find the fitting admin 2 for each market
    gdf_markets_with_admin2 = gpd.sjoin(gdf_final_markets.to_crs(crs=crs_adm2), malawi_adm2, how="inner",
                                        predicate="intersects")

    gdf_merged = stats.mean_column_per_group(gdf_markets_with_admin2, group="District", column="AdjPrice")

    # add spei mean
    gdf_merged = stats.mean_column_per_group(gdf_merged, group="District", column="Spei")

    print("Shape: ", gdf_merged.shape)
    print(len(gdf_merged.MeanAdjPricePerDistrict.unique()))
    print(len(gdf_merged.District.unique()))

    adj_prices_scaled = gdf_merged.AdjPrice * 0.02
    sc = plt.scatter(gdf_merged.MarketLongitude, gdf_merged.MarketLatitude, c="darkblue", edgecolor="orange",
                     s=adj_prices_scaled, alpha=0.7, zorder=2, label=adj_prices_scaled)

    legend_prices = ax.legend(*sc.legend_elements("sizes", num=6), loc="lower left", bbox_to_anchor=(-1.1, 0.35),
                              title="Prices")

    # legend_prices = ax.legend(scatterpoints=5)

    # 1) Plot Malawi
    malawi_adm2.plot(column="MeanSpeiPerDistrict", ax=ax, legend=True, legend_kwds={"loc": "lower left",
                                                                                    "bbox_to_anchor": (1.1, -0.1),
                                                                                    "fontsize": "x-small",
                                                                                    "title": "Districts"},
                     cmap=cmap)

    # manually add legend for prices back
    ax.add_artist(legend_prices)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.suptitle("Malawi - Markets")
    plt.title("AdjPrices")

    plt.savefig(f"{output_path_maps}/{country}-Districts-Adm2-Prices.png")

    plt.show()


def plot_malawi_districts_adm2(df_final):
    """

    :return:
    """
    country = df_final.Country.unique()[0]
    output_path_maps = f"../output/{country}/plots/maps"
    if os.path.exists(output_path_maps) is False:
        os.makedirs(output_path_maps)

    malawi_adm2 = gpd.read_file("../input/Malawi/maps/WFPGeoNode/mwi_bnd_admin2/mwi_bnd_admin2.shp")
    malawi_adm2.rename(columns={"NAME_2": "District"}, inplace=True)
    fig, ax = plt.subplots(1, 1)

    crs_adm2 = malawi_adm2.crs

    # convert regular dataframe to geopandas df
    gdf_final_markets = gpd.GeoDataFrame(
        df_final, geometry=gpd.points_from_xy(df_final.MarketLongitude, df_final.MarketLatitude)
    )

    gdf_final_markets.crs = crs_adm2

    print(gdf_final_markets.columns)
    print(malawi_adm2.columns)
    cmap = "summer"

    malawi_adm2.plot(column="District", ax=ax, legend=True, legend_kwds={"loc": "lower left",
                                                                         "bbox_to_anchor": (1.1, -0.1),
                                                                         "fontsize": "x-small"},
                     cmap=cmap)

    # spatial join: find the fitting admin 2 for each market
    gdf_markets_with_admin2 = gpd.sjoin(gdf_final_markets.to_crs(crs=crs_adm2), malawi_adm2, how="inner",
                                        predicate="intersects")

    # gdf_markets_with_admin2.plot(kind="geo", column="Region", ax=ax, legend=True, legend_kwds={"loc": "lower left",
    #                                                                    "bbox_to_anchor": (0.6, 0.8)},
    #                  cmap=cmap)

    plt.scatter(df_final.MarketLongitude, df_final.MarketLatitude, c="darkblue", edgecolor="orange")
    # plt.tight_layout()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.suptitle("Malawi - Districts")
    # plt.title("Malawi - Regions", loc="left")

    plt.savefig(f"{output_path_maps}/{country}-Districts-Adm2.png")

    plt.show()


def plot_malawi(df_final):
    """

    Admin1: Regions
    Admin2: Markets
    Structure Shape File
    NAME_0: Name of country
    NAME_1: Name of bigger regions
    NAME_2: Name of cities


    References
    ----------
    Data source/ shape files extracted via: https://www.diva-gis.org/datadown
    :return:
    """
    # Read shape file(s)
    # WFP GeoData
    malawi_adm1 = gpd.read_file("../input/Malawi/maps/WFPGeoNode/mwi_bnd_admin1/mwi_bnd_admin1.shp")
    malawi_adm2 = gpd.read_file("../input/Malawi/maps/WFPGeoNode/mwi_bnd_admin2/mwi_bnd_admin2.shp")

    # Humanitarian Data Exchange
    malawi_adm3 = gpd.read_file("../input/Malawi/maps/hum-data-exchange/mwi_adm_nso_20181016_shp"
                                "/mwi_admbnda_adm3_nso_20181016.shp")

    print("Malawi admin 1\n", malawi_adm1, "\nColumns", malawi_adm1.columns)
    print("Malawi admin 2\n", malawi_adm2, "\nColumns", malawi_adm2.columns)
    print("Malawi admin 3\n", malawi_adm3, "\nColumns", malawi_adm3.columns)

    # print(malawi_adm3.Market.unique(), len(malawi_adm2.Market.unique()))

    malawi_adm1.rename(columns={"NAME_1": "Region"}, inplace=True)
    malawi_adm2.rename(columns={"NAME_2": "Market"}, inplace=True)
    malawi_adm3.rename(columns={"ADM3_EN": "TA"}, inplace=True)

    print(malawi_adm1.Region.unique())
    print(malawi_adm2.Market.unique(), len(malawi_adm2.Market.unique()))
    print(malawi_adm3.TA.unique(), len(malawi_adm3.TA.unique()))
    # merge adm1 data (for now) with df_final
    # df_final_merged = malawi_adm1.merge(df_final, on="Market")
    df_final_merged = malawi_adm2.merge(df_final, on="Market")

    print("Unique Markets\n",
          len(df_final_merged.Market.unique()), len(df_final.Market.unique()))

    # convert regular dataframe to geopandas df
    gdf_final = gpd.GeoDataFrame(
        df_final, geometry=gpd.points_from_xy(df_final.MarketLongitude, df_final.MarketLatitude)
    )

    gdf_markets_with_admin3 = gpd.sjoin(gdf_final, malawi_adm3, how="inner", predicate="intersects")
    print(gdf_markets_with_admin3.columns, gdf_markets_with_admin3.shape)

    print(f"Merged shape: {df_final_merged.shape}")

    print(malawi_adm2.Admin2.unique())
    print(malawi_adm1.Admin1.unique())

    # Plot boundaries Admin 1

    # Plot boundaries Admin 2
    # ax = gplt.polyplot(malawi_adm2)
    ax = gplt.polyplot(malawi_adm2)

    # PLOT POINTS / Markets on it
    gplt.pointplot(gdf_final, ax=ax, hue="Region")
    # gplt.pointplot(gdf_final, ax=ax)

    # ax = gplt.polyplot(df_final_merged, hue="Region")
    # ax = gplt.polyplot(malawi_adm1, projection=gcrs.AlbersEqualArea())
    # print(df_final_merged.head())
    # gplt.choropleth(
    #     gdf_markets_with_admin3,
    #     hue="Market",
    #     edgecolor="white",
    #     linewidth=1,
    #     cmap="Greens",
    #     legend=True,
    #     scheme="FisherJenks",
    #     ax=ax
    # )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.suptitle("Malawi")
    plt.title("Markets (per Region)")
    plt.legend()
    plt.savefig("../output/Malawi/plots/Map-Markets.png")
    plt.show()


def visualize_prices(df_final):
    """

    :param df_final:
    :return:
    """
    pass
