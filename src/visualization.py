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

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates


import plotly.express as px

import missingno as msgo
import statistics_snippets as stats


# ----------------------------------------------------------------------------------------------------------------------
# (LINE) PLOTS
# ----------------------------------------------------------------------------------------------------------------------

def line_plot_spei(df_spei, country, show=True, alpha=0.5):
    """

    :param df_spei:
    :param show:
    :param alpha:
    :return:
    """

    print("Columns df spei:\n", df_spei.columns)

    # Make sure Output dir exists
    output_dir = f"../output/{country}/plots/line-plots/spei"
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    plt.plot(df_spei.time, df_spei["spei"], label="Original points",
             alpha=alpha)

    plt.title(f"Spei")
    plt.xlabel("Time")
    plt.ylabel(f"Spei")
    plt.legend()

    plt.savefig(f"{output_dir}/scatter-spei.png")

    if show:
        plt.show()


def line_plot_mean_min_max_spei_per_time(df_spei, country, time="Year", show=True, alpha=0.5):
    """

    :param df_wfp_and_spei:
    :param show:
    :param alpha:
    :return:
    """

    output_dir = f"../output/{country}/plots/line-plots/spei"
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    # compute means per time
    means_per_time = df_spei.groupby(time).mean()

    # compute mins, maxs per time
    mins_per_time = df_spei.groupby(time).min()
    maxs_per_time = df_spei.groupby(time).max()

    if "spei" in means_per_time:
        mean_spei_per_time = means_per_time["spei"]
        min_spei_per_time = mins_per_time["spei"]
        max_spei_per_time = maxs_per_time["spei"]
    elif "Spei" in means_per_time:
        mean_spei_per_time = means_per_time["Spei"]
        min_spei_per_time = mins_per_time["Spei"]
        max_spei_per_time = maxs_per_time["Spei"]

    else:
        raise KeyError("Identifier for SPEI in passed df not recognized. "
                       "Only valid column names are: spei, Spei.\n"
                       f"Not found inL {df_spei.columns}")

    plt.plot(df_spei[time].unique(), mean_spei_per_time, alpha=alpha, label="Mean")
    plt.plot(df_spei[time].unique(), min_spei_per_time, alpha=alpha, label="Min")
    plt.plot(df_spei[time].unique(), max_spei_per_time, alpha=alpha, label="Max")

    # plt.scatter(df_spei[time].unique(), mean_spei_per_time, alpha=alpha, label="Mean")
    # plt.scatter(df_spei[time].unique(), min_spei_per_time, alpha=alpha, label="Min")
    # plt.scatter(df_spei[time].unique(), max_spei_per_time, alpha=alpha, label="Max")

    plt.hlines(-1, df_spei[time].unique().min(), df_spei[time].unique().max(),
               colors="red",
               label="Drought",
               linestyles="dashed")
    plt.hlines(1, df_spei[time].unique().min(), df_spei[time].unique().max(),
               colors="blue",
               label="Flood",
               linestyles="dashed")

    axs = plt.gca()
    # Remove axes splines
    # removed_axes_splines = ['top', 'bottom', 'left', 'right']
    removed_axes_splines = ['top', 'right']
    for s in removed_axes_splines:
        axs.spines[s].set_visible(False)

    # Add padding between axes and labels
    axs.xaxis.set_tick_params(pad=5)
    axs.yaxis.set_tick_params(pad=5)

    # Add x, y gridlines
    axs.grid(b=True, color='grey',
             linestyle='-.', linewidth=0.5,
             alpha=0.6)

    plt.suptitle(f"SPEI per {time}")
    plt.title("Mean, Min, Max", style="italic")
    plt.xlabel(f"Time", style="italic")
    plt.ylabel(f"SPEI", style="italic")

    # shift more to right: increase x-value in tuple
    plt.legend(loc="lower left", bbox_to_anchor=(0.9, 0.8))

    plt.savefig(f"{output_dir}/scatter-mean-spei-per-{time}.png")

    if show:
        plt.show()


def line_plot_spei_per_region(df_wfp_and_spei, show=True, alpha=0.5):
    """

    :return:
    """
    country = df_wfp_and_spei.Country.unique()[0]

    # Make sure Output dir exists
    output_dir = f"../output/{country}/plots/line-plots/spei"
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    for region in df_wfp_and_spei.Region.unique():

        # plt.scatter(df_wfp_and_spei.TimeWFP, df_wfp_and_spei["Spei"], label="Original points",
        #            alpha=alpha)
        plt.plot(df_wfp_and_spei.TimeWFP, df_wfp_and_spei["Spei"], label="Original points",
                 alpha=alpha)

        plt.hlines(-1, df_wfp_and_spei["TimeWFP"].unique().min(), df_wfp_and_spei["TimeWFP"].unique().max(),
                   colors="red",
                   label="Drought",
                   linestyles="dashed")
        plt.hlines(1, df_wfp_and_spei["TimeWFP"].unique().min(), df_wfp_and_spei["TimeWFP"].unique().max(),
                   colors="blue",
                   label="Flood",
                   linestyles="dashed")

        plt.title(f"SPEI - Region: {region},")
        plt.xlabel("Time")
        plt.ylabel(f"SPEI")
        plt.legend()

        plt.savefig(f"{output_dir}/{region}-scatter-spei.png")

        if show:
            plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# SCATTER PLOTS
# ----------------------------------------------------------------------------------------------------------------------


def scatter_adj_price_region_all_commodities(df_wfp, alpha=0.5):
    """

    :param df_wfp:
    :return:
    """
    currency = df_wfp.Currency.unique()[0]
    for commodity in df_wfp.Commodity.unique():
        df_wfp_commodity = df_wfp[df_wfp.Commodity == commodity]

        plt.scatter(df_wfp_commodity.TimeWFP, df_wfp_commodity["AdjPrice"], label="Original points",
                    alpha=alpha)

        plt.suptitle("(Inflation-Adjusted) Price Distribution")
        plt.title(f"Commodity: {commodity}")
        plt.xlabel("Time")
        plt.ylabel(f"Price [{currency}]")
        plt.legend()

        # option 1)
        plt.tight_layout()

        # format the axis
        # just display the year
        # plt.gca().xaxis.set_major_formatter(DateFormatter("%Y"))
        # display year and month
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y, %m"))
        plt.xticks(rotation=30)


        # Add x, y gridlines
        plt.gca().xaxis.grid(b=True, color='grey',
                               linestyle='-.', linewidth=0.5,
                               alpha=0.6)
        plt.gca().yaxis.grid(b=True, color='grey',
                               linestyle='-.', linewidth=0.5,
                               alpha=0.6)

        # Make sure Output dir exists
        country = df_wfp.Country.unique()[0]
        output_dir = f"../output/{country}/plots/scatter-plots/{commodity}"
        if os.path.exists(output_dir) is False:
            os.makedirs(output_dir)

        plt.savefig(f"{output_dir}/{commodity}-scatter-adj-prices-STEP3.png")
        plt.show()



def scatter_adj_prices_per_region_one_fig(df_wfp):
    """

    :return:
    """

    n_regions = len(df_wfp.Region.unique())

    for commodity in df_wfp.Commodity.unique():
        df_wfp_commodity = df_wfp[df_wfp.Commodity == commodity]

        fig, ax = plt.subplots(n_regions, 1, sharex=True, sharey=True)

        for i, region in enumerate(df_wfp_commodity.Region.unique()):
            df_wfp_commodity_region = df_wfp_commodity[df_wfp_commodity.Region == region]
            # ax[0, i].scatter(df_wfp.TimeWFP, df_wfp.AdjPrice, c = ["red" if df_wfp.Drought else "blue"], alpha=0.5)
            ax[i].scatter(df_wfp_commodity_region.TimeWFP, df_wfp_commodity_region.AdjPrice, alpha=0.5)
            ax[i].set_title(f"{region}")

            # Add padding between axes and labels
            ax[i].get_xaxis().set_tick_params(pad=5)
            ax[i].get_yaxis().set_tick_params(pad=5)

            if i != n_regions -1:
                ax[i].get_xaxis().set_visible(False)

            # # Add x, y gridlines
            ax[i].get_xaxis().grid(b=True, color='grey',
                     linestyle='-.', linewidth=0.5,
                     alpha=0.6)
            ax[i].get_yaxis().grid(b=True, color='grey',
                                   linestyle='-.', linewidth=0.5,
                                   alpha=0.6)



        # set the spacing between subplots
        # option 1)
        fig.tight_layout()

        # format the axis
        # just display the year
        # plt.gca().xaxis.set_major_formatter(DateFormatter("%Y"))
        # display year and month
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y, %m"))
        plt.xticks(rotation=30)

        # option 2)
        plt.subplots_adjust(left=0.1,
                            bottom=0.15,
                            right=0.9,
                            top=0.85,
                            wspace=0.4,
                            hspace=0.4)

        # define range of x axis
        # plt.xticks(ticks=df_wfp_commodity.Year)


        fig.suptitle(f"Commodity: {commodity}")

        # Make sure Output dir exists
        country = df_wfp.Country.unique()[0]
        output_dir = f"../output/{country}/plots/scatter-plots/{commodity}"
        output_dir_extrapolation = output_dir + "/extrapolation"
        if os.path.exists(output_dir_extrapolation) is False:
            os.makedirs(output_dir_extrapolation)
        plt.savefig(f"{output_dir}/{commodity}-scatter-adj-prices-regions-one-fig.png")
        plt.show()


def scatter_adj_prices_per_region(df_region, commodity, currency, show=False, alpha=0.5):
    """

    :param commodity:
    :param region:
    :return:
    """
    country = df_region.Country.unique()[0]
    region = df_region.Region.unique()[0]

    # Make sure Output dir exists
    output_dir = f"../output/{country}/plots/scatter-plots/{commodity}"
    output_dir_extrapolation = output_dir + "/extrapolation"
    if os.path.exists(output_dir_extrapolation) is False:
        os.makedirs(output_dir_extrapolation)
    plt.scatter(df_region.TimeWFP, df_region["AdjPrice"], label="Original points",
                alpha=alpha)

    plt.suptitle("(Inflation-Adjusted) Price Distribution")
    plt.title(f"Region: '{region}', Commodity: {commodity}")
    plt.xlabel("Time")
    plt.ylabel(f"Price [{currency}]")
    plt.legend()

    plt.savefig(f"{output_dir}/{region}-{commodity}-scatter-adj-prices.png")

    if show:
        plt.show()


def scatter_extrapolated_adj_prices_per_region(df_region, df_region_extrapolated,
                                               commodity, currency, interpolation_method, order,
                                               show=False, alpha=0.5,
                                               show_original_points=True):
    # Extract data
    country = df_region.Country.unique()[0]
    region = df_region.Region.unique()[0]

    # Make sure Output dir exists
    output_dir = f"../output/{country}/plots/scatter-plots/{commodity}"
    output_dir_extrapolation = output_dir + "/extrapolation"
    if os.path.exists(output_dir_extrapolation) is False:
        os.makedirs(output_dir_extrapolation)

    # Plot raw data or not
    if show_original_points:
        scatter_adj_prices_per_region(df_region=df_region, currency=currency,
                                      commodity=commodity, show=False, alpha=alpha)
    # Plot extrapolated data
    plt.scatter(df_region_extrapolated.TimeWFP[df_region.Price.isna()],
                df_region_extrapolated.AdjPrice[df_region.Price.isna()],
                color="green", label="Extra-/Interpolated points", marker="*", alpha=alpha)
    plt.legend()
    plt.savefig(f"{output_dir_extrapolation}/{region}-{commodity}-scatter-prices-extrapolated-"
                f"{interpolation_method}-{order}.png")
    if show:
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# BAR CHARTS
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# HISTOGRAMS
# ----------------------------------------------------------------------------------------------------------------------

def plot_hist(df, column_x, rotation=45, bins=7, orientation="vertical", rwidth=0.75, png_appendix=""):
    """

    :param df:
    :param column_x:
    :param column_y:
    :return:
    """

    if column_x not in df:
        raise ValueError(f"Can't plot histogram, as column {column_x} not existent. \n"
                         f"Please revise your definition and use one of the following columns instead:\n{df.columns}")

    print(f"Uniques:\n {df[column_x].unique()}")
    print(df[df[column_x].isna()]["Spei"])
    # "royalblue" with 0.5 gives light blue
    # blue, alpha=0.5 -> purple
    n, bins, patches = plt.hist(df[column_x], bins=bins, rwidth=rwidth, align="mid", orientation=orientation,
                                color="lightseagreen", alpha=1)
    plt.xticks(rotation=rotation)

    if orientation == "vertical":
        plt.ylabel("Frequency", fontweight="normal", style="italic")
        plt.xlabel(column_x, fontweight="normal", style="italic")
    else:
        plt.xlabel("Frequency", fontweight="normal", style="italic")
        plt.ylabel(column_x, fontweight="normal", style="italic")
    plt.title(f"Histogram {column_x}", fontweight="normal", style="italic")

    axs = plt.gca()
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        axs.spines[s].set_visible(False)

    # Add padding between axes and labels
    axs.xaxis.set_tick_params(pad=5)
    axs.yaxis.set_tick_params(pad=10)

    # show number
    # size = 10
    axs.bar_label(axs.containers[0], size=8)

    # Add x, y gridlines
    axs.grid(b=True, color='grey',
             linestyle='-.', linewidth=0.5,
             alpha=0.6)

    # TODO: maybe color bars differently, for now: leave it like that (just one color)
    # Color according to colormap
    # cm = plt.cm.get_cmap('RdYlBu_r')
    # # summer
    # # tab20c
    # # cm = plt.cm.get_cmap('Set1')
    #
    # # To normalize your values
    # col = (n - n.min()) / (n.max() - n.min())
    # for c, p in zip(col, patches):
    #     plt.setp(p, 'facecolor', cm(c))

    country = df.Country.unique()[0]

    output_path = f"../output/{country}/plots/histograms"
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    title = f"{output_path}/{column_x}-hist{png_appendix}.png"

    # make sure that everything is visible in layout
    plt.tight_layout()
    # alternatively:
    # plt.subplots_adjust(left=0.5, right=0.5)

    plt.savefig(title)
    plt.show()


def histogram_pandas(df, column, commodity, year, month, bins=20, save_fig=False):
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


# ----------------------------------------------------------------------------------------------------------------------
# MISSINGS
# ----------------------------------------------------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------------------------------------------------
# MAPS
# ----------------------------------------------------------------------------------------------------------------------

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
    gdf_markets_with_admin2 = gpd.sjoin(malawi_adm2, gdf_final_markets.to_crs(crs=crs_adm2), how="inner",
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

    # PLOT BASED ON DISTRICT
    gdf_merged.plot(column="District", ax=ax, legend=True, legend_kwds={"loc": "lower left",
                                                                        "bbox_to_anchor": (1.1, -0.105),
                                                                        "fontsize": "x-small",
                                                                        "title": "District"},
                    cmap=cmap, edgecolor="darkgreen",
                    missing_kwds={
                        "color": "lightgrey",
                        "edgecolor": "red",
                        "hatch": "///",
                        "label": "Missing values"
                    })

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


def plot_country_adm2_price_spei(df_final, year, month, commodity=None,
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
    gdf_markets_with_admin2 = gpd.sjoin(malawi_adm2, gdf_final_markets.to_crs(crs=crs_adm2), how="inner",
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

    # PLOT BASED ON SPEI
    # cmap = "Spectral"
    cmap = "coolwarm_r"
    # cmap = "bwr_r"
    cmap = "seismic_r"
    gdf_merged.plot(column="Spei", ax=ax, legend=True,
                    cmap=cmap, edgecolor="darkgreen",
                    missing_kwds={
                        "color": "lightgrey",
                        "edgecolor": "red",
                        "hatch": "///",
                        "label": "Missing values"
                    })

    # manually add legend for prices back
    ax.add_artist(legend_prices)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{year}, {month}")

    # TODO: check
    # set range of colorbar
    plt.clim((-3, 3))

    if commodity is not None:
        plt.suptitle(f"Malawi - {commodity}")
        plt.savefig(f"{output_path_maps}/{country}-Districts-Adm2-Prices-{year}-{month}-{commodity}.png")
    else:
        plt.suptitle("Malawi - Markets")
        plt.savefig(f"{output_path_maps}/{country}-Districts-Adm2-Prices-{year}-{month}-SPEI.png")

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
