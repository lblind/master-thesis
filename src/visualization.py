"""
VISUALIZATION
-------------
All source code for visualizations
"""

import os
import geoplot as gplt
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import preprocessing as preproc
import utils
import seaborn as sns
import missingno as msgo
import analysis as stats
import plotly.express as px
import geoplot.crs as gcrs
import imageio
import pathlib
import matplotlib.animation as animation
import matplotlib.dates as mdates
import folium
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mapclassify as mc


# ----------------------------------------------------------------------------------------------------------------------
# DMD results
# ----------------------------------------------------------------------------------------------------------------------

def plot_abs_error_matrix(abs_error, country, rank, algorithm, commodity, transposed=False):
    """
    Plots for the absolute error matrix & the mean error over time

    :param abs_error: pd.DataFrame
        Absolute error matrix
    :param country: str
        Country for which the DMD has been computed
    :param rank: int
        Rank of the underlying SVD
    :param algorithm: str
        PyDMD algorithm set (Default: Base)
    :param commodity: str
        Commodity for which the DMD has been computed
    :param transposed: boolean
        Whether or not the snapshot matrix has been transposed
    :return: Nothing
        The resulting graphics are stored in the respective output folder

    """
    output_dir_dmd = f"../output/{country}/plots/dmd"
    if os.path.exists(output_dir_dmd) is False:
        os.makedirs(output_dir_dmd)

    plt.imshow(abs_error)
    plt.title(f"Absolute Error {commodity} ({country})")
    # plt.suptitle(f"Absolute error {commodity} ({country})")
    # plt.title("Observed - Reconstructed Time Series")
    if transposed:
        plt.xlabel("Markets $M_i$")
        plt.ylabel("Time $t_k$")
        plt.colorbar(orientation="vertical")
    else:
        plt.ylabel("Markets $M_i$")
        plt.xlabel("Time $t_k$")
        plt.colorbar(orientation="horizontal")

    plt.savefig(f"{output_dir_dmd}/{algorithm}-error-matrix-r-{rank}-T-{transposed}.png")

    plt.show()

    if transposed:
        t = np.linspace(1, abs_error.shape[0], abs_error.shape[0])
        # calculate mean per row / month
        mean_err = abs_error.mean(axis=1)

    else:
        t = np.linspace(1, abs_error.shape[1], abs_error.shape[1])
        # calculate mean per column / month
        mean_err = abs_error.mean(axis=0)

    plt.title("Mean error over time")
    plt.plot(t, mean_err)
    plt.ylabel("Mean error per $t_k$")
    plt.xlabel("Time $t_k$")

    plt.savefig(f"{output_dir_dmd}/{algorithm}-error-over-time-r-{rank}-T-{transposed}.png")
    plt.show()



def plot_dmd_results(dmd, country, commodity, svd_rank, algorithm="base", transposed=True):
    """
    Plots the Results of the DMD algorith,

    :param dmd: pyDMD
        Trained pyDMD object
    :param country: str
        Country for which the DMD has been computed
    :param commodity: str
        Commodity for which the DMD has been computed
    :param svd_rank: int
        Rank of the underlying SVD
    :param algorithm: str
        Algorithm set for the PyDMD implementation
    :param transposed: boolean
        Whether or not the snapshot matrix has been transposed
    :return: Nothing
        Stores the results in the respective folder
    """
    # make sure output dir exists
    output_path = f"../output/{country}/plots/dmd/svd-rank-{svd_rank}"

    if transposed:
        output_path += "/transposed"

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    rank = dmd.eigs.shape[0]

    # png_appendix_eigs = "-T" if transposed else ""
    # I have to put rank as integer, as floating point not accepted by dmd.plot_eigs
    dmd.plot_eigs(filename=f"{output_path}/{commodity}-{algorithm}-eigs-{rank}-T-{transposed}.png")
    plt.show()

    # 51 = number of markets
    x = np.linspace(1, dmd.snapshots.shape[0], dmd.snapshots.shape[0])
    # 216 = number of time steps
    t = np.linspace(1, dmd.snapshots.shape[1], dmd.snapshots.shape[1])

    # MODES
    # ------------------------------------
    for i, mode in enumerate(dmd.modes.T):
        plt.plot(x, mode.real, label=f"#{i + 1}")

    if transposed:
        plt.ylabel("Markets $M_i$")
        plt.xlabel("Value")
    else:
        plt.xlabel("Markets $M_i$")
        plt.ylabel("Value")

    plt.title(f"Modes {commodity} ({country})")
    plt.legend()
    plt.savefig(f"{output_path}/{commodity}-{algorithm}-modes-{svd_rank}-T-{transposed}.png")
    plt.show()

    # DYNAMICS OF MODES
    # ------------------------------------
    print(f"Dynamics: {dmd.dynamics}")
    for i, dynamic in enumerate(dmd.dynamics):
        print("Dynamic:", dynamic)
        plt.plot(t, dynamic, label=f"#{i + 1}")

    plt.title(f"Dynamics of modes {commodity} ({country})")

    if transposed:
        plt.xlabel("Markets $M_i$")
    else:
        plt.xlabel("Time $t_k$")
    plt.tight_layout()
    # plt.legend(loc="lower left", bbox_to_anchor=(0.7, 0.5))
    plt.legend()
    plt.title(f"Dynamics of DMD modes {commodity} ({country})")
    plt.savefig(f"{output_path}/{commodity}-{algorithm}-dynamics-{svd_rank}-T-{transposed}.png")
    plt.show()

    # SNAPSHOT MATRIX (ORIGINAL)
    # ------------------------------------
    plt.imshow(dmd.snapshots)

    if transposed:
        plt.xlabel("Markets $M_i$")
        plt.ylabel("Time $t_k$")
        plt.colorbar(orientation="vertical")
    else:
        plt.ylabel("Markets $M_i$")
        plt.xlabel("Time $t_k$")
        plt.colorbar(orientation="horizontal")
    plt.title("Original Snapshot Matrix")
    plt.savefig(f"{output_path}/{algorithm}-original-snapshot-matrix-{svd_rank}-T-{transposed}.png")
    plt.show()

    # SNAPSHOT MATRIX (RECONSTRUCTED)
    # ------------------------------------
    plt.imshow(dmd.reconstructed_data.real)

    if transposed:
        plt.xlabel("Markets $M_i$")
        plt.ylabel("Time $t_k$")
        plt.colorbar(orientation="vertical")
    else:
        plt.ylabel("Markets $M_i$")
        plt.xlabel("Time $t_k$")
        plt.colorbar(orientation="horizontal")
    plt.title(f"Reconstructed Matrix {commodity} ({country})")
    plt.savefig(f"{output_path}/{commodity}-{algorithm}-reconstructed-matrix-{svd_rank}-T-{transposed}.png")
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# (LINE) PLOTS
# ----------------------------------------------------------------------------------------------------------------------

def line_plot_spei(df_spei, country, show=True, alpha=0.5):
    """
    Creates a line plot for the development of the SPEI indicator over time

    :param df_spei: pandas.DataFrame
        Dataframe containing the SPEI measure points
    :param country: str
        COuntry to which the data belongs
    :param show: boolean
        Whether or not to show the graphics
    :param alpha: float
        Configuration parameter to set the opacity of the drawings
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
    Creates a line plot for the mean/ min/ max values per delta t (month)
    including the boundary lines for floods and droughts (|SPEI| > 1)

    :param df_spei: pandas.DataFrame
        Dataframe containing the SPEI observations
    :param country: str
        Name of the country the data belongs to
    :param time: str
        Aggregation of time that should be used (Mean per year/month/etc.)
    :param show: boolean
        Whether or not to show the results
    :param alpha: float
        COnfiguration parameter to set the opacity of the drawings
    :return: None
        Stores the results in the respective folder.
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
    Iterates over all unique regions for one country and creates a line plot
    for those

    :param df_wfp_and_spei: pandas.DataFrame
        Dataframe containing both WFP and SPEI indciators
    :param show: boolean
        Whether or not to show the plot
    :param alpha: float
        CConfiguration parameter to set the opacity
    :return: None
        Stores the results in the respective folder
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


def scatter_adj_prices_all_commodities(df_wfp, alpha=0.5, preproc_step=4):
    """
    Scatters the inflation-adjusted prices for all commodities

    :param df_wfp: pandas.DataFrame
        Dataset containing the WFP prices
    :param alpha: float
        Configuration parameter to set the opacity
    :param preproc_step: int
        Number of the preprocessing step tha correspond to the datset ("state" of the dataset)
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

        plt.savefig(f"{output_dir}/{commodity}-scatter-adj-prices-STEP{preproc_step}.png")
        plt.show()


def scatter_adj_prices_all_commodities_droughts(df_wfp, alpha=0.5, preproc_step=4):
    """
    Creates a 3x1 plot plotting

    - both drought and non-drought points in one fig
    - only the drought points
    - only the non-drought points

    :param df_wfp: pandas.DataFrame
        Input dataset
    :param alpha: float
        Configuration paramter to control the opacity
    :param preproc_step: int
        Number of preprocessing step the dataset resulted of
    :return: None
        Plots will be stored in output folder
    """
    country = df_wfp.Country.unique()[0]

    # merge SPEI to price data
    df_wfp = utils.merge_drought_to_df_wfp(df_wfp)

    currency = df_wfp.Currency.unique()[0]
    for commodity in df_wfp.Commodity.unique():
        # Three subplots per commodity
        fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)

        df_wfp_commodity = df_wfp[df_wfp.Commodity == commodity]

        df_wfp_commodity_drought = df_wfp_commodity[df_wfp_commodity.Spei < -1]
        df_wfp_commodity_no_drought = df_wfp_commodity[df_wfp_commodity.Spei >= -1]

        # TODO: somehow this indexing didn't work for the non-drought case (assumption: bug, check back later)
        # df_wfp_commodity_drought = df_wfp_commodity[df_wfp_commodity.Drought]
        # df_wfp_commodity_no_drought = df_wfp_commodity[~df_wfp_commodity["Drought"]]

        ax[1].scatter(df_wfp_commodity_drought.TimeWFP, df_wfp_commodity_drought["AdjPrice"], label="Drought",
                      color="red",
                      alpha=alpha)
        ax[1].set_title("Drought")

        # Don't display xlabel for these
        ax[1].set(xlabel=None)
        ax[2].set(xlabel=None)

        for i in range(0, 2):
            ax[i].get_xaxis().set_visible(False)
            # ax[i].get_xaxis().set_xlabel(None)

        # Set grid
        for i in range(3):
            # Add x, y gridlines

            ax[i].get_xaxis().grid(b=True, color='grey',
                                   linestyle='-.', linewidth=0.5,
                                   alpha=0.6)
            ax[i].get_yaxis().grid(b=True, color='grey',
                                   linestyle='-.', linewidth=0.5,
                                   alpha=0.6)

        ax[0].scatter(df_wfp_commodity_drought.TimeWFP, df_wfp_commodity_drought["AdjPrice"], label="Drought",
                      color="red",
                      alpha=alpha)
        ax[0].scatter(df_wfp_commodity_no_drought.TimeWFP, df_wfp_commodity_no_drought["AdjPrice"], label="No Drought",
                      alpha=alpha)
        ax[0].set_title("Both")

        # plt.vlines(df_wfp_commodity_drought.TimeWFP, 0, 4000)
        ax[2].scatter(df_wfp_commodity_no_drought.TimeWFP, df_wfp_commodity_no_drought["AdjPrice"], label="No Drought",
                      alpha=alpha)
        ax[2].set_title("No drought")

        fig.suptitle(f"(Inflation-Adjusted) Price Distribution - {commodity}")

        # plt.title(f"Commodity: {commodity}")
        ax[2].set_xlabel("Time")
        ax[1].set_ylabel(f"Price [{currency}]")
        # plt.legend()

        # option 1)
        plt.tight_layout()

        # option 2)
        plt.subplots_adjust(left=0.1,
                            bottom=0.15,
                            right=0.9,
                            top=0.85,
                            wspace=0.4,
                            hspace=0.4)

        # format the axis
        # just display the year
        # plt.gca().xaxis.set_major_formatter(DateFormatter("%Y"))
        # display year and month
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y, %m"))
        plt.xticks(rotation=30)

        # Make sure Output dir exists
        country = df_wfp.Country.unique()[0]
        output_dir = f"../output/{country}/plots/scatter-plots/{commodity}"
        if os.path.exists(output_dir) is False:
            os.makedirs(output_dir)

        plt.savefig(f"{output_dir}/{commodity}-scatter-adj-prices-STEP{preproc_step}-drought.png")
        plt.show()


def scatter_adj_prices_per_region_one_fig(df_wfp, title_appendix="", png_appendix="", c=None):
    """
    Creates a #unique regions x 1 figure
    where one figure contains only the observations corresponding to that unique region

    :param df_wfp: pandas.DataFrame
        Input dataset
    :param title_appendix: str
        Appendix to add to the default title
    :param png_appendix:  str
        Appendix to add to the output png image
    :param c: str
        Specific color to use for plot
    :return: None
        Plots will be stored in respective output folder
    """

    n_regions = len(df_wfp.Region.unique())

    for commodity in df_wfp.Commodity.unique():
        df_wfp_commodity = df_wfp[df_wfp.Commodity == commodity]

        fig, ax = plt.subplots(n_regions, 1, sharex=True, sharey=True)

        for i, region in enumerate(df_wfp_commodity.Region.unique()):
            df_wfp_commodity_region = df_wfp_commodity[df_wfp_commodity.Region == region]
            # ax[0, i].scatter(df_wfp.TimeWFP, df_wfp.AdjPrice, c = ["red" if df_wfp.Drought else "blue"], alpha=0.5)

            if c is not None:
                ax[i].scatter(df_wfp_commodity_region.TimeWFP, df_wfp_commodity_region.AdjPrice, alpha=0.5, c=c)
            else:
                ax[i].scatter(df_wfp_commodity_region.TimeWFP, df_wfp_commodity_region.AdjPrice, alpha=0.5)
            ax[i].set_title(f"{region}")

            # Add padding between axes and labels
            ax[i].get_xaxis().set_tick_params(pad=5)
            ax[i].get_yaxis().set_tick_params(pad=5)

            # Don't display xlabel for these
            if i != n_regions - 1:
                ax[i].get_xaxis().set_visible(False)
                # ax[i].get_xaxis().set_xlabel(None)

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

        # Add padding between axes and labels
        plt.gca().xaxis.set_tick_params(pad=5)
        plt.gca().yaxis.set_tick_params(pad=10)

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

        fig.suptitle(f"Commodity: {commodity}{title_appendix}")

        # Make sure Output dir exists
        country = df_wfp.Country.unique()[0]
        output_dir = f"../output/{country}/plots/scatter-plots/{commodity}"
        output_dir_extrapolation = output_dir + "/extrapolation"
        if os.path.exists(output_dir_extrapolation) is False:
            os.makedirs(output_dir_extrapolation)
        plt.savefig(f"{output_dir}/{commodity}-scatter-adj-prices-regions-one-fig{png_appendix}.png")
        plt.show()


def scatter_adj_price_per_region_drought_one_fig(df_wfp, c="red"):
    """
    Creates a # unique regions x 1 figure
    For each region, plots only the observations where a drought has occured

    :param df_wfp: pandas.DataFrame
        Input dataset
    :param c: str
        Color to use for plotting the (drought) points
    :return: None
        Output will be stored in respective folder
    """
    # merge drought data
    df_wfp_droughts = utils.merge_drought_to_df_wfp(df_wfp)

    # just extract the slice where a drought occured
    df_wfp_droughts = df_wfp_droughts[df_wfp_droughts.Drought]

    scatter_adj_prices_per_region_one_fig(df_wfp_droughts, title_appendix=" (Drought occured)", png_appendix="-drought",
                                          c=c)


def scatter_adj_prices_per_region(df_region, commodity, currency, show=False, alpha=0.5):
    """
    For the dataframe containing only the data for one region,
    plot the inlfation-adjusted prices

    :param df_region: pandas.DataFrame
        Dataset containing only observations for one region
    :param commodity: str
        Commodity for which to plot the data
    :param currency: str
        Name of the currency used
    :param show: boolean
        Whether or not to show the results
    :param alpha: float
        Configuration parameter to set the opacity
    :return: None
        Results are stored in respective output folder
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
    """
    For the dataset corresponding to a unique region and commodity, plot the extrapolated points
    (and the original ones)

    :param df_region: pandas.DataFrame
        Dataset containing only the subset for one unique region & commodity
    :param df_region_extrapolated:  pandas.DataFrame
        Dataset containing only the subset for one unique region & commodity with the extrapolated data
    :param commodity: str
        Name of the commodity this dataset belongs to
    :param currency: str
        Name of the currency the prices belong ot
    :param interpolation_method: str
        Used interpolation method (quadratic, linear, ...)
    :param order: int
        Order of the interpolation function used
    :param show: boolean
        Whether or not to show the results
    :param alpha: float
        Configuration parameter for the opacity
    :param show_original_points: boolean
        Whether or not to show the original (non-extrapolated) points as well
    :return: None
        The results will be stored in the respective directory
    """
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


def scatter_spikes_per_commodity(df):
    """
    For each commodity in df:
    Plot spikes in magenta and non spikes in blue

    :param df: pandas.DataFrame
        Input dataset
    :return: None
        The results will be stored in the respective directory
    """
    country = df.Country.unique()[0]
    currency = df.Currency.unique()[0]
    output_path = f"../output/{country}/plots/spikes"

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    for commodity in df.Commodity.unique():
        df_commodity = df[df.Commodity == commodity]
        # only consider non-nan entries
        df_commodity = df_commodity[df_commodity.SpikeAdjPrice.notna()]
        print(df_commodity.SpikeAdjPrice.unique())

        # extract part where a spike occured
        df_commodity_spike = df_commodity[df_commodity.SpikeAdjPrice]
        # can't invert the mask, because there might still be some nan values in it (non-extrapolated part)
        # df_commodity_no_spike = df_commodity[~df_commodity.SpikeAdjPrice]
        df_commodity_no_spike = df_commodity[df_commodity.DevMean <= 0]

        plt.scatter(df_commodity_spike.TimeWFP, df_commodity_spike.AdjPrice, c="m", marker="*", alpha=0.5,
                    label="Spike (> mean)")

        plt.scatter(df_commodity_no_spike.TimeWFP, df_commodity_no_spike.AdjPrice, c="blue", alpha=0.5,
                    label="No spike")
        plt.title("Spikes")
        plt.suptitle(f"(Inflation-Adjusted) Price Distribution - Commodity: {commodity}")

        plt.hlines(df_commodity.MeanAdjPrice.unique()[0], df_commodity.TimeWFP.min(), df_commodity.TimeWFP.max(),
                   label="Mean", colors="c", linewidth=2)

        # # if you want to plot the upper quartile as well -> outcomment
        # # upper quartile
        # upper_quartile = np.nanpercentile(df_commodity.AdjPrice, 75)
        # plt.hlines(upper_quartile, df_commodity.TimeWFP.min(), df_commodity.TimeWFP.max(),
        #            label="Upper quartile", colors="c", linewidth=2)

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(f"Price [{currency}]")
        plt.grid(b=True, color='grey',
                 linestyle='-.', linewidth=0.5,
                 alpha=0.7)
        # display year and month
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y, %m"))

        # Add padding between axes and labels
        plt.gca().xaxis.set_tick_params(pad=5)
        plt.gca().yaxis.set_tick_params(pad=10)

        # plt.xticks(df_commodity.TimeWFP.unique(), rotation=30)
        # one for each year
        # print(df_commodity.groupby("Year")["TimeWFP"])

        # always plot january on x-axis (one tick for each year)

        # # one tick per year (January)
        dates_xticks = df_commodity[df_commodity.Month == 1]["TimeWFP"].unique()
        # two ticks per year (January and July)
        # dates_xticks = df_commodity[df_commodity.Month.isin([1, 7])]["TimeWFP"].unique()

        # two ticks per year
        plt.xticks(dates_xticks, rotation=90)

        plt.tight_layout()

        # plt.xticks(df_commodity.groupby("Year")["TimeWFP"][0], rotation=30)
        plt.savefig(f"{output_path}/{commodity}-Spikes.png")
        plt.show()

        # --------------------------------------------------------------------------------------------------------------
        # PLOT ONLY SPIKES / Data > mean
        # --------------------------------------------------------------------------------------------------------------
        # one tick per year (January)
        dates_xticks = df_commodity[df_commodity.Month == 1]["TimeWFP"].unique()
        plt.xticks(dates_xticks, rotation=90)
        plt.scatter(df_commodity_spike.TimeWFP, df_commodity_spike.AdjPrice, c="m", marker="*", alpha=0.5,
                    label="Spike (> mean)")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(f"Price [{currency}]")
        plt.grid(b=True, color='grey',
                 linestyle='-.', linewidth=0.5,
                 alpha=0.7)
        # display year and month
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y, %m"))

        # Add padding between axes and labels
        plt.gca().xaxis.set_tick_params(pad=5)
        plt.gca().yaxis.set_tick_params(pad=10)
        plt.savefig(f"{output_path}/{commodity}-Spikes-only-spike-1-tick.png")
        plt.show()

        dates_xticks = df_commodity[df_commodity.Month.isin([1, 7])]["TimeWFP"].unique()
        # o ticks per year
        plt.xticks(dates_xticks, rotation=90)
        plt.scatter(df_commodity_spike.TimeWFP, df_commodity_spike.AdjPrice, c="m", marker="*", alpha=0.5,
                    label="Spike (> mean)")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(f"Price [{currency}]")
        plt.grid(b=True, color='grey',
                 linestyle='-.', linewidth=0.5,
                 alpha=0.7)
        # display year and month
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y, %m"))
        # Add padding between axes and labels
        plt.gca().xaxis.set_tick_params(pad=5)
        plt.gca().yaxis.set_tick_params(pad=10)
        plt.savefig(f"{output_path}/{commodity}-Spikes-only-spike-2-ticks.png")
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# BOXPLOTS
# ----------------------------------------------------------------------------------------------------------------------

def boxplot_adj_prices(df, png_appendix="-all-commodities", by="Commodity", suptitle_appendix="", title_appendix=""):
    """
    Plots the boxplot(s) for a dataframe (if indicated by a specific group)

    :param df: pandas.DataFrame
        Input dataset
    :param png_appendix: str
        Appendix for the output image
    :param by: str
        Group for which the boxplots should be computed (one per unique value in that group)
    :param suptitle_appendix: str
        Appendix for the suptitle of the plot
    :param title_appendix: str
        Appendix for the title of the plot
    :return:
    """

    if by is None:
        df.boxplot(column="AdjPrice", grid=True)
    else:
        # boxplot per commodity/ group
        df.boxplot(column="AdjPrice", by=by, grid=True)

    country = df.Country.unique()[0]
    currency = df.Currency.unique()[0]

    output_path = f"../output/{country}/plots/boxplots"

    # if title_appendix is not "":
    if by is None:
        plt.suptitle(f"Boxplot")
    else:
        plt.suptitle(f"Boxplot grouped by {by}{suptitle_appendix}")
        output_path += f"/{by}"

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    plt.title(f"(Inflation-Adjusted) Prices{title_appendix}")
    plt.ylabel(f"Price [{currency}]")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_path}/hist{png_appendix}.png")
    plt.show()


def box_plot_for_all_commodities_by_group(df, by=None):
    """
    Creates boxplots for all unique commodities in a dataframe
    (if indicating within that unique commodity by a soecific group)

    :param df: pandas.DataFrame
        Input dataset
    :param by: str
        if None, boxpots will be created for all commodities (and no further grouping will occur)
        otherwise: Name of the group for which to compute the boxplots (e.g. "Region", "Drought")
    :return: None
        Outputs will be stored in the respective directory
    """
    # merge spei data to df
    df = utils.merge_drought_to_df_wfp(df)
    # classify droughts
    df = preproc.classify_droughts(df)
    for commodity in df.Commodity.unique():
        df_commodity = df[df.Commodity == commodity]

        if by is not None:
            png_appendix = f"-{commodity}-{by}"
        else:
            png_appendix = f"-{commodity}"
        boxplot_adj_prices(df_commodity, png_appendix=png_appendix, by=by,
                           title_appendix=f" - {commodity}")

# ----------------------------------------------------------------------------------------------------------------------
# BAR CHARTS
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# HISTOGRAMS
# ----------------------------------------------------------------------------------------------------------------------


def plot_hist(df, column_x, rotation=45, bins=7, orientation="vertical", rwidth=0.75, png_appendix="",
              title_appendix=""):
    """
    Plots the histogram for a input dataset df and a respective column column_x

    :param df: pandas.DataFrame
        Input dataset
    :param column_x: str
        Name of the column that should be used
    :param rotation: int
        Degree of the rotation of the legend
    :param bins: int
        Number of bins to create
    :param orientation: str (vertical or horizontal)
        Orientation of the histogram (horiontal or normal barchart)
    :param rwidth: float
        Relative width of the bars as a fraction of the bin width
        (For more info cf. matplotlib documentation)
    :param png_appendix: str
        Appendix for the output image
    :param title_appendix: str
        Appendix for the title
    :return: None
        Results will be stored in respective directory

    References
    ----------
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    """
    if column_x not in df:
        raise ValueError(f"Can't plot histogram, as column {column_x} not existent. \n"
                         f"Please revise your definition and use one of the following columns instead:\n{df.columns}")

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
    plt.title(f"Histogram {column_x}{title_appendix}", fontweight="normal", style="italic")

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


def plot_hist_for_all_commodities(df, bins=20):
    """
    For each commodity, extract the respective subset
    and plot the histogram

    :param df: pandas.DataFrame
        Input dataset containing all commodities
    :param bins: int
        Number of bins to create per histogram
    :return: None
        Results are stored in respective folder
    """
    for commodity in df.Commodity.unique():
        df_commodity = df[df.Commodity == commodity]
        plot_hist(df_commodity, column_x="AdjPrice", png_appendix=f"-{commodity}-{bins}", bins=bins,
                  title_appendix=f" - {commodity}")


def histogram_pandas(df, column, commodity, year, month, bins=20, save_fig=False):
    """
    Plots the histogram, but by directly calling the pandas histogram function
    (Different style)

    :param df: pandas.DataFrame
        Input dataset
    :param column: str
        Name of the column to plot the distribution for
    :param commodity: str
        Name of the commodity for which to plot the histogram
    :param year:  int
        Year
    :param month: int
        Month
    :param bins: int
        Number of bins to create
    :param save_fig: boolean
        Whether to store/save the png or not
    :return: None
        Will store the outputs if indicated via save_fig
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

def plot_missings(df_final):
    """
    Plots the number and percent of missing values for the entire dataset

    :param df_final: pandas.DataFrame
        Dataset for which to compute the missing values
    :return: None
        (Results are currently not stored in this implementation)
    """
    msgo.bar(df_final)

    plt.title("Missing values")
    plt.xlabel("Variable")
    plt.ylabel("Percent")
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# MAPS
# ----------------------------------------------------------------------------------------------------------------------

def plot_malawi_regions_adm1(df_final, scatter_markets=True):
    """
    Plots Malawi and colors the map based on administrative level 1 (Region)
    If indicated, plots the markets in the scatterplot as well

    :param df_final: pandas.DataFrame
        Input dataset
    :param scatter_markets: boolean
        Whether or not to plot the markets on the map as well
    :return: None
        The outputs will be stored in the respective folder
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
    cmap = "summer" + "_r"
    # cmap = "summer"

    # ax.set_prop_cycle(color=cmap[1:])
    malawi_adm1.plot(column="Region", ax=ax, legend=True, legend_kwds={"loc": "lower left",
                                                                       "bbox_to_anchor": (0.6, 0.8),
                                                                       "fontsize": "x-small",
                                                                       "title": "Regions"},
                     cmap=cmap)
    # edgecolor="darkgreen"
    # default
    # malawi_adm1.plot(column="Region", ax=ax, legend=True, legend_kwds={"loc": "lower left",
    #                                                                    "bbox_to_anchor": (0.6, 0.8)})

    # plt.tight_layout()

    # convert regular dataframe to geopandas df
    gdf_final_markets = gpd.GeoDataFrame(
        df_final, geometry=gpd.points_from_xy(df_final.MarketLongitude, df_final.MarketLatitude)
    )

    if scatter_markets:
        plt.scatter(df_final.MarketLongitude, df_final.MarketLatitude, c="darkblue", edgecolor="orange",
                    alpha=0.5)

    # gplt.pointplot(gdf_final_markets, ax=ax)
    # gdf_final_markets.plot(kind="scatter", ax=ax)
    # plt.grid()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    title = "Malawi"
    if scatter_markets:
        title += " - Markets"
    plt.suptitle(title)
    # plt.title("Malawi - Regions", loc="left")

    plt.savefig(f"{output_path_maps}/{country}-Regions.png")

    plt.show()


def plot_country_adm2_prices_for_year_month(df_final, year, month, commodity=None,
                                            path_shape_file="../input/Malawi/maps/WFPGeoNode/mwi_bnd_admin2/mwi_bnd_admin2.shp"):
    """
    Plots Malawi and the prices for a specific date/ time entry (year, month)

    :param df_final: pandas.DataFrame
        INput dataset
    :param year: int
        year for which to extract the prices
    :param month: int
        month (in combination with year) for which to extract the prices
    :param commodity: str
        Commodity for which to extract the prices
    :param path_shape_file:str
        Path to required shape file
    :return: None
        Results will be stored in respective directory
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

     Plots Malawi and the prices for a specific date/ time entry (year, month)
     and colors the map based on the average SPEI for each district

    :param df_final: pandas.DataFrame
        INput dataset
    :param year: int
        year for which to extract the prices
    :param month: int
        month (in combination with year) for which to extract the prices
    :param commodity: str
        Commodity for which to extract the prices
    :param path_shape_file:str
        Path to required shape file
    :return: None
        Results will be stored in respective directory
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
        plt.savefig(f"{output_path_maps}/{country}-Districts-Adm2-Prices-{year}-{month}-{commodity}-SPEI.png")
    else:
        plt.suptitle("Malawi - Markets")
        plt.savefig(f"{output_path_maps}/{country}-Districts-Adm2-Prices-{year}-{month}-SPEI.png")

    plt.show()


def plot_malawi_districts_adm2(df_final, plot_markets=True):
    """
    Plots Malawi and colors the map based on administrative level 2 (District)
    If indicated, plots the markets in the scatterplot as well

    :param df_final: pandas.DataFrame
        Input dataset
    :param plot_markets: boolean
        Whether or not to plot the markets on the map as well
    :return: None
        The outputs will be stored in the respective folder
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
                                                                         "fontsize": "x-small",
                                                                         "title": "Districts"},
                     cmap=cmap)

    # spatial join: find the fitting admin 2 for each market
    gdf_markets_with_admin2 = gpd.sjoin(gdf_final_markets.to_crs(crs=crs_adm2), malawi_adm2, how="inner",
                                        predicate="intersects")

    # gdf_markets_with_admin2.plot(kind="geo", column="Region", ax=ax, legend=True, legend_kwds={"loc": "lower left",
    #                                                                    "bbox_to_anchor": (0.6, 0.8)},
    #                  cmap=cmap)

    if plot_markets:
        plt.scatter(df_final.MarketLongitude, df_final.MarketLatitude, c="darkblue", edgecolor="orange")
    # plt.tight_layout()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    title = "Malawi"
    if plot_markets:
        title += " - Markets"

    plt.suptitle(title)
    # plt.title("Malawi - Regions", loc="left")

    plt.savefig(f"{output_path_maps}/{country}-Districts-Adm2.png")

    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# CORRELATIONS / Heatmap
# ----------------------------------------------------------------------------------------------------------------------
def plot_correlation_matrix(df):
    """
    Makes a heatplot for the correlation matrix
    Correlations have been computed via Stata

    :param df: pandas.DataFrame
        input dataframe
    :return: None
        Results will be stored in respective output folder.
    """
    country = df.Country.unique()[0]
    output_path_corr = f"../output/{country}/plots/corr"
    if os.path.exists(output_path_corr) is False:
        os.makedirs(output_path_corr)

    df_corr = pd.DataFrame({
        "Adjusted Price": [1, -0.0024, -0.0255, -0.1291, -0.0666, 0.0975],
        "Region": [-0.0024, 1, -0.0683, 0.0171, 0.0415, -0.0179],
        "Market": [-0.0255, -0.0683, 1, 0.0077, 0.0061, -0.0109],
        "Commodity": [-0.1291, 0.0171, 0.0077, 1, 0.0083, 0.0045],
        "Drought": [-0.0666, 0.0415, 0.0061, 0.0083, 1, -0.6751],
        "SPEI": [0.0975, -0.0179, -0.0109, 0.0045, -0.6751, 1]
    }, ["Adjusted Price", "Region", "Market", "Commodity", "Drought", "SPEI"])

    sns.heatmap(df_corr, annot=True, cmap=plt.cm.Reds)
    plt.tight_layout()
    plt.title("Correlation matrix")

    plt.savefig(f"{output_path_corr}/corr_matrix.png")
    plt.show()
