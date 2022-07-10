"""
VISUALIZATION
-------------

"""

# TODO: color regions
# TODO: visualize prices (e.g. color scale, kdeplot)
# from mpl_toolkits.basemap import Basemap

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



def plot_malawi(df_final):
    """

    Structure Shape File
    NAME_0: Name of country
    NAME_1: Name of bigger regions
    NAME_2: Name of cities


    References
    ----------
    Data source/ shape files extracted via: https://www.diva-gis.org/datadown
    :return:
    """

    # VERSION 1 -> Shapefile via geopandas (but only country boundaries, nothing
    # more fine granular
    world_filepath = gpd.datasets.get_path('naturalearth_lowres')
    world = gpd.read_file(world_filepath)
    print(world.head())
    print(world.columns)

    world_malawi = world[world.name == "Malawi"]
    # world_malawi.plot()

    # VERSION 2 -> Shae files via external source (e.g. diva-gis)
    # pure country boundaries
    malawi_adm0 = gpd.read_file("../input/Malawi/maps/diva-gis/MWI_adm/MWI_adm0.shp")
    malawi_adm1 = gpd.read_file("../input/Malawi/maps/diva-gis/MWI_adm/MWI_adm1.shp")
    malawi_adm2 = gpd.read_file("../input/Malawi/maps/diva-gis/MWI_adm/MWI_adm2.shp")
    malawi_adm3 = gpd.read_file("../input/Malawi/maps/diva-gis/MWI_adm/MWI_adm3.shp")

    malawi_adm1.rename(columns={"NAME_1" : "Market"}, inplace=True)

    # merge adm1 data (for now) with df_final
    df_final_merged = malawi_adm1.merge(df_final, left_on="Market", right_on="Market")

    # pure country
    # malawi_adm0.plot()
    # adm1 -> regions?
    # malawi_adm1.plot()
    print(malawi_adm3.columns)
    # adm2 -> more finegranular (cities?)
    # malawi_adm2.plot()
    # malawi_adm3.plot()



    # # print(malawi_adm1.NAME_1)
    # print("NAME 0\n", malawi_adm3.NAME_0.unique())
    # print("NAME 1\n", malawi_adm3.NAME_1.unique(), len(malawi_adm3.NAME_1.unique()))
    # print("NAME 2\n",malawi_adm3.NAME_2.unique(), len(malawi_adm3.NAME_2.unique()))
    # print("NAME 3\n",malawi_adm3.NAME_3.unique())
    # # Merge df_final with shape dataset
    #
    # print("ID 0\n", malawi_adm3.ID_0.unique())
    # print("ID 1\n", malawi_adm3.ID_1.unique(), len(malawi_adm3.ID_1.unique()))
    # print("ID 2\n", malawi_adm3.ID_2.unique(), len(malawi_adm3.ID_2.unique()))
    # print("ID 3\n", malawi_adm3.ID_3.unique(), len(malawi_adm3.ID_3.unique()))
    #
    # print("NL NAME 3 3\n", malawi_adm3.NL_NAME_3.unique(), len(malawi_adm3.NL_NAME_3.unique()))

    # print(malawi_adm3)

    # convert regular dataframe to geopandas df
    # gdf_final = gpd.GeoDataFrame(
    #     df_final, geometry=gpd.points_from_xy(df_final.MarketLatitude, df_final.MarketLongitude)
    # )
    gdf_final = gpd.GeoDataFrame(
        df_final, geometry=gpd.points_from_xy(df_final.MarketLongitude, df_final.MarketLatitude)
    )
    ax = gplt.polyplot(malawi_adm1)
    # ax = gplt.polyplot(df_final_merged, hue="Region")
    # ax = gplt.polyplot(malawi_adm1, projection=gcrs.AlbersEqualArea())
    print(df_final_merged.head())
    gplt.choropleth(
        df_final_merged,
        hue="Region",
        edgecolor="white",
        linewidth=1,
        cmap="Greens",
        legend=True,
        scheme="FisherJenks",
        ax=ax
    )

    gplt.pointplot(gdf_final, ax=ax, hue="Region")
    # gplt.pointplot(gdf_final, ax=ax)

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