"""
VISUALIZATION
-------------

"""

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

    # pure country
    # malawi_adm0.plot()
    # adm1 -> regions?
    # malawi_adm1.plot()
    print(malawi_adm1.columns)
    # adm2 -> more finegranular (cities?)
    # malawi_adm2.plot()
    # malawi_adm3.plot()



    # convert regular dataframe to geopandas df
    # gdf_final = gpd.GeoDataFrame(
    #     df_final, geometry=gpd.points_from_xy(df_final.MarketLatitude, df_final.MarketLongitude)
    # )
    gdf_final = gpd.GeoDataFrame(
        df_final, geometry=gpd.points_from_xy(df_final.MarketLongitude, df_final.MarketLatitude)
    )
    ax = gplt.polyplot(malawi_adm1)
    # ax = gplt.polyplot(malawi_adm1, projection=gcrs.AlbersEqualArea())

    gplt.pointplot(gdf_final, ax=ax, hue="Region")
    # gplt.pointplot(gdf_final, ax=ax)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.suptitle("Malawi")
    plt.title("Markets (per Region)")
    plt.legend()
    plt.savefig("../output/Malawi/plots/Map-Markets.png")
    plt.show()
