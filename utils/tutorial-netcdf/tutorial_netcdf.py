import rioxarray
import matplotlib.pyplot as plt
import xarray

import netCDF4 as nc


def tutorial_read_plot_netcdf():
    """
    Source: https://opensourceoptions.com/blog/a-better-way-to-read-netcdf-with-python-rioxarray/
    :return:
    """
    path_netcdf = "./pr_2020.nc"
    xr = rioxarray.open_rasterio(path_netcdf)

    # read attributes
    # print(xr.attrs)

    # change units to just one measure (as the same for every day, instead of repeating it 365 times)
    xr.attrs['units'] = 'mm'

    # Check structure of data
    # Dimensions: day, y, x
    print(xr.dims)
    # Attributes: note: scale_factor = 0.1 (you need to multiply data by 0.1 to get them in correct units)
    print(xr.attrs)
    print(xr.coords)

    # ------------------------------------------------------------------------------------------------------------------
    # INDEXING
    # ------------------------------------------------------------------------------------------------------------------
    # Look into them
    # Example array for rows 100-105, columns 100-105 for 113th day of the year
    print(xr[112, 100:105, 100:105])

    # all data on a specfic day:
    print(xr[0, :, :])

    # all data for a specific location (row, col)
    print(xr[:, 100, 100])

    # Plotting netCDF data with xarray:
    # plot precipitation for year 2020 at location 100, 100
    xr[:, 100, 100].plot()

    plt.show()

    # plot map of entire US precipitation on 113th day of 2020

    # Step1: exclude missing values
    xr_masked = xr.where(xr != xr.attrs['missing_value'])
    xr_masked[112, :, :].plot()

    plt.show()

    print(f"Average precipitation 2020 US (scaled): {xr_masked.mean()}")
    print(f"Average precipitation 2020 US (real value): {xr_masked.mean() * xr.attrs['scale_factor']}")

    # plot average precipitation accross US
    xr_masked.mean(dim="day").plot()
    plt.show()

    # more useful to plot total annual precipitation
    (xr_masked.sum(dim="day") * xr.attrs["scale_factor"]).plot()
    plt.show()


def read_netCDF():
    """
    Tutorial provided in
    https://towardsdatascience.com/read-netcdf-data-with-python-901f7ff61648
    :return:
    """
    fn = "pr_2020.nc"
    # load dataset
    ds = nc.Dataset(fn)
    print(ds)

    # netCDF 3 parts;
    # metadata
    # dimensions
    # variables (contain both metadata and data)

    # variables we're interested in: lat, lon, precipitation_amount
    # file contains entire year (dimension of day = 366)

    # access metadata via python dict
    print(ds.__dict__)

    # access any metadata via dict key
    print(ds.__dict__["geospatial_lon_max"])

    # access to dimensions is similar
    # metadata for each dimension stored in dimension class
    for dim in ds.dimensions.values():
        print(dim)

    # information on just one dimension:
    print(ds.dimensions["lon"])

    # Access METADATA
    for var in ds.variables.values():
        print(var)

    # access information for specific VARIABLE
    print(ds["precipitation_amount"])

    # access data VALUES
    prcp = ds["precipitation_amount"][:]
    print(prcp)




# tutorial_read_plot_netcdf()
read_netCDF()

