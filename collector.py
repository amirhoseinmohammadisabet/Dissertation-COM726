import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from netCDF4 import Dataset


PATH_2018 = "Data/sst.wkmean.1981-1989.nc"
PATH_2021 = "Data/sst.wkmean.1990-present.nc"

READ_2018 = Dataset(PATH_2018)
READ_2021 = Dataset(PATH_2021)

# print(READ_2018)
# print(READ_2021)
print(READ_2018.variables.keys())
print(READ_2021.variables.keys())

GENERAL_KEY_LIST = ['sst']
YEAR_NAME_LIST = ["2018","2021"]


LON_2018 = READ_2018.variables["lon"][:]
LAT_2018 = READ_2018.variables["lat"][:]
TIME_2018 = READ_2018.variables["time"][:]
TDROP_2018 = READ_2018.variables["sst"][0,:,:]


LON_2021 = READ_2021.variables["lon"][:]
LAT_2021 = READ_2021.variables["lat"][:]
TIME_2021 = READ_2021.variables["time"][:]
TDROP_2021 = READ_2021.variables["sst"][0,:,:]


LAT_TDROP_2018 = TDROP_2018[:,0]
LON_TDROP_2018 = TDROP_2018[0,:]

LAT_TDROP_2021 = TDROP_2021[:,0]
LON_TDROP_2021 = TDROP_2021[0,:]

GENERAL_PARAMS_2018 = [TDROP_2018,]
GENERAL_PARAMS_2021 = [TDROP_2021]


plt.style.use("dark_background")


for x_climate_params,x_params_name in zip(GENERAL_PARAMS_2021,GENERAL_KEY_LIST):


    figure = plt.figure(figsize=(12,8))

    axis_func = plt.axes(projection=ccrs.Robinson())
    axis_func.set_global()
    axis_func.coastlines(resolution="110m",linewidth=1)
    axis_func.gridlines(linestyle='--',color='black',linewidth=2)

    plt.contourf(LON_2021, LAT_2021, x_climate_params, transform=ccrs.PlateCarree(),cmap="RdGy")
    color_bar_func = plt.colorbar(ax=axis_func, orientation="horizontal",aspect=14, shrink=0.8, extend="max")
    color_bar_func.ax.tick_params(labelsize=10)

    plt.title(READ_2021.variables[x_params_name].long_name + " " + "2021")
    plt.tight_layout()
    plt.show()