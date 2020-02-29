""" TELECONNECTIONS """

# This is a part of a class project for ATMS 597, Spring 2020, at the University of Illinois at Urbana Champaign.
# Create code using python xarray to organize and reduce climate data. 
# The goal of this analysis will be to detect global atmospheric circulation patterns (or teleconnections) 
# associated with extreme daily precipitation in a certain part of the globe. 

# Created by Puja Roy, Joyce Yang and Jun Zhang. 

#################################################################################

# install necessary packages 

%pylab inline
!pip install netcdf4
!pip install pydap
!pip install nc-time-axis
!pip install wget

# import necessary packages

import xarray as xr
import nc_time_axis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import wget
import glob

# mount the drive
from google.colab import drive
drive.mount('/content/gdrive/')

# Download the data from the NCEI-NOAA server to the google drive shared folder
wget --recursive --no-parent -P '/content/gdrive/My Drive/ATMS_597_Project_3_Group_D/Data/'https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/'

# creating an array for the number of years for downloading data
n = np.arange(1996,2020,1) 

# creating a Xarray master dataset DS containing netcdf files for each year

DS = []
DS_slice = []

# For loop to load in the data files from the shared folder

for i in range(0,len(n)):
    print(n[i])
    data_path = '/content/gdrive/My Drive/ATMS_597_Project_3_Group_D/Data/www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/'+str(n[i])+'/*.nc'
    DS_slice = xr.open_mfdataset(data_path,engine='netcdf4',combine='nested',concat_dim='time')
    DS.append(DS_slice['precip'])

# Make three bunches of lists for the years (trying to merge all the years together would crash the system!!)

list_DS1 = [DS[0],DS[1],DS[2],DS[3],DS[4],DS[5],DS[6],DS[7],DS[8],DS[9]]
list_DS2 = [DS[10],DS[11],DS[12],DS[13],DS[14],DS[15],DS[16]]
list_DS3 = [DS[17],DS[18],DS[19],DS[20],DS[21],DS[22],DS[23]]

# Merge the sub groups of datasets
Merged_DS1 = xr.merge(list_DS1).load()
Merged_DS1.to_netcdf('/content/gdrive/My Drive/ATMS_597_Project_3_Group_D/Merged_DS1.nc')

Merged_DS2 = xr.merge(list_DS2).load()
Merged_DS2.to_netcdf('/content/gdrive/My Drive/ATMS_597_Project_3_Group_D/Merged_DS2.nc')

Merged_DS3 = xr.merge(list_DS3).load()
Merged_DS3.to_netcdf('/content/gdrive/My Drive/ATMS_597_Project_3_Group_D/Merged_DS3.nc')


# Merge the merges
Merged_DS = xr.merge([Merged_DS1,Merged_DS2,Merged_DS3]).load()
Merged_DS.to_netcdf('/content/gdrive/My Drive/ATMS_597_Project_3_Group_D/Merged_DS.nc')

Merged_DS_hilo = Merged_DS.sel(longitude=-155.0868+360,latitude=19.7241, method='nearest') # Lat Long of Hilo
Merged_DS_hilo.to_netcdf('/content/gdrive/My Drive/ATMS_597_Project_3_Group_D/Merged_DS_hilo.nc')

# Converting Xarray Dataset 'Merged_DS_hilo' to a Pandas Dataframe 'DF_Hilo' 
# And then, selecting data from only the three months - Oct, Nov and Dec - that was alotted to our group

DF_Hilo = Merged_DS_hilo.to_dataframe()
DF_hilo_OND = DF_Hilo[(DF_Hilo.index.month==10) | (DF_Hilo.index.month==11) | (DF_Hilo.index.month==12)]

# Adding another column "Time" to the dataframe and viewing a certain section of the dataframe

DF_hilo_OND['Time'] = DF_hilo_OND.index

# 95 percentile value of rainfall for OND for Hilo = 9.157839012145992 mm

perc_95 = DF_hilo_OND['precip'].quantile(0.95)
print("95 percentile value : ", perc_95)  

print("OND Hilo Data : \n ", DF_hilo_OND)  

# selecting rows based on condition 
Xtreme_days = DF_hilo_OND.loc[DF_hilo_OND['precip'] >= perc_95] 

print('\n OND Extreme Precip Days - Hilo Data :\n', Xtreme_days )

# Save the list of Extreme Prceipitation days as a csv file

Xtreme_days.to_csv('/content/gdrive/My Drive/ATMS_597_Project_3_Group_D/Xtreme_Precip_days_Hilo.csv')

# Plot CDF and point to 95th percentile

Precip_S_hilo = pd.Series(DF_hilo_OND['precip'])
Precip_S_hilo  = Precip_S_hilo.sort_values()
cum_dist = np.linspace(0.,1.,len(Precip_S_hilo))
Precip_S_hilo_cdf = pd.Series(cum_dist, index=Precip_S_hilo)

plt.figure(figsize=(12,8))
Precip_S_hilo_cdf.plot(drawstyle='steps',label='CDF')
plt.axvline(x=perc_95,linestyle='-.',color ='r',label='95$^{th}$ percentile value')
plt.xlabel("Daily Rainfall (mm)",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Cumulative Distribution Function",fontsize=14)
plt.title("Cumulative Distribution Function of Daily Rainfall at Hilo, USA",fontsize=16)
plt.legend(loc='best',fontsize=14)
plt.savefig('/content/gdrive/My Drive/ATMS_597_Project_3_Group_D/CDF_Rainfall_Hilo.png',dpi=500)
plt.show()


!pip install pydap
!pip install netcdf4

!apt-get -qq install libproj-dev proj-data proj-bin libgeos-dev
!pip install Cython
!pip install --upgrade --force-reinstall shapely --no-binary shapely
!pip install cartopy

%pylab inline
import pandas as pd
import xarray as xr

#  reading extreme precipitation dates
precip_data = pd.read_csv("/content/drive/My Drive/Xtreme_Precip_days_Hilo.csv")

# extract data fields on extreme precipitation days
years = pd.date_range(start='1996-01-01', end='2019-12-31', freq='D')
yearmon = years[(years.month==10) | (years.month==11) | (years.month==12)]

windv_250_ds = []
windu_250_ds = []
windv_500_ds = []
windu_500_ds = []
hgt_500_ds = []
windv_850_ds = []
windu_850_ds = []
temp_850_ds = []
shum_850_ds = []
sktemp_ds = []
surf_windu_ds = []
surf_windv_ds = []
pr_wtr_ds = []

years = [i for i in range(1996,2020)]


for iyr in years:
    print('working on '+str(iyr))
    if len(precip_data['time'][pd.DatetimeIndex(precip_data['time']).year.values == iyr])>0:
      dates_year = pd.to_datetime(np.array(precip_data['time'][pd.DatetimeIndex(precip_data['time']).year.values == iyr]))

      windu_250 = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/uwnd.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(level=250,time=dates_year)
      windu_250_ds.append(windu_250)

      windv_250 = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/vwnd.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(level=250,time=dates_year)
      windv_250_ds.append(windv_250)

      windu_500 = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/uwnd.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(level=500,time=dates_year)
      windu_500_ds.append(windu_500)

      windv_500 = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/vwnd.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(level=500,time=dates_year)
      windv_500_ds.append(windv_500)

      hgt_500 = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/hgt.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(level=500,time=dates_year)
      hgt_500_ds.append(hgt_500)

      temp_850 = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/air.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(level=850,time=dates_year)
      temp_850_ds.append(temp_850)

      shum_850 = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/shum.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(level=850,time=dates_year)
      shum_850_ds.append(shum_850)

      windu_850 = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/uwnd.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(level=850,time=dates_year)
      windu_850_ds.append(windu_850)

      windv_850 = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/vwnd.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(level=850,time=dates_year)
      windv_850_ds.append(windv_850)

      sktemp = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/surface_gauss/skt.sfc.gauss.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(time=dates_year)
      sktemp_ds.append(sktemp)

      surf_windu = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/surface/uwnd.sig995.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(time=dates_year)
      surf_windu_ds.append(surf_windu)

      surf_windv = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/surface/vwnd.sig995.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(time=dates_year)
      surf_windv_ds.append(surf_windv)

      pr_wtr = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/surface/pr_wtr.eatm.'+str(iyr)+'.nc',
                          engine='netcdf4').sel(time=dates_year)
      pr_wtr_ds.append(pr_wtr)

# combine each dataset fields relative to years
windu_250_ds_yearcombined = xr.concat(windu_250_ds, dim='time')
windu_250_ds_yearcombined

windv_250_ds_yearcombined = xr.concat(windv_250_ds, dim='time')
windv_250_ds_yearcombined

hgt_500_ds_yearcombined = xr.concat(hgt_500_ds, dim='time')
hgt_500_ds_yearcombined

windu_500_ds_yearcombined = xr.concat(windu_500_ds, dim='time')
windu_500_ds_yearcombined

windv_500_ds_yearcombined = xr.concat(windv_500_ds, dim='time')
windv_500_ds_yearcombined

windu_850_ds_yearcombined = xr.concat(windu_850_ds, dim='time')
windu_850_ds_yearcombined

windv_850_ds_yearcombined = xr.concat(windv_850_ds, dim='time')
windv_850_ds_yearcombined

temp_850_ds_yearcombined = xr.concat(temp_850_ds, dim='time')
temp_850_ds_yearcombined

shum_850_ds_yearcombined = xr.concat(shum_850_ds, dim='time')
shum_850_ds_yearcombined

sktemp_ds_yearcombined = xr.concat(sktemp_ds, dim='time')
sktemp_ds_yearcombined

surf_windu_ds_yearcombined = xr.concat(surf_windu_ds, dim='time')
surf_windu_ds_yearcombined

surf_windv_ds_yearcombined = xr.concat(surf_windv_ds, dim='time')
surf_windv_ds_yearcombined

pr_wtr_ds_yearcombined = xr.concat(pr_wtr_ds, dim='time')
pr_wtr_ds_yearcombined

# save dataset fields on extreme precipitation days to netcdf files
pr_wtr_ds_yearcombined.to_netcdf('pr_wtr_ds_yearcombined_extrem.nc')
surf_windv_ds_yearcombined.to_netcdf('surf_windv_ds_yearcombined_extrem.nc')
surf_windu_ds_yearcombined.to_netcdf('surf_windu_ds_yearcombined_extrem.nc')
sktemp_ds_yearcombined.to_netcdf('sktemp_ds_yearcombined_extrem.nc')
shum_850_ds_yearcombined.to_netcdf('shum_850_ds_yearcombined_extrem.nc')
temp_850_ds_yearcombined.to_netcdf('temp_850_ds_yearcombined_extrem.nc') 
windv_850_ds_yearcombined.to_netcdf('windv_850_ds_yearcombined_extrem.nc')
windu_850_ds_yearcombined.to_netcdf('windu_850_ds_yearcombined_extrem.nc')
windv_500_ds_yearcombined.to_netcdf('windv_500_ds_yearcombined_extrem.nc')
windu_500_ds_yearcombined.to_netcdf('windu_500_ds_yearcombined_extrem.nc')
hgt_500_ds_yearcombined.to_netcdf('hgt_500_ds_yearcombined_extrem.nc')
windv_250_ds_yearcombined.to_netcdf('windv_250_ds_yearcombined_extrem.nc')
windu_250_ds_yearcombined.to_netcdf('windu_250_ds_yearcombined_extrem.nc')

!mv *.nc "/content/drive/My Drive/"

# extract precipitable water and plot
pr_wtr = xr.open_dataset('/content/drive/My Drive/pr_wtr_ds_yearcombined_extrem.nc', engine='netcdf4')
pr_wtr_avg = pr_wtr.mean(dim='time')

pr_wtr_avg['pr_wtr']

import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

fig = plt.figure(figsize=(15,10))

ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))
ax.set_global()

ax.add_feature(cfeature.COASTLINE.with_scale('110m'))

c1 = ax.contourf(pr_wtr_avg['lon'], pr_wtr_avg['lat'], pr_wtr_avg['pr_wtr'],
             transform=ccrs.PlateCarree(), cmap='BrBG')

ax.plot(205.0, 20.0, 'r*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='r',transform=ccrs.PlateCarree())
g1 = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='-')
g1.xlocator = mticker.FixedLocator(np.arange(-180,180,30))
g1.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
g1.xformatter = LONGITUDE_FORMATTER
g1.yformatter = LATITUDE_FORMATTER
g1.xlabel_style = {'size': 12}
g1.ylabel_style = {'size': 12}
g1.xlabels_top = False
g1.ylabels_right = False
ax.coastlines()

plt.title('Extreme precipitation days: Precipitable Water', fontsize = 30)

cb = fig.colorbar(c1, shrink=0.6)
cb.ax.tick_params(labelsize = 12)
cb.set_ticks(np.arange(0, 64, 4))
cb.set_label('Precipitable Water [mm]', fontsize = 20)
plt.tight_layout()
plt.show()
fig.savefig('pr_wtr.png',dpi=300, bbox_inches='tight')


# Extreme precipitation days: 500-hPa Geopotential Height
hgt = xr.open_dataset('/content/drive/My Drive/hgt_500_ds_yearcombined_extrem.nc', engine='netcdf4')
uwind = xr.open_dataset('/content/drive/My Drive/windu_500_ds_yearcombined_extrem.nc', engine='netcdf4')
vwind = xr.open_dataset('/content/drive/My Drive/windv_500_ds_yearcombined_extrem.nc', engine='netcdf4')


hgt_avg = hgt.mean(dim='time')
vwind_avg = vwind.mean(dim='time')
uwind_avg = uwind.mean(dim='time')

uwind_avg['uwnd']

import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

fig = plt.figure(figsize=(15,10))


ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))
ax.set_extent([-180, 180, -90, 90])

ax.add_feature(cfeature.COASTLINE.with_scale('110m'))
c1 = ax.contourf(hgt_avg['lon'], hgt_avg['lat'], hgt_avg['hgt']/10,
             transform=ccrs.PlateCarree(), levels = np.linspace(450, 600, 16), cmap='viridis')

ax.barbs(uwind_avg.lon[::4].values, uwind_avg.lat[::4].values, uwind_avg['uwnd'][::4, ::4].values, vwind_avg['vwnd'][::4, ::4].values, length=5,
         sizes=dict(emptybarb=0.25, spacing=0.3, height=0.7),linewidth=0.9)
ax.plot(205.0, 20.0, 'b*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='r',transform=ccrs.PlateCarree())
g1 = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='-')
g1.xlocator = mticker.FixedLocator(np.arange(-180,180,30))
g1.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
g1.xformatter = LONGITUDE_FORMATTER
g1.yformatter = LATITUDE_FORMATTER
g1.xlabel_style = {'size': 12}
g1.ylabel_style = {'size': 12}
g1.xlabels_top = False
g1.ylabels_right = False
ax.coastlines()


plt.title('Extreme precipitation days: 500-hPa Geopotential Height', fontsize = 25)
cb = fig.colorbar(c1, shrink=0.5)
cb.ax.tick_params(labelsize = 12)
cb.set_label('Geopotential Height [dam]', fontsize = 16)
cb.set_ticks(np.arange(450, 601, 10))
plt.tight_layout()
plt.show()
fig.savefig('hgt_wind_500.png',dpi=300, bbox_inches='tight')


# Extreme precipitation days: 250-hPa Wind
uwind = xr.open_dataset('/content/drive/My Drive/windu_250_ds_yearcombined_extrem.nc', engine='netcdf4')
vwind = xr.open_dataset('/content/drive/My Drive/windv_250_ds_yearcombined_extrem.nc', engine='netcdf4')

vwind_avg = vwind.mean(dim='time')
uwind_avg = uwind.mean(dim='time')

uwind_avg['uwnd']

import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

fig = plt.figure(figsize=(15,10))


ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))
ax.set_extent([-180, 180, -90, 90])

ax.add_feature(cfeature.COASTLINE.with_scale('110m'))
wind_speed = np.sqrt(uwind_avg['uwnd']**2 + vwind_avg['vwnd']**2)
c1 = ax.contourf(uwind_avg.lon, uwind_avg.lat, wind_speed, transform=ccrs.PlateCarree(), levels = 15, cmap='rainbow')

ax.barbs(uwind_avg.lon[::4].values, uwind_avg.lat[::4].values, uwind_avg['uwnd'][::4, ::4].values, vwind_avg['vwnd'][::4, ::4].values, length=5,
         sizes=dict(emptybarb=0.25, spacing=0.3, height=0.7),linewidth=0.9)
ax.plot(205.0, 20.0, 'r*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='r',transform=ccrs.PlateCarree())
g1 = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='-')
g1.xlocator = mticker.FixedLocator(np.arange(-180,180,30))
g1.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
g1.xformatter = LONGITUDE_FORMATTER
g1.yformatter = LATITUDE_FORMATTER
g1.xlabel_style = {'size': 12}
g1.ylabel_style = {'size': 12}
g1.xlabels_top = False
g1.ylabels_right = False
ax.coastlines()


plt.title('Extreme precipitation days: 250-hPa Wind', fontsize = 30)
cb = fig.colorbar(c1, shrink=0.6)
cb.ax.tick_params(labelsize = 12)
cb.set_label('Wind [m $s^{-1}$]', fontsize = 20)
plt.tight_layout()
plt.show()
fig.savefig('wind_250.png',dpi=300, bbox_inches='tight')


# Extreme precipitation days: Specific Humidity [kg/kg]
uwind = xr.open_dataset('/content/drive/My Drive/windu_850_ds_yearcombined_extrem.nc', engine='netcdf4')
vwind = xr.open_dataset('/content/drive/My Drive/windv_850_ds_yearcombined_extrem.nc', engine='netcdf4')
shum = xr.open_dataset('/content/drive/My Drive/shum_850_ds_yearcombined_extrem.nc', engine='netcdf4')

vwind_avg = vwind.mean(dim='time')
uwind_avg = uwind.mean(dim='time')
shum_avg = shum.mean(dim='time')
shum['shum'].attrs

import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

fig = plt.figure(figsize=(15,10))


ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))
ax.set_extent([-180, 180, -90, 90])

ax.add_feature(cfeature.COASTLINE.with_scale('110m'))
c1 = ax.contourf(shum_avg['lon'], shum_avg['lat'], shum_avg['shum'],
             transform=ccrs.PlateCarree(), levels = 10, cmap='Reds')

ax.barbs(uwind_avg.lon[::4].values, uwind_avg.lat[::4].values, uwind_avg['uwnd'][::4, ::4].values, vwind_avg['vwnd'][::4, ::4].values, length=5,
         sizes=dict(emptybarb=0.25, spacing=0.3, height=0.7),linewidth=0.9)
ax.plot(205.0, 20.0, 'b*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='g',transform=ccrs.PlateCarree())
g1 = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='-')
g1.xlocator = mticker.FixedLocator(np.arange(-180,180,30))
g1.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
g1.xformatter = LONGITUDE_FORMATTER
g1.yformatter = LATITUDE_FORMATTER
g1.xlabel_style = {'size': 14}
g1.ylabel_style = {'size': 14}
g1.xlabels_top = False
g1.ylabels_right = False
ax.coastlines()

plt.title('850-hPa Specific Humidity', fontsize = 20)
cb = fig.colorbar(c1, shrink=0.5)
cb.ax.tick_params(labelsize = 12)
cb.set_label('Extreme precipitation days: Specific Humidity [kg/kg]', fontsize = 16)
plt.tight_layout()
plt.show()
fig.savefig('shum_850.png')


# Extreme precipitation days: 850-hPa Temperature and specific humidity
uwind = xr.open_dataset('/content/drive/My Drive/windu_850_ds_yearcombined_extrem.nc', engine='netcdf4')
vwind = xr.open_dataset('/content/drive/My Drive/windv_850_ds_yearcombined_extrem.nc', engine='netcdf4')
temp = xr.open_dataset('/content/drive/My Drive/temp_850_ds_yearcombined_extrem.nc', engine='netcdf4')


vwind_avg = vwind.mean(dim='time')
uwind_avg = uwind.mean(dim='time')
temp_avg = temp.mean(dim='time')
temp_avg['air']

import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

fig = plt.figure(figsize=(15,10))


ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))
ax.set_extent([-180, 180, -90, 90])

ax.add_feature(cfeature.COASTLINE.with_scale('110m'))
c1 = ax.contour(temp_avg['lon'], temp_avg['lat'], temp_avg['air']-273.15,
             transform=ccrs.PlateCarree(), levels = 10, cmap='Reds')
c2 = ax.contourf(shum_avg['lon'], shum_avg['lat'], shum_avg['shum'],
             transform=ccrs.PlateCarree(), levels = 10, cmap='rainbow')

ax.barbs(uwind_avg.lon[::4].values, uwind_avg.lat[::4].values, uwind_avg['uwnd'][::4, ::4].values, vwind_avg['vwnd'][::4, ::4].values, length=5,
         sizes=dict(emptybarb=0.25, spacing=0.3, height=0.7),linewidth=0.9)
ax.plot(205.0, 20.0, 'b*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='r',transform=ccrs.PlateCarree())
g1 = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='-')
g1.xlocator = mticker.FixedLocator(np.arange(-180,180,30))
g1.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
g1.xformatter = LONGITUDE_FORMATTER
g1.yformatter = LATITUDE_FORMATTER
g1.xlabel_style = {'size': 12}
g1.ylabel_style = {'size': 12}
g1.xlabels_top = False
g1.ylabels_right = False
ax.coastlines()

cb2 = fig.colorbar(c1, shrink=0.6)
cb2.ax.tick_params(labelsize = 12)
cb2.set_label('Temperature [C]', fontsize = 16)

plt.title('Extreme precipitation days: 850-hPa Temperature and specific humidity', fontsize = 20)
cb = fig.colorbar(c2, shrink=0.8, orientation = 'horizontal')
cb.ax.tick_params(labelsize = 12)
cb.set_label('Specific Humidity [kg/kg]', fontsize = 16)

plt.tight_layout()
plt.show()
fig.savefig('temp_shum_850.png',dpi=300, bbox_inches='tight')



# Extreme precipitation days: Skin temperature and Surface wind
surf_uwind = xr.open_dataset('/content/drive/My Drive/surf_windu_ds_yearcombined_extrem.nc', engine='netcdf4')
surf_vwind = xr.open_dataset('/content/drive/My Drive/surf_windv_ds_yearcombined_extrem.nc', engine='netcdf4')
stemp = xr.open_dataset('/content/drive/My Drive/sktemp_ds_yearcombined_extrem.nc', engine='netcdf4')

surf_vwind_avg = surf_vwind.mean(dim='time')
surf_uwind_avg = surf_uwind.mean(dim='time')
stemp_avg = stemp.mean(dim='time')
stemp_avg['skt']

import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

fig = plt.figure(figsize=(15,10))


ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))
ax.set_extent([-180, 180, -90, 90])

ax.add_feature(cfeature.COASTLINE.with_scale('110m'))
c1 = ax.contourf(stemp_avg['lon'], stemp_avg['lat'], stemp_avg['skt']-273.15,
             transform=ccrs.PlateCarree(), levels = 10, cmap='seismic')

ax.barbs(surf_uwind_avg.lon[::4].values, surf_uwind_avg.lat[::4].values, surf_uwind_avg['uwnd'][::4, ::4].values, surf_vwind_avg['vwnd'][::4, ::4].values, length=5,
         sizes=dict(emptybarb=0.25, spacing=0.3, height=0.7),linewidth=0.9)
ax.plot(205.0, 20.0, 'b*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='y',transform=ccrs.PlateCarree())
g1 = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='-')
g1.xlocator = mticker.FixedLocator(np.arange(-180,180,30))
g1.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
g1.xformatter = LONGITUDE_FORMATTER
g1.yformatter = LATITUDE_FORMATTER
g1.xlabel_style = {'size': 12}
g1.ylabel_style = {'size': 12}
g1.xlabels_top = False
g1.ylabels_right = False
ax.coastlines()


plt.title('Extreme precipitation days: Skin temperature and Surface wind', fontsize = 25)
cb = fig.colorbar(c1, shrink=0.6)
cb.ax.tick_params(labelsize = 12)
cb.set_label('Temperature [C]', fontsize = 16)
cb.set_ticks(np.arange(-48, 40, 4))
plt.tight_layout()
plt.show()
fig.savefig('surface_temp.png',dpi=300, bbox_inches='tight')


    



