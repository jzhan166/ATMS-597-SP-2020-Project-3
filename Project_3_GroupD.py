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


##### BASELINE FOR COMPARISON (1981-2010)

months = xr.cftime_range(start='0001-01-01', end='0001-12-01', freq='MS', calendar = 'standard') # selecting long-term mean data for 1981-2010 with cftime.DatetimeGregorian format
months = months[(months.month==10)|(months.month==11)|(months.month==12)] # selecting long-term mean data for OND in 1981-2010


##### Importing Data from THREDDS Server 

# For each variable, we directly selected the relevant pressure level and time period desired to import 
# 250 hPa wind vectors and wind speed

windu_250_baseline = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/uwnd.mon.1981-2010.ltm.nc', engine='netcdf4').sel(level=250, time=months)

windv_250_baseline = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/vwnd.mon.1981-2010.ltm.nc', engine='netcdf4').sel(level=250, time=months)

windspd_250_baseline = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/wspd.mon.1981-2010.ltm.nc', engine='netcdf4').sel(level=250, time=months)

# 500 hPa wind vectors and geopotential height 

windu_500_baseline = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/uwnd.mon.1981-2010.ltm.nc', engine='netcdf4').sel(level=500, time=months)

windv_500_baseline = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/vwnd.mon.1981-2010.ltm.nc', engine='netcdf4').sel(level=500, time=months)

hgt_500_baseline = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/hgt.mon.1981-2010.ltm.nc', engine='netcdf4').sel(level=500, time=months)

# 850 hPa temperature, specific humidity, and wind vectors 

temp_850_baseline = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/air.mon.1981-2010.ltm.nc', engine='netcdf4').sel(level=850, time=months)

spec_hum_850_baseline = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/shum.mon.1981-2010.ltm.nc', engine='netcdf4').sel(level=850, time=months)

windu_850_ltm_baseline = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/uwnd.mon.1981-2010.ltm.nc', engine='netcdf4').sel(level=850, time=months)

windv_850_ltm_baseline = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/vwnd.mon.1981-2010.ltm.nc', engine='netcdf4').sel(level=850, time=months)

# skin temperature 

skin_temp_ltm_baseline = xr.open_dataset('https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/skt.sfc.mon.1981-2010.ltm.nc', engine='netcdf4').sel(time=months)

# surface winds (sig995??)

surf_wind_u_baseline = xr.open_dataset('http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface/uwnd.sig995.mon.1981-2010.ltm.nc', engine='netcdf4').sel(time=months)

surf_wind_v_baseline = xr.open_dataset('http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface/vwnd.sig995.mon.1981-2010.ltm.nc', engine='netcdf4').sel(time=months)

# total atmospheric column water vapor 

atm_col_wv_baseline = xr.open_dataset('http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface/pr_wtr.eatm.day.1981-2010.ltm.nc', engine='netcdf4').sel(time=months)


##### Calculating seasonal (OND) means by averaging over time dimension 

# 250 hPa u-wind  
windu_250_ltm_OND = windu_250_baseline.mean(dim='time')
# 250 hPa v-wind 
windv_250_ltm_OND = windv_250_baseline.mean(dim='time')
# 250 hPa wind speed 
windspd_250_ltm_OND = windspd_250_baseline.mean(dim='time')

# 500 hPa u-wind
windu_500_ltm_OND = windu_500_baseline.mean(dim='time')
# 500 hPa v-wind
windv_500_ltm_OND = windv_500_baseline.mean(dim='time')
# 500 hPa geopotential height
hgt_500_ltm_OND = hgt_500_baseline.mean(dim='time')

# 850 hPa temperature 
temp_850_ltm_OND = temp_850_baseline.mean(dim='time')
# 850 hPa specific humidity 
spec_hum_850_ltm_OND = spec_hum_850_baseline.mean(dim='time')
# 850 hPa u-wind
windu_850_ltm_OND = windu_850_ltm_baseline.mean(dim='time')
# 850 hPa v-wind
windv_850_ltm_OND = windv_850_ltm_baseline.mean(dim='time')

# Skin temperature 
skin_temp_ltm_OND = skin_temp_ltm_baseline.mean(dim='time')
# Surface u-wind
surf_wind_u_ltm_OND = surf_wind_u_baseline.mean(dim='time')
# Surface v-wind
surf_wind_v_ltm_OND = surf_wind_v_baseline.mean(dim='time')
# Atmospheric column water vapor 
atm_col_wv_ltm_OND = atm_col_wv_baseline.mean(dim='time')


##### LONG-TERM MEAN MAPS 

import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

## 250 hPa wind, long term mean 
# Setup empty figure 
fig = plt.figure(figsize=(20,10))

# Define projection (Plate Carree)
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))

# Mark and label location 
ax.plot(205.0, 20.0, 'w*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='w',transform=ccrs.PlateCarree())

# Set up grey gridlines 
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

# Draw coastlines
ax.coastlines()

# Add title 
plt.title('Long-Term Mean: 250 hPa Wind Vectors', fontsize = 30)

# Plot filled contours of wind speed
c1 = ax.contourf(windspd_250_ltm_OND['lon'], windspd_250_ltm_OND['lat'], windspd_250_ltm_OND['wspd'].values.squeeze(), 20, transform=ccrs.PlateCarree(), cmap='rainbow')

# Plot wind vector quivers 
ax.quiver(windu_250_ltm_OND['lon'][::4], windu_250_ltm_OND['lat'][::4], windu_250_ltm_OND['uwnd'][::4,::4].values.squeeze(), windv_250_ltm_OND['vwnd'][::4,::4].values.squeeze(), 20, transform=ccrs.PlateCarree())

# Add colorbar 
cb = fig.colorbar(c1, ax=ax, shrink=0.8)
cb.ax.tick_params(labelsize = 10)
cb.set_label('Wind Speed', fontsize = 20)

# Save figure 
plt.savefig('/content/drive/My Drive/ATMS_597_Project_3_Group_D/long_term_250hPa.png')

## 500 hPa wind speed and geopotential height, long term mean 
# Set up empty figure 
plt.figure(figsize=(20,10))

# Define projection 
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))

# Mark location 
ax.plot(205.0, 20.0, 'r*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='r',transform=ccrs.PlateCarree())

# Add gridlines 
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

# Draw coastlines
ax.coastlines()

# Add plot title 
plt.title('Long-Term Mean: 500 hPa Winds and Geopotential Height', fontsize = 30)

# Plot filled contours of geopotential height 
c1 = ax.contourf(hgt_500_ltm_OND['lon'], hgt_500_ltm_OND['lat'], hgt_500_ltm_OND['hgt'].values.squeeze(),20, transform=ccrs.PlateCarree())

# Plot wind vectors 
ax.quiver(windu_500_ltm_OND['lon'][::4], windu_500_ltm_OND['lat'][::4], windu_500_ltm_OND['uwnd'][::4,::4].values.squeeze(), windv_500_ltm_OND['vwnd'][::4,::4].values.squeeze(), 20, transform=ccrs.PlateCarree())

# Add colorbar
cb = fig.colorbar(c1, ax=ax, shrink=0.6)
cb.ax.tick_params(labelsize = 10)
cb.set_label('Geopotential Height', fontsize = 20)

plt.savefig('/content/drive/My Drive/ATMS_597_Project_3_Group_D/long_term_500hPa.png')

## 850 hPa Specific Humidity, Temperature, Wind Vectors, long term mean 
# Set up empty figure 
plt.figure(figsize=(20,10))

# Define projection 
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))

# Mark location 
ax.plot(205.0, 20.0, 'r*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='r',transform=ccrs.PlateCarree())

# Add gridlines 
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

# Draw coastlines 
ax.coastlines()

# Add title 
plt.title('Long-Term Mean: 850 hPa Specific Humidity, Temperature, Winds', fontsize = 20)
 
# Plot filled contours of specific humidity 
c1 = ax.contourf(spec_hum_850_ltm_OND['lon'], spec_hum_850_ltm_OND['lat'], spec_hum_850_ltm_OND['shum'].values.squeeze(), 20, transform=ccrs.PlateCarree(), cmap='rainbow', alpha = 0.9)
# Add colorbar 
cb1 = fig.colorbar(c1, ax=ax, shrink=0.6, orientation="horizontal")
cb1.ax.tick_params(labelsize = 10)
cb1.set_label('Specific Humidity', fontsize = 12)

# Temperature
# Plot contours of temperature  
c2 = ax.contour(temp_850_ltm_OND['lon'], temp_850_ltm_OND['lat'], temp_850_ltm_OND['air'].values.squeeze(), 20, transform=ccrs.PlateCarree(), cmap="seismic", vmin=-27.5, vmax=27.5)
# Add colorbar 
cb2 = fig.colorbar(c2, ax=ax, shrink=1.0)
cb2.ax.tick_params(labelsize = 10)
cb2.set_label('Temperature', fontsize = 12)

# Wind Vectors 
# Plot wind vectors 
ax.quiver(windu_850_ltm_OND['lon'][::3], windu_850_ltm_OND['lat'][::3], windu_850_ltm_OND['uwnd'][::3,::3].values.squeeze(), windv_850_ltm_OND['vwnd'][::3,::3].values.squeeze(), 20, transform=ccrs.PlateCarree())

# Save figure 
plt.savefig('/content/drive/My Drive/ATMS_597_Project_3_Group_D/long_term_850hPa_all.png')

## Skin Temperature and Surface Winds 

# Set up empty figure 
plt.figure(figsize=(20,10))

# Define projection 
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))

# Mark location 
ax.plot(205.0, 20.0, 'y*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='y',transform=ccrs.PlateCarree())

# Add gridlines 
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

# Add coastlines 
ax.coastlines()

# Add title 
plt.title('Long-Term Mean: Skin Temperature and Surface Winds', fontsize = 30)

# Skin temperature 
# Plot filled contours of skin temperature 
c1 = ax.contourf(skin_temp_ltm_OND['lon'], skin_temp_ltm_OND['lat'], skin_temp_ltm_OND['skt'].values.squeeze(), 20, transform=ccrs.PlateCarree(), cmap='seismic')
# Add colorbar 
cb = fig.colorbar(c1, ax=ax, shrink=0.8)
cb.ax.tick_params(labelsize = 10)
cb.set_label('Skin Temperature (C)', fontsize = 20)

# Wind vectors
# Plot wind vectors 
ax.quiver(surf_wind_u_ltm_OND['lon'][::3], surf_wind_u_ltm_OND['lat'][::3], surf_wind_u_ltm_OND['uwnd'][::3,::3].values.squeeze(), surf_wind_v_ltm_OND['vwnd'][::3,::3].values.squeeze(), 20, transform=ccrs.PlateCarree())

# Save figure 
plt.savefig('/content/drive/My Drive/ATMS_597_Project_3_Group_D/long_term_sktemp_surfwinds.png')

## Atmospheric Column Water Vapor 
# Set up empty figure 
plt.figure(figsize=(20,10))

# Define projection 
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))

# Mark location 
ax.plot(205.0, 20.0, 'b*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='b',transform=ccrs.PlateCarree())

# Draw gridlines 
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

# Draw coastlines 
ax.coastlines()

# Add title 
plt.title('Long-Term Mean: Atmospheric Column Water Vapor', fontsize = 30)

# Plot filled contours of atmospheric column water vapor 
c1 = ax.contourf(atm_col_wv_ltm_OND['lon'], atm_col_wv_ltm_OND['lat'], atm_col_wv_ltm_OND['pr_wtr'].values.squeeze(), 20, transform=ccrs.PlateCarree(), cmap='BrBG')

# Add colorbar 
cb = fig.colorbar(c1, ax=ax, shrink=0.8)
cb.ax.tick_params(labelsize = 10)
cb.set_label('Atmospheric Column Water Vapor', fontsize = 20)

# Save figure 
plt.savefig('/content/drive/My Drive/ATMS_597_Project_3_Group_D/long_term_pr_wtr.png')

#### Calculating Seasonal Anomalies

import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

## 250 hPa Wind vectors and wind speed

# Import netcdf files (previously created) with weather fields on extreme precipitation days 
windu_250_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/windu_250_ds_yearcombined_extrem.nc', engine='netcdf4') # 250 u-wind
windv_250_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/windv_250_ds_yearcombined_extrem.nc', engine='netcdf4') # 250 v-wind

# Calculate seasonal anomalies: subtract seasonal global long term mean from the mean on extreme precipitation days 
sa_windu_250_ds_yearcombined_extrem = windu_250_ds_yearcombined_extrem.mean(dim='time') - windu_250_ltm_OND # 250 hPa u-wind
sa_windv_250_ds_yearcombined_extrem = windv_250_ds_yearcombined_extrem.mean(dim='time') - windv_250_ltm_OND # 250 hPa v-wind

# Set up empty figure 
plt.figure(figsize=(20,10))

# Define projection 
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))

# Mark location 
ax.plot(205.0, 20.0, 'w*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='w',transform=ccrs.PlateCarree())

# Draw gridlines 
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

# Draw coastlines 
ax.coastlines()

# Wind speed 
# Calculate wind speeds from u-wind and v-wind values 
sa_windspd_250_ds_yearcombined_extrem = np.sqrt(sa_windu_250_ds_yearcombined_extrem['uwnd']**2 + sa_windv_250_ds_yearcombined_extrem['vwnd']**2)
# Draw filled contours of wind speed 
c1 = ax.contourf(sa_windspd_250_ds_yearcombined_extrem.lon, sa_windspd_250_ds_yearcombined_extrem.lat, sa_windspd_250_ds_yearcombined_extrem, cmap = 'rainbow',transform=ccrs.PlateCarree())

# Wind vectors
# Plot wind vectors 
ax.quiver(sa_windu_250_ds_yearcombined_extrem['lon'][::4].values, sa_windu_250_ds_yearcombined_extrem['lat'][::4].values, sa_windu_250_ds_yearcombined_extrem['uwnd'][::4,::4].values, sa_windv_250_ds_yearcombined_extrem['vwnd'][::4,::4].values,transform=ccrs.PlateCarree(central_longitude=180.))

# Add plot title 
plt.title('Seasonal Anomaly: 250 hPa Wind Vectors', fontsize = 30)

# Add colorbar
cb = fig.colorbar(c1, ax = ax, shrink=0.8)
cb.ax.tick_params(labelsize = 12)
cb.set_label('Wind Speed', fontsize = 20)

# Save figure 
plt.savefig('/content/drive/My Drive/ATMS_597_Project_3_Group_D/seasonal_anomaly_250hPa.png')

## 500 hPa Wind vectors and geopotential height 

# Import netcdf files (previously created) with weather fields on extreme precipitation days 
windu_500_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/windu_500_ds_yearcombined_extrem.nc', engine='netcdf4') # 500 hPa u-wind
windv_500_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/windv_500_ds_yearcombined_extrem.nc', engine='netcdf4') # 500 hPa v-wind
hgt_500_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/hgt_500_ds_yearcombined_extrem.nc', engine='netcdf4') # 500 hPa geopotential height

# Calculate seasonal anomalies: subtract seasonal global long term mean from the mean on extreme precipitation days 
sa_windu_500_ds_yearcombined_extrem = windu_500_ds_yearcombined_extrem.mean(dim='time') - windu_500_ltm_OND # 500 hPa u-wind
sa_windv_500_ds_yearcombined_extrem = windv_500_ds_yearcombined_extrem.mean(dim='time') - windv_500_ltm_OND # 500 hPa v-wind
sa_hgt_500_ds_yearcombined_extrem = (hgt_500_ds_yearcombined_extrem.mean(dim='time') - hgt_500_ltm_OND)/10 # divided by 10 to convert to dam units

# 500 hPa wind vectors and geopotential height
# Set up empty figure 
plt.figure(figsize=(20,10))

# Define projection 
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))

# Mark location 
ax.plot(205.0, 20.0, 'b*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='y',transform=ccrs.PlateCarree())

# Draw gridlines 
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

# Draw coastlines 
ax.coastlines()

# Add title 
plt.title('Seasonal Anomaly: 500 hPa Wind Vectors', fontsize = 30)

# Geopotential height
# Plot filled contours of geopotential height
c1 = ax.contourf(sa_hgt_500_ds_yearcombined_extrem['lon'], sa_hgt_500_ds_yearcombined_extrem['lat'], sa_hgt_500_ds_yearcombined_extrem['hgt'].values.squeeze(),20, cmap='viridis', transform=ccrs.PlateCarree())

# Wind vectors 
# Plot wind vectors 
ax.quiver(sa_windu_500_ds_yearcombined_extrem['lon'][::4], sa_windu_500_ds_yearcombined_extrem['lat'][::4], sa_windu_500_ds_yearcombined_extrem['uwnd'][::4,::4].values.squeeze(), sa_windv_500_ds_yearcombined_extrem['vwnd'][::4,::4].values.squeeze(), 20, transform=ccrs.PlateCarree())

# Add colorbar 
cb = fig.colorbar(c1, ax=ax, shrink=0.8)
cb.ax.tick_params(labelsize = 10)
cb.set_label('Geopotential Height (dam)', fontsize = 20)

# Save figure
plt.savefig('/content/drive/My Drive/ATMS_597_Project_3_Group_D/seasonal_anomaly_500hPa.png')

## 850 hPa Temperature, wind vectors, and specific humidity 

# Import netcdf files (previously created) with weather fields on extreme precipitation days 
temp_850_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/temp_850_ds_yearcombined_extrem.nc', engine='netcdf4') # 850 hPa temperature
shum_850_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/shum_850_ds_yearcombined_extrem.nc', engine='netcdf4') # 850 hPa specific humidity 
windu_850_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/windu_850_ds_yearcombined_extrem.nc', engine='netcdf4') # 850 hPa u-wind
windv_850_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/windv_850_ds_yearcombined_extrem.nc', engine='netcdf4') # 850 hPa v-wind

# Calculate seasonal anomalies: subtract seasonal global long term mean from the mean on extreme precipitation days 
sa_temp_850_ds_yearcombined_extrem = temp_850_ds_yearcombined_extrem.mean(dim='time')-273.15 - temp_850_ltm_OND # 850 hPa temperature, converted to C 
sa_shum_850_ds_yearcombined_extrem = shum_850_ds_yearcombined_extrem.mean(dim='time') - spec_hum_850_ltm_OND/1000 # 850 hPa specific humidity, converted to kg/kg
sa_windu_850_ds_yearcombined_extrem = windu_850_ds_yearcombined_extrem.mean(dim='time') - windu_850_ltm_OND # 850 hPa u-wind
sa_windv_850_ds_yearcombined_extrem = windv_850_ds_yearcombined_extrem.mean(dim='time') - windv_850_ltm_OND # 850 hPa v-wind

# 850 hPa wind vectors, temperature, and specific humidity 
# Set up empty figure  
fig = plt.figure(figsize=(20,10))

# Define projection 
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))

# Draw gridlines
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

# Draw coastlines 
ax.coastlines()

# Add title 
plt.title('Seasonal Anomaly: 850 hPa Winds, Temperature, and Specific Humidity', fontsize = 25)

# Plot contours of temperature 
c1 = ax.contour(sa_temp_850_ds_yearcombined_extrem['lon'], sa_temp_850_ds_yearcombined_extrem['lat'], sa_temp_850_ds_yearcombined_extrem['air'], vmin=-5, vmax=5, transform=ccrs.PlateCarree(), cmap='seismic', levels = 20)
# Plot filled contours of specific humidity 
c2 = ax.contourf(sa_shum_850_ds_yearcombined_extrem['lon'], sa_shum_850_ds_yearcombined_extrem['lat'], sa_shum_850_ds_yearcombined_extrem['shum'], 20, cmap='rainbow', transform=ccrs.PlateCarree(), alpha=0.8)
# Plot wind vectors
c3 = ax.quiver(sa_windu_850_ds_yearcombined_extrem['lon'][::2], sa_windu_850_ds_yearcombined_extrem['lat'][::2], sa_windu_850_ds_yearcombined_extrem['uwnd'][::2,::2].values.squeeze(), sa_windv_850_ds_yearcombined_extrem['vwnd'][::2,::2].values.squeeze(), 20, transform=ccrs.PlateCarree())

# Add colorbars 
cb1 = fig.colorbar(c1, shrink=0.8)
cb1.ax.tick_params(labelsize = 12)
cb1.set_label('Temperature [C]', fontsize = 20)

# Add colorbars
cb2 = fig.colorbar(c2, shrink=0.8, orientation="horizontal")
cb2.ax.tick_params(labelsize = 12)
cb2.set_label('Specific Humidity (kg/kg)', fontsize = 20)

# Mark location 
ax.plot(205.0, 20.0, 'r*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='r',transform=ccrs.PlateCarree())

plt.savefig('/content/drive/My Drive/ATMS_597_Project_3_Group_D/seasonal_anomaly_850hPa_all.png')

## Skin Temperature and Surface Winds 
# Import netcdf files (previously created) with weather fields on extreme precipitation days 
sktemp_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/sktemp_ds_yearcombined_extrem.nc', engine='netcdf4') # skin temperature 
surf_windu_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/surf_windu_ds_yearcombined_extrem.nc', engine='netcdf4') # surface u-wind
surf_windv_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/surf_windv_ds_yearcombined_extrem.nc', engine='netcdf4') # surface v-wind

# Calculate seasonal anomalies: subtract seasonal global long term mean from the mean on extreme precipitation days 
sa_sktemp_ds_yearcombined_extrem = sktemp_ds_yearcombined_extrem.mean(dim='time')-273.15 - skin_temp_ltm_OND # skin temperature, converted to C 
sa_surf_windu_ds_yearcombined_extrem = surf_windu_ds_yearcombined_extrem.mean(dim='time') - surf_wind_u_ltm_OND # surface u-wind
sa_surf_windv_ds_yearcombined_extrem = surf_windv_ds_yearcombined_extrem.mean(dim='time') - surf_wind_v_ltm_OND # surface v-wind

# Set up empty figure 
plt.figure(figsize=(20,10))

# Define projection 
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))

# Mark location 
ax.plot(205.0, 20.0, 'r*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='r',transform=ccrs.PlateCarree())

# Draw gridlines 
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

# Draw coastlines 
ax.coastlines()

# Add title 
plt.title('Seasonal Anomaly: Skin Temperature and Surface Winds', fontsize = 30)

# Plot filled contours of skin temperature 
c1 = ax.contourf(sa_sktemp_ds_yearcombined_extrem['lon'], sa_sktemp_ds_yearcombined_extrem['lat'], sa_sktemp_ds_yearcombined_extrem['skt'].values.squeeze(), 20, transform=ccrs.PlateCarree(), cmap='seismic')
# Plot wind vectors 
ax.quiver(sa_surf_windu_ds_yearcombined_extrem['lon'][::3], sa_surf_windu_ds_yearcombined_extrem['lat'][::3], sa_surf_windu_ds_yearcombined_extrem['uwnd'][::3,::3].values.squeeze(), sa_surf_windv_ds_yearcombined_extrem['vwnd'][::3,::3].values.squeeze(), 20, transform=ccrs.PlateCarree())

# Add colorbar 
cb = fig.colorbar(c1, ax=ax, shrink=0.8)
cb.ax.tick_params(labelsize = 10)
cb.set_label('Skin Temperature [C]', fontsize = 20)

# Save figure 
plt.savefig('/content/drive/My Drive/ATMS_597_Project_3_Group_D/seasonal_anomaly_sktemp_surfwinds.png')

## Atmospheric Column Water Vapor 

# Import netcdf files (previously created) with weather fields on extreme precipitation days 
pr_wtr_ds_yearcombined_extrem = xr.open_dataset('/content/drive/My Drive/ATMS_597_Project_3_Group_D/fields on extreme precip days/pr_wtr_ds_yearcombined_extrem.nc', engine='netcdf4') # atmospheric column water vapor 

# Calculate seasonal anomalies: subtract seasonal global long term mean from the mean on extreme precipitation days 
sa_pr_wtr_ds_yearcombined_extrem = pr_wtr_ds_yearcombined_extrem.mean(dim='time') - atm_col_wv_ltm_OND # atmospheric column water vapor 

## Atmospheric Column Water Vapor 
# Set up empty figure 
plt.figure(figsize=(20,10))

# Define projection 
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))

# Mark location 
ax.plot(205.0, 20.0, 'b*', markersize = 20, transform=ccrs.PlateCarree())
plt.text(205.0, 20.0,'Hilo',fontsize=20,fontweight='bold',ha='right',va = 'top', color='b',transform=ccrs.PlateCarree())

# Draw gridlines 
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

# Draw coastlines 
ax.coastlines()

# Add title 
plt.title('Seasonal Anomaly: Atmospheric Column Water Vapor', fontsize = 30)

# Plot filled contours of atmospheric column water vapor 
c1 = ax.contourf(sa_pr_wtr_ds_yearcombined_extrem['lon'], sa_pr_wtr_ds_yearcombined_extrem['lat'], sa_pr_wtr_ds_yearcombined_extrem['pr_wtr'].values.squeeze(), 20, transform=ccrs.PlateCarree(), cmap='BrBG', alpha=0.8)

# Add colorbar 
cb = fig.colorbar(c1, ax=ax, shrink=0.8)
cb.ax.tick_params(labelsize = 10)
cb.set_label('Atmospheric Column Water Vapor', fontsize = 20)

# Save figure 
plt.savefig('/content/drive/My Drive/ATMS_597_Project_3_Group_D/seasonal_anomaly_pr_wtr.png')







    



