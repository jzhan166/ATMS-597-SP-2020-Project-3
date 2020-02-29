""" TELECONNECTIONS """

# This is a part of a class project for ATMS 597, Spring 2020, at the University of Illinois at Urbana Champaign.
# Create code using python xarray to organize and reduce climate data. 
# The goal of this analysis will be to detect global atmospheric circulation patterns (or teleconnections) 
# associated with extreme daily precipitation in a certain part of the globe. 

# Created by Puja Roy, Joyce Yang and Zhang Jun. 

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

    



