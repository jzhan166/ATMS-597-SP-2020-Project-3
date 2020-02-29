# ATMS-597-SP-2020-Project-3

## Teleconnections

The goal of this analysis is to detect global atmospheric circulation patterns (or teleconnections) associated with extreme daily precipitation in a certain part of the globe.
We are looking at a very beautiful town of Hawaii - Hilo, in the middle of the Pacific Ocean! We are trying to correlate the extreme precipitation signatures for this station during the months of October, November and December for the years 1996 to 2019 to the global scale circulation patterns.

For this project, we had to :
(1) Aggregate daily rainfall data from the Global Precipitaiton Climatology Project 1 degree daily precipitation data over the period 1996 - 2019 into a single file from daily files, available here: [https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/].

(2) Determine the 95% values of daily precipitation during a selected particular 3-month period (given in the table below by group) over the grid box closest to Hilo and plot a cumulative distribution function of all values daily precipitation values, illustrating the 95% value of daily precipitation in millimeters.

(3) Use output from the NCEP Reanalysis [https://journals.ametsoc.org/doi/pdf/10.1175/1520-0477(1996)077%3C0437%3ATNYRP%3E2.0.CO%3B2](Kalnay et al. 1996) to compute the global mean fields and seasonal anomaly fields for days meeting and exceeding the threshold of precipitation calculated in the previous step (using the 1981-2010 as a base period for anomalies) of

250 hPa wind vectors and wind speed,
500 hPa wind vectors and geopotential height,
850 hPa temperature, specific humidity, and winds,
skin temperature, and surface wind vectors (sig995 level), and
total atmospheric column water vapor.

(4) Create maps showing the mean fields for the extreme precipitation day composites, long term mean composites for the selected months, and the anomaly fields for each variable, using contours and vectors whenever appropriate.

## References:

1. Daily rainfall data from the Global Precipitaiton Climatology Project 1 degree daily precipitation data over the period 1996 - 2019 available here: https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/

2. NCEP reanalysis data is available from the NOAA Physical Sciences Division THREDDS server:
https://www.esrl.noaa.gov/psd/thredds/catalog/Datasets/catalog.html


## Project Team:
- Puja Roy
- Joyce Yang
- Zhang Jun
