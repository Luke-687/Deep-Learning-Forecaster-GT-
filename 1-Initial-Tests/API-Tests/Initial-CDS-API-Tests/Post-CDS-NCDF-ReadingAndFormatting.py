#Creator: Luke O'Callaghan
#Date: 9/1/26
#Notes: Seperate processing environment to isolate the creation of numpy array from a month of data
#Last Edited: 9/1/26

#Read the resulting data, this test functions only with a single month of data from 1970
import xarray as xr
import numpy as np

dsLevels = xr.open_dataset('era5_pl_1970_01.nc')
dsSurface = xr.open_dataset('era5_sl_1970_01.nc')

#Define the exact latitude and longitude position of the Mt. Washington observatory and isolate column data including the coordinates
washingtonLat, washingtonLong = 44.2705, -71.30325
#Simplify data sets down to column index including Mt. Washington observatory
dsWashLevels = dsLevels.sel(latitude = washingtonLat, longitude = washingtonLong, method='nearest')
dsWashSurface = dsSurface.sel(latitude = washingtonLat, longitude = washingtonLong, method='nearest')

#Define presure specifications to direct the data selection loop to only search for instances of certain variables at certain specified levels
levelsSpec = [
    ('t', 850, 'T_850'), ('t', 700, 'T_700'), ('t', 500, 'T_500'),
    ('r', 700, 'RH_700'), ('q', 700, 'Q_700'), ('z', 500, 'Z_500'),
    ('w', 700, 'W_700'), ('u', 850, 'U_850'), ('v', 850, 'V_850'),
]
#Define surface specifications such that data can be tagged during these early stage tests, this will not be necessary in the future
surfaceSpec = [
    ('t2m', 'T_2m'), ('d2m', 'Td_2m'), ('msl', 'MSLP'),
    ('u10', 'U_10'), ('v10', 'V_10'), ('tcwv', 'TCWV'),
]

#Access data from the single column data set, grouped together to represent weather metrics as future inputs in NN training
trainingSamplesTest = [] #Will eventually format into 2D numpy array with each internal sample holding 62 weather metrics and 2 output metrics
for day, dsDayLevels in dsWashLevels.groupby(dsWashLevels.valid_time.dt.date):
    #Empty array for each new day
    dailyData = [day]
    #Set daily surface level data set to same valid times as daily presure level data set
    dsDaySurface = dsWashSurface.sel(valid_time=dsDayLevels.valid_time) #Sets daily surface level data set to same valid times as daily presure level data set
    #dsDayLevels and dsDaySurface hold all important values for a single day data set
    for timestamp in dsDayLevels.valid_time.values:
        for var, lvl, tag in levelsSpec:
            value = float((dsDayLevels[var].sel(pressure_level=lvl, valid_time=timestamp)).values)
            dailyData.append((f'{tag}_t{list(dsDayLevels.valid_time.values).index(timestamp)}', value))
        for var, tag in surfaceSpec:
            value = float((dsDaySurface[var].sel(valid_time=timestamp)).values)
            dailyData.append((f'{tag}_t{list(dsDaySurface.valid_time.values).index(timestamp)}', value))
    trainingSamplesTest.append(dailyData)

resultingWeatherData = np.array(trainingSamplesTest, dtype=object)
print(resultingWeatherData[0])
np.save('mtwashington_cdsapi_test_january_1970.npy', resultingWeatherData, allow_pickle=True)
