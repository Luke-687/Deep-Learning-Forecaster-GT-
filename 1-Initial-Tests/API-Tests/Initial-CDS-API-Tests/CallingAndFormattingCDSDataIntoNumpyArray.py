#Working through the input vector to snow prediction formating for training samples
#Neural network training demands a rigid input to output foundation, where key 62 weather metrics are directly associated with the resulting snowfall and related confidence metric
#Weather metrics which will ultimately be the inputs into the neural network are sourced from multiple data repositories, making not just data collection but data format essential

#Creator: Luke O'Callaghan
#Date: 9/1/26
#Notes: Defining the data collection protocol that will enable an input-output split of weather metrics to resulting snow that the neural network needs for deep learning
#Last Edited: 9/1/26

#Necessary libraries
import cdsapi
import xarray as xr
import numpy as np

#CDS API - Pressure and Surface Level Data - 60/62 Input Metrics
weatherClient = cdsapi.Client()
year = '1970' #Range from 1950 to Present
month = '01'
daysInMonth = '31'

#Redefine the request for each day of interest
#PRESSURE LEVEL REQUEST
pressureRequest = {
    'product_type': ['reanalysis'],
    'variable': ['temperature', 'relative_humidity', 'specific_humidity', 'geopotential', 'vertical_velocity', 'u_component_of_wind', 'v_component_of_wind'],
    'pressure_level': ['500', '700', '850'],
    'year': [year], 'month': [month], 'day': [f'{d:02d}' for d in range(1,int(daysInMonth)+1)],
    'time': ['00:00', '06:00', '12:00', '18:00'],
    'area': [44.28, -71.32, 44.26, -71.29],
    'data_format': 'netcdf',
}
#Call to CDS API with historic pressure level request and format results in Network Common Data Form (Binary)
weatherClient.retrieve('reanalysis-era5-pressure-levels', pressureRequest, 'era5_pl_1970_01.nc')

#SURFACE REQUEST
singleRequest = {
    'product_type': ['reanalysis'],
    'variable': ['2m_temperature', '2m_dewpoint_temperature', 'mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', 'total_column_water_vapour'],
    'year': [year], 'month': [month], 'day': [f'{d:02d}' for d in range(1, int(daysInMonth)+1)],
    'time': ['00:00', '06:00', '12:00', '18:00'],
    'area': [44.5, -71.6, 44.0, -71.0],
    'data_format': 'netcdf',
}
#Call to CDS API with historic surface level request and format results in Network Common Data Form (Binary)
weatherClient.retrieve('reanalysis-era5-single-levels', singleRequest, 'era5_sl_1970_01.nc')

#Read the resulting data, this test functions only with a single month of data from 1970
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
    dailyData = []
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
print(resultingWeatherData)
np.save('mtwashington_cdsapi_test_january_1970.npy', resultingWeatherData, allow_pickle=True)
    
