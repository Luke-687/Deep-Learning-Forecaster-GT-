#Working through the input vector to snow prediction formating for training samples
#Neural network training demands a rigid input to output foundation, where key 62 weather metrics are directly associated with the resulting snowfall and related confidence metric
#Weather metrics which will ultimately be the inputs into the neural network are sourced from multiple data repositories, making not just data collection but data format essential

#Creator: Luke O'Callaghan
#Date: 9/1/26
#Notes: Defining the data collection protocol that will enable an input-output split of weather metrics to resulting snow that the neural network needs for deep learning
#Last Edited: 9/1/26

#CDS API - Pressure and Surface Level Data - 60/62 Input Metrics
import cdsapi
weatherClient = cdsapi.Client()
year = '1950' #Range from 1950 to Present
month = '01'
daysInMonth = '31'

#PRESSURE LEVEL REQUEST
pressureRequest = {
    'product_type': ['reanalysis'],
    'variable': ['temperature', 'relative_humidity', 'specific_humidity', 'geopotential', 'vertical_velocity', 'u_component_of_wind', 'v_component_of_wind'],
    'pressure_level': ['500', '700', '850'],
    'year': [year], 'month': [month], 'day': [f'{d:02d}' for d in range(1,int(daysInMonth)+1)],
    'time': ['00:00', '06:00', '12:00', '18:00'],
    'area': [44.5, -71.6, 44.0, -71.0],
    'data_format': 'netcdf',
}
#Call to CDS API with historic pressure level request and format results in Network Common Data Form (Binary)
weatherClient.retrieve('reanalysis-era5-pressure-levels', pressureRequest, 'era5_pl_1950_01.nc')

#SURFACE REQUEST
singleRequest = {
    'product_type': ['reanalysis'],
    'variable': ['2m_temperature', '2m_dewpoint_temperature', 'mean_sea_;evel_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', 'total_column_water_vapour'],
    'year': [year], 'month': [month], 'day': [f'{d:02d}' for d in range(1, int(daysInMonth)+1)],
    'time': ['00:00', '06:00', '12:00', '18:00'],
    'area': [44.5, -71.6, 44.0, -71.0],
    'data_format': 'netcdf',
}
#Call to CDS API with historic surface level request and format results in Network Common Data Form (Binary)
weatherClient.request('reanalysis-era5-single-levels', singleRequest, 'era5_sl_1950_01.nc')
