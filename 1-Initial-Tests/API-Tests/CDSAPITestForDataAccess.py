import cdsapi
client = cdsapi.Client()
client.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': ['reanalysis'],
        'variable': ['2m_temperature'],
        'year': ['2000'], 'month':['01'], 'day':['01'],
        'time':['12:00'],
        'area':[44.5, -71.6, 44.0, -71.0],
        'data_format': 'netcdf',
    },
    'test.nc'
)