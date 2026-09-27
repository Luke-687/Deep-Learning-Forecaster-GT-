import openmeteo_requests
import numpy as np
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

washingtonLat, washingtonLong = 44.2705, -71.30325

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": washingtonLat,
	"longitude": washingtonLong,
	"daily": ["temperature_2m_min", "temperature_2m_max"],
	"hourly": ["temperature_2m", "temperature_850hPa", "temperature_700hPa", "temperature_500hPa", "relative_humidity_700hPa", "dew_point_2m", "wind_direction_10m", "wind_speed_10m", "wind_speed_850hPa", "wind_direction_850hPa", "geopotential_height_500hPa", "pressure_msl", "total_column_integrated_water_vapour", "dew_point_700hPa"],
	"timezone": "America/New_York",
	"forecast_days": 1,
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_temperature_850hPa = hourly.Variables(1).ValuesAsNumpy()
hourly_temperature_700hPa = hourly.Variables(2).ValuesAsNumpy()
hourly_temperature_500hPa = hourly.Variables(3).ValuesAsNumpy()
hourly_relative_humidity_700hPa = hourly.Variables(4).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(5).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(6).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(7).ValuesAsNumpy()
hourly_wind_speed_850hPa = hourly.Variables(8).ValuesAsNumpy()
hourly_wind_direction_850hPa = hourly.Variables(9).ValuesAsNumpy()
hourly_geopotential_height_500hPa = hourly.Variables(10).ValuesAsNumpy()
hourly_pressure_msl = hourly.Variables(11).ValuesAsNumpy()
hourly_total_column_integrated_water_vapour = hourly.Variables(12).ValuesAsNumpy()
hourly_dew_point_700hPa = hourly.Variables(13).ValuesAsNumpy()

#Open-Meteo does not expose specific humidity directly, so derive it at 700hPa from dew point.
#Air is saturated at its own dew point, so the Magnus-Tetens formula evaluated at Td gives the
#actual (not saturation) vapor pressure; the standard specific humidity formula follows from that
#(both vapor pressures in hPa, matching the 700hPa level, output in kg/kg)
actual_vapor_pressure_700hPa = 6.112 * np.exp(17.62 * hourly_dew_point_700hPa / (243.12 + hourly_dew_point_700hPa))
hourly_specific_humidity_700hPa = 0.622 * actual_vapor_pressure_700hPa / (700 - 0.378 * actual_vapor_pressure_700hPa)

hourly_data = {
	"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	).tz_convert(response.Timezone().decode())
}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["temperature_850hPa"] = hourly_temperature_850hPa
hourly_data["temperature_700hPa"] = hourly_temperature_700hPa
hourly_data["temperature_500hPa"] = hourly_temperature_500hPa
hourly_data["relative_humidity_700hPa"] = hourly_relative_humidity_700hPa
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_speed_850hPa"] = hourly_wind_speed_850hPa
hourly_data["wind_direction_850hPa"] = hourly_wind_direction_850hPa
hourly_data["geopotential_height_500hPa"] = hourly_geopotential_height_500hPa
hourly_data["pressure_msl"] = hourly_pressure_msl
hourly_data["total_column_integrated_water_vapour"] = hourly_total_column_integrated_water_vapour
hourly_data["dew_point_700hPa"] = hourly_dew_point_700hPa
hourly_data["specific_humidity_700hPa"] = hourly_specific_humidity_700hPa

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)
#Save the produced data frame to test what variables can be accessed and what transformations of information must be made, including units
hourly_dataframe.to_csv('TestOpenMeteoDataFrameOutputForWash.csv', index=False)