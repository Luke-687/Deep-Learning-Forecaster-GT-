#Creator: Luke O'Callaghan
#Notes: Loads the trained snow network and runs it against live Open-Meteo column data to forecast today's snowfall
#Open-Meteo call structure reused from 1-Initial-Tests/API-Tests/Initial-Open-Meteo-Tests/OpenMeteoRequestTest.py

#Necessary libraries
import openmeteo_requests
import numpy as np
import pandas as pd
import requests_cache
import torch
import os
import argparse
import importlib.util
from retry_requests import retry

#Resolve paths off this file so the script runs from any working directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
#The model and training folders are siblings of this script's folder, not children of it
NETWORK_ROOT = os.path.dirname(THIS_DIR)
MODEL_DIR = os.path.join(NETWORK_ROOT, 'Save-Trained-Models')
NETWORK_DEFINITION = os.path.join(NETWORK_ROOT, 'Train-Models', 'NeuralNetwork(2000|2020).py')

washingtonLat, washingtonLong = 44.2705, -71.30325

#ERA5 samples the day at these four UTC hours, the live vector has to be built from the same four
TRAINING_HOURS_UTC = [0, 6, 12, 18]
#Standard gravity, converts Open-Meteo geopotential height in m to the ERA5 geopotential in m2/s2
GRAVITY = 9.80665

# **** LOAD THE TRAINED NETWORK ****

#The training file name holds parentheses and a pipe, so import it by path rather than by module name
#Loading it this way means the network class and the column order come from one source of truth
def loadNetworkDefinition():
    spec = importlib.util.spec_from_file_location('snowNetworkDefinition', NETWORK_DEFINITION)
    definition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(definition)
    return definition

def loadTrainedModel(definition, trainLabel):
    checkpointPath = os.path.join(MODEL_DIR, f'snowNetwork({trainLabel}).pt')
    if not os.path.exists(checkpointPath):
        raise FileNotFoundError(f'No trained model found, run NeuralNetwork({trainLabel}).py first: {checkpointPath}')
    network, metricMeans, metricDeviations = definition.loadNetwork(checkpointPath)
    checkpoint = torch.load(checkpointPath, weights_only=False)
    print(f'Loaded model  {os.path.basename(checkpointPath)}')
    print(f'  best training epoch {checkpoint["bestEpoch"]}')
    scores = checkpoint['testScores']
    print(f'  2021 test scores  MAE {scores["mae"]:.2f} mm, occurrence accuracy {scores["accuracy"]:.3f}, brier {scores["brier"]:.4f}')
    return network, metricMeans, metricDeviations

# **** OPEN METEO CALL ****

#Reuses the cached and retrying session from OpenMeteoRequestTest.py
def buildOpenMeteoClient():
    cache_session = requests_cache.CachedSession(os.path.join(THIS_DIR, '.cache'), expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    return openmeteo_requests.Client(session = retry_session)

#The 13 hourly variables needed to rebuild all 14 training metrics, order matters for Variables(i) below
OPEN_METEO_HOURLY = [
    'temperature_2m', 'temperature_850hPa', 'temperature_700hPa', 'temperature_500hPa',
    'relative_humidity_700hPa', 'dew_point_2m', 'dew_point_700hPa',
    'wind_direction_10m', 'wind_speed_10m', 'wind_speed_850hPa', 'wind_direction_850hPa',
    'geopotential_height_500hPa', 'pressure_msl', 'total_column_integrated_water_vapour',
]

def pullLiveWeather(openmeteo, forecastDays):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": washingtonLat,
        "longitude": washingtonLong,
        "hourly": OPEN_METEO_HOURLY,
        #UTC, not local time, the training timestamps are UTC hours and a local offset would shift every metric
        "timezone": "UTC",
        #Wind arrives in km/h unless this is set, which would inflate every wind component by 3.6
        "wind_speed_unit": "ms",
        "forecast_days": forecastDays,
    }
    responses = openmeteo.weather_api(url, params = params)

    #Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f'\nOpen-Meteo grid point  {response.Latitude():.4f}N {response.Longitude():.4f}E, elevation {response.Elevation():.0f} m')

    hourly = response.Hourly()
    hourlyValues = {name: hourly.Variables(i).ValuesAsNumpy() for i, name in enumerate(OPEN_METEO_HOURLY)}

    #Timestamps stay in UTC so the 00/06/12/18 rows can be selected directly
    hourlyTimes = pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left",
    )
    return hourlyTimes, hourlyValues

# **** UNIT CONVERSION TO TRAINING FORM ****

#Open-Meteo does not expose specific humidity directly, so derive it at 700hPa from dew point.
def specificHumidity700(dewPoint700Celsius):
    actual_vapor_pressure_700hPa = 6.112 * np.exp(17.62 * dewPoint700Celsius / (243.12 + dewPoint700Celsius))
    return 0.622 * actual_vapor_pressure_700hPa / (700 - 0.378 * actual_vapor_pressure_700hPa)

#ERA5 stores wind as u and v components, Open-Meteo reports speed with the direction the wind blows from.
#Meteorological convention means a 270 degree wind comes out of the west and pushes air east, so both
#components carry a negative sign (speed in m/s, direction in degrees, output in m/s)
def windComponents(windSpeed, windDirectionDegrees):
    directionRadians = np.deg2rad(windDirectionDegrees)
    eastwardComponent = -windSpeed * np.sin(directionRadians)
    northwardComponent = -windSpeed * np.cos(directionRadians)
    return eastwardComponent, northwardComponent

#Rebuild the 14 ERA5 training metrics for a single hour, in the exact order YearSnowMetricPull.py appended them
def metricsForHour(hourlyValues, hourIndex):
    pick = lambda name: float(hourlyValues[name][hourIndex])

    #Open-Meteo reports Celsius, ERA5 trained on Kelvin
    temperature850 = pick('temperature_850hPa') + 273.15
    temperature700 = pick('temperature_700hPa') + 273.15
    temperature500 = pick('temperature_500hPa') + 273.15
    relativeHumidity700 = pick('relative_humidity_700hPa')
    specificHumidity = specificHumidity700(pick('dew_point_700hPa'))
    #Open-Meteo reports geopotential height in m, ERA5 stores geopotential in m2/s2
    geopotential500 = pick('geopotential_height_500hPa') * GRAVITY
    eastward850, northward850 = windComponents(pick('wind_speed_850hPa'), pick('wind_direction_850hPa'))
    temperature2m = pick('temperature_2m') + 273.15
    dewPoint2m = pick('dew_point_2m') + 273.15
    #Open-Meteo reports mean sea level pressure in hPa, ERA5 stores it in Pa
    meanSeaLevelPressure = pick('pressure_msl') * 100.0
    eastward10, northward10 = windComponents(pick('wind_speed_10m'), pick('wind_direction_10m'))
    totalColumnWaterVapour = pick('total_column_integrated_water_vapour')

    return [temperature850, temperature700, temperature500, relativeHumidity700, specificHumidity,
            geopotential500, eastward850, northward850, temperature2m, dewPoint2m,
            meanSeaLevelPressure, eastward10, northward10, totalColumnWaterVapour]

#Assemble one 56 value input vector per forecast day, timestamp major exactly like the training samples
def buildDailyVectors(hourlyTimes, hourlyValues):
    forecastDates = sorted({timestamp.date() for timestamp in hourlyTimes})
    dailyVectors = []
    coveredDates = []
    for forecastDate in forecastDates:
        hourIndices = []
        for targetHour in TRAINING_HOURS_UTC:
            matches = [i for i, timestamp in enumerate(hourlyTimes)
                       if timestamp.date() == forecastDate and timestamp.hour == targetHour]
            if matches:
                hourIndices.append(matches[0])
        #A partial day cannot fill the 56 slots, skip it rather than pad with nonsense
        if len(hourIndices) != len(TRAINING_HOURS_UTC):
            print(f'  skipping {forecastDate}, only {len(hourIndices)} of {len(TRAINING_HOURS_UTC)} required UTC hours returned')
            continue
        dailyVector = []
        for hourIndex in hourIndices:
            dailyVector.extend(metricsForHour(hourlyValues, hourIndex))
        dailyVectors.append(dailyVector)
        coveredDates.append(forecastDate)
    return np.array(dailyVectors, dtype=np.float32), coveredDates

# **** SANITY CHECK AGAINST TRAINING RANGE ****

#A unit mistake shows up as a metric sitting many deviations from everything the network trained on
def checkAgainstTrainingRange(liveVectors, metricMeans, metricDeviations, featureNames, warnThreshold):
    deviationsFromMean = np.abs((liveVectors - metricMeans) / metricDeviations)
    worstPerFeature = deviationsFromMean.max(axis=0)
    suspectFeatures = np.argsort(-worstPerFeature)[:3]
    if worstPerFeature.max() > warnThreshold:
        print(f'\nWARNING  live metrics sit far outside the training range, check units before trusting the forecast')
        for featureIndex in suspectFeatures:
            if worstPerFeature[featureIndex] > warnThreshold:
                print(f'  {featureNames[featureIndex]:12s} {worstPerFeature[featureIndex]:5.1f} deviations from the training mean')
    else:
        print(f'Unit sanity check passed, every live metric within {worstPerFeature.max():.1f} deviations of the training mean')

# **** MAIN ****

def main():
    inputs = buildingParser().parse_args()

    definition = loadNetworkDefinition()
    network, metricMeans, metricDeviations = loadTrainedModel(definition, inputs.train_label)

    openmeteo = buildOpenMeteoClient()
    hourlyTimes, hourlyValues = pullLiveWeather(openmeteo, inputs.forecast_days)

    liveVectors, coveredDates = buildDailyVectors(hourlyTimes, hourlyValues)
    if len(liveVectors) == 0:
        raise RuntimeError('Open-Meteo returned no day with all four required UTC hours')
    print(f'Built {len(liveVectors)} daily input vector(s) of {liveVectors.shape[1]} metrics')

    checkAgainstTrainingRange(liveVectors, metricMeans, metricDeviations,
                              definition.featureNames(), inputs.warn_deviations)

    #Normalize with the statistics saved alongside the weights, never recomputed from live data
    normalizedVectors = torch.tensor(definition.applyNormalizer(liveVectors, metricMeans, metricDeviations))
    expectedSnow, snowChance, amountIfSnowing = definition.predictSnow(network, normalizedVectors)

    print('\n**** SNOW FORECAST, MT WASHINGTON SUMMIT ****')
    print(f'{"date":<12} {"expected":>10} {"if snowing":>11} {"confidence":>11}')
    for i, forecastDate in enumerate(coveredDates):
        print(f'{str(forecastDate):<12} {expectedSnow[i]:7.1f} mm {amountIfSnowing[i]:8.1f} mm {snowChance[i]:10.0%}')

    #The headline answer for today
    todaysSnow = expectedSnow[0]
    todaysChance = snowChance[0]
    print(f'\nTODAY ({coveredDates[0]}): {todaysSnow:.1f} mm of snow expected, {todaysChance:.0%} chance of any snowfall')

    return expectedSnow, snowChance, coveredDates

def buildingParser():
    parser = argparse.ArgumentParser(description='Runs the trained snow network against live Open-Meteo column data')
    parser.add_argument('--train_label', type=str, default='2000|2020', help='which saved model to load')
    parser.add_argument('--forecast_days', type=int, default=1, help='1 is today, Open-Meteo projects up to 16 days ahead')
    parser.add_argument('--warn_deviations', type=float, default=6.0, help='flag live metrics this far outside the training range')
    return parser

if __name__ == '__main__':
    main()
