#Creator: Luke O'Callaghan
#Date: 9/1/26
#Notes: Full data pipeline including CDS API and NOAA GHCND calls with formated numpy sample vector storage
#Last Edited: 9/1/26

#Necessary libraries
import cdsapi
import xarray as xr
import numpy as np
import argparse
import requests
import time
import os
from dotenv import load_dotenv
from datetime import date

#CDS API - Pressure and Surface Level Data - 60/62 Input Metrics
weatherClient = cdsapi.Client()

#NOAA GHCND API - Load token from .env instead of hardcoding it in source
load_dotenv()
NOAA_TOKEN = os.environ['NOAA_TOKEN']

#Loop through an inputted year, accessing all months and a key of days in that month
def monthDays(monthIndex, year):
    if((monthIndex == 1) or (monthIndex == 3) or (monthIndex == 5) or (monthIndex == 7) or (monthIndex == 8) or (monthIndex == 10) or (monthIndex == 12)):
        return 31
    elif((monthIndex == 4) or (monthIndex == 6) or (monthIndex == 9) or (monthIndex == 11)):
        return 30
    elif((monthIndex == 2)): #Leap year logic
        if((year%4==0 and not year%100==0) or (year%400==0)):
            return 29
        else:
            return 28
    else:
        return 30
    
def main():
    #ACCESS PARSER INPUTS
    inputs = buildingParser().parse_args()
    year = inputs.year_of_interest
    #NECESSARY CONTEXT AND SPECIFICATIONS
    washingtonLat, washingtonLong = 44.2705, -71.30325
    #Define presure specifications to direct the data selection loop to only search for instances of certain variables at certain specified levels
    levelsSpec = [('t', 850, 'T_850'), ('t', 700, 'T_700'), ('t', 500, 'T_500'),('r', 700, 'RH_700'), ('q', 700, 'Q_700'), ('z', 500, 'Z_500'),('w', 700, 'W_700'), ('u', 850, 'U_850'), ('v', 850, 'V_850'),]
    #Define surface specifications such that data can be tagged during these early stage tests, this will not be necessary in the future
    surfaceSpec = [('t2m', 'T_2m'), ('d2m', 'Td_2m'), ('msl', 'MSLP'),('u10', 'U_10'), ('v10', 'V_10'), ('tcwv', 'TCWV'),]
    #ACCESS DATA FROM EACH MONTH, COMBINE FOR YEARLY DATA
    fullYearOfData = np.array([], dtype=object)
    for month in range(1,13):
        daysInMonth = monthDays(month, year)
        fullMonthOfData = [] #Clear previous daily weather metric and ouput vectors for a month
        # **** CDS API CALLS ****
        #PRESSURE LEVEL REQUEST
        pressureRequest = {
            'product_type': ['reanalysis'],
            'variable': ['temperature', 'relative_humidity', 'specific_humidity', 'geopotential', 'vertical_velocity', 'u_component_of_wind', 'v_component_of_wind'],
            'pressure_level': ['500', '700', '850'],
            'year': [year], 'month': [month], 'day': [f'{d:02d}' for d in range(1,int(daysInMonth)+1)],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            #DEBUG FIX: original box (44.28,-71.32,44.26,-71.29) is ~0.02x0.03 deg, smaller than the
            #ERA5 0.25deg grid spacing, so MARS finds zero grid points inside it and returns 400
            #("Assertion failed: Area: non-empty area crop/mask"). Widened to match the surface box.
            'area': [44.5, -71.6, 44.0, -71.0],
            'data_format': 'netcdf',
        }
        weatherClient.retrieve('reanalysis-era5-pressure-levels', pressureRequest, f'era5_pl_{year}_{month}.nc')  #Network Common Data Form (Binary)
        #SURFACE REQUEST
        singleRequest = {
            'product_type': ['reanalysis'],
            'variable': ['2m_temperature', '2m_dewpoint_temperature', 'mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', 'total_column_water_vapour'],
            'year': [year], 'month': [month], 'day': [f'{d:02d}' for d in range(1, int(daysInMonth)+1)],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'area': [44.5, -71.6, 44.0, -71.0],
            'data_format': 'netcdf',
        }
        weatherClient.retrieve('reanalysis-era5-single-levels', singleRequest, f'era5_sl_{year}_{month}.nc')  #Network Common Data Form (Binary)
        # **** READ AND RECORD CDS DATA ****
        dsLevels = xr.open_dataset(f'era5_pl_{year}_{month}.nc')
        dsSurface = xr.open_dataset(f'era5_sl_{year}_{month}.nc')
        #SIMPLIFY TO COLUMN CONTAINING MT.WASHINGTON SUMMIT
        dsWashLevels = dsLevels.sel(latitude = washingtonLat, longitude = washingtonLong, method='nearest')
        dsWashSurface = dsSurface.sel(latitude = washingtonLat, longitude = washingtonLong, method='nearest')
        #APPEND ALL DAILY DATA FOR MONTH
        for day, dsDayLevels in dsWashLevels.groupby(dsWashLevels.valid_time.dt.date):
            dsDaySurface = dsWashSurface.sel(valid_time=dsDayLevels.valid_time)
            #Empty array for each new day
            dailyData = [day]
            #dsDayLevels and dsDaySurface hold all important values for a single day data set
            for timestamp in dsDayLevels.valid_time.values:
                for var, lvl, tag in levelsSpec:
                    fullTagLevels = f'{tag}_t{list(dsDayLevels.valid_time.values).index(timestamp)}'
                    value = float((dsDayLevels[var].sel(pressure_level=lvl, valid_time=timestamp)).values)
                    dailyData.append((fullTagLevels,value))
                for var, tag in surfaceSpec:
                    fullTagSurface = f'{tag}_t{list(dsDayLevels.valid_time.values).index(timestamp)}'
                    value = float((dsDaySurface[var].sel(valid_time=timestamp)).values)
                    dailyData.append((fullTagSurface, value))
            fullMonthOfData.append(dailyData)
        #COMBINE SINGLE MONTH CDS DATA WITH FULL YEAR CDS DATA
        #DEBUG FIX: np.concatenate returns a new array, it does not mutate in place - the result was
        #previously discarded, leaving fullYearOfData empty for the rest of the script.
        #DEBUG FIX: np.array(fullMonthOfData, dtype=object) collapses into a 2D array whenever every
        #day happens to have the same number of entries, which then fails to concatenate with the
        #1D fullYearOfData. Force a 1D array of day-list objects via slice assignment instead.
        monthArr = np.empty(len(fullMonthOfData), dtype=object)
        monthArr[:] = fullMonthOfData
        fullYearOfData = np.concatenate((fullYearOfData, monthArr))
    # **** ACCESS FULL YEAR GHCND DATA ****
    fullDataGHCND = [] #Will eventually feed into the fullYearOfData numpy array
    offset = 1
    while True:
        #DEBUG FIX: NOAA's CDO API caps requests at 5/sec and was returning 429 Too Many Requests
        #with no delay between pagination calls. Pace every call, and back off further if still 429.
        for attempt in range(5):
            time.sleep(1)
            resp = requests.get(
                'https://www.ncei.noaa.gov/cdo-web/api/v2/data',
                headers={'token': NOAA_TOKEN},
                    params={
                    'datasetid': 'GHCND',
                    'stationid': 'GHCND:USW00014755',
                    'datatypeid': ['TMIN', 'TMAX', 'SNOW'],
                    'startdate': f'{year}-01-01',
                    'enddate': f'{year}-12-31',
                    'units': 'metric',
                    'limit': 1000,
                    'offset': offset,
                },
            )
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            break
        resp.raise_for_status()
        responsePayload = resp.json()
        results = responsePayload.get('results', [])
        if not results:
            break
        fullDataGHCND.extend(results)
        if len(results)<1000:
            break #Final break when fullDataGHCND is entirely filled with essential weather metric and resulting snow data
        #DEBUG FIX: offset was never advanced, so a full (1000-result) page would re-request the
        #same page forever - an infinite loop. NOAA's offset is 1-based, so advance by the page size.
        offset += 1000
    # **** PARSE AND GROUP GHCND DATA ****
    byDateGHCND = {} #Formated into [date, [tmin, val], [tmax,val], [snow,val]] for each date
    #DEBUG FIX: iterating over `results` only covered the final pagination page. Should walk the
    #full accumulated list built across every page of the while loop above.
    for dataRow in fullDataGHCND: #dataRow has the 3 data points grouped into date
        day = date.fromisoformat(dataRow['date'][:10])
        datatype = dataRow['datatype']
        value = dataRow['value']
        if datatype in ('TMIN', 'TMAX', 'SNOW'):
            byDateGHCND.setdefault(day,[]).append((datatype, value)) #Tag each data point
    # ***** COMBINE GHCND DATA WITH FULL YEAR CDS DATA ****
    #Loop through every day in the year and check
    #DEBUG FIX: `index` was never incremented, so this always compared against fullYearOfData[0]
    #(and would IndexError immediately once fullYearOfData was empty, per the concatenate fix above).
    #Look days up by date instead of assuming both lists are in lockstep order.
    dateToIndex = {row[0]: i for i, row in enumerate(fullYearOfData)}
    for day in byDateGHCND:
        #Fix data formating in each daily byDateGHCND
        dayDataDictionary = dict(byDateGHCND[day])
        dataToAppendGHCND = [('TMIN', dayDataDictionary.get('TMIN')),('TMAX', dayDataDictionary.get('TMAX')),('SNOW', dayDataDictionary.get('SNOW'))]
        #Input the 'day' into the fullYearOfData
        index = dateToIndex.get(day)
        if index is not None: #Date matching check:
            dataToAppendGHCND = np.array(dataToAppendGHCND, dtype=object)
            fullYearOfData[index].append(dataToAppendGHCND) #Appends the data from GHCND into the full yearly data logs
    # **** RETURN FINAL DATA RESULTS ****
    return fullYearOfData
        
        
def buildingParser():
    parser = argparse.ArgumentParser(description='Takes input of year data will be accessed from for CDS and GHCND API calls')
    parser.add_argument('--year_of_interest', type=int, default=1950)
    return parser

if __name__ == '__main__':
    results = main()
    print(results[0]) #Print a single sample of day level data from the results of data collection and combination
    np.save('mtwashington1970TestFullDataPull.npy', results, allow_pickle=True)