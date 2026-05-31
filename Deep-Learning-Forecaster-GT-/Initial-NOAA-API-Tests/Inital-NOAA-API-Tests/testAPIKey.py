import requests
import argparse
from datetime import date, timedelta
from time import sleep

#Contsants for area specific temperature data
baseURL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
countyID = "FIPS:34027"
dataID = "GHCND"

#Format resulting weather data

tempDataTypes = {
    "tempMin":"Minimum temperature",
    "tempMax":"Maximum temperature",
    "tempAvg":"Average temperature"
}

#Temperature stored in terms of tenths of celsius
#Functon for conversion is needed
def tempConvert(value):
    tC = value/10
    return round((tC*9/5+32),1)

#Accessing all stations connected with the constant countyID
def getStations(token:str)->list[dict]:
    #Access all weather stations in region countyID
    headers = {"token": token}
    stations = []
    offset = 1

    while True:
        params = {
            "dataSetID": dataID,
            "locationID": countyID,
            "dataTypeID": "tempMax",
            "limit": 1000,
            "offset": offset
        }

        resp = requests.get(f"{baseURL}/stations", headers=headers, params=params)
        resp.raise_for_status()
        data=resp.json()

        #Save results
        results = data.get("results", [])
        if(not results):
            break

        stations.extend(results)
        total = data.get("metadata", {}).get("resultset",{}).get("count",len(stations))
        
        if(len(stations>=total)):
            break
        offset+=len(results)
        sleep(0.5)

    return stations

def getTemperatureData(token: str, stationID: str, start: str, end: str)->list[dict]:
    #Use API key to access daily records of temperature at various weather stations within countyID
    headers = {"token":token}
    records = []
    offset = 1

    while True:
        params = {
            "dataSetID": dataID,
            "stationID": stationID,
            "dataTypeID": list(tempDataTypes.keys()),
            "startData": start,
            "endData": end,
            "limit": 1000,
            "offset": offset,
            "units": "standard"
        }

        resp = requests.get(f"{baseURL}/data", headers=headers, params=params)
        resp.raise_for_status()
        data=resp.json()

        results = data.get("results", [])
        if not results:
            break

        records.extend(results)
        total = data.get("metadata", {}).get("resultset", {}).get("count", len(records))

        if(len(records)>=total):
            break
        offset+=len(results)
        sleep(0.5)
    
    return records
