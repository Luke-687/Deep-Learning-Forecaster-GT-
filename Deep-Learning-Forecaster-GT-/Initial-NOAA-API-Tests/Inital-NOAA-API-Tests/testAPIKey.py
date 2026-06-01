import requests
import argparse
from datetime import date, timedelta
from time import sleep
import pandas as pd


# Constants of location and data set access
BASE_URL   = "https://www.ncei.noaa.gov/cdo-web/api/v2"
COUNTY_ID  = "FIPS:34027"   # Morris County
DATASET_ID = "GHCND"        # Global Historical Climatology Network — Daily

# Temperature data type IDs given by NOAA
TEMP_DATATYPES = {
    "TMAX": "Max Temperature (°F × 10)",
    "TMIN": "Min Temperature (°F × 10)",
    "TAVG": "Average Temperature (°F × 10)",
}

temperatureDataFrame = pd.DataFrame() #Define in the format: station-elevation-year-month-day-tempMin-tempMax-tempAvg


#Convert temperature data from tenths of celsius to fahrenheit
def f10_to_f(value):
    celsius = value / 10.0
    return round((celsius * 9 / 5) + 32, 1)


def get_stations(token: str) -> list[dict]:
    headers = {"token": token}
    stations = []
    offset = 1

    print(f"\nFetching stations in {COUNTY_ID}")

    while True:
        params = {
            "datasetid":  DATASET_ID,
            "locationid": COUNTY_ID,
            "datatypeid": "TMAX",       # only stations that have temp data
            "limit":      1000,
            "offset":     offset,
        }
        resp = requests.get(f"{BASE_URL}/stations", headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break

        stations.extend(results)
        total = data.get("metadata", {}).get("resultset", {}).get("count", len(stations))
        print(f"  Retrieved {len(stations)} / {total} stations")

        if len(stations) >= total:
            break
        offset += len(results)
        sleep(0.5)  #Delay before next API request, mindful of daily limits

    return stations


def get_temperature_data(token: str, station_id: str, start: str, end: str) -> list[dict]:
    headers = {"token": token}
    records = []
    offset = 1

    while True:
        params = {
            "datasetid":  DATASET_ID,
            "stationid":  station_id,
            "datatypeid": list(TEMP_DATATYPES.keys()),
            "startdate":  start,
            "enddate":    end,
            "limit":      1000,
            "offset":     offset,
            "units":      "standard",   # returns °F directly (no ×10 conversion needed)
        }
        resp = requests.get(f"{BASE_URL}/data", headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break

        records.extend(results)
        total = data.get("metadata", {}).get("resultset", {}).get("count", len(records))

        if len(records) >= total:
            break
        offset += len(results)
        sleep(0.5)

    return records

def createTemperatureDataFrame(station: dict, records: list[dict]):
    stationTemperatureData = pd.DataFrame(columns=["Station", "Elevation","Date","TMin", "TMax", "TAvg"])
    #Define constants for all rows appended for the given station
    stationName = station['name']
    stationElevation = station.get('elevation', 'n/A')

    #Check that there is temperature data to record
    if not records:
        print("No temperature data found for this period.")
        return

    # Group records by date, then append each dated entry into the data frame for the station
    by_date: dict[str, dict] = {}
    for rec in records:
        date_str = rec["date"][:10]  # YYYY-MM-DD
        if date_str not in by_date:
            by_date[date_str] = {}
        by_date[date_str][rec["datatype"]] = rec["value"]

    rowsForStation = []
    for date_str in sorted(by_date.keys()):
        day=by_date[date_str]
        rowsForStation.append({
            "Station": stationName,
            "Elevation": stationElevation,
            "Date": date_str,
            "TMin": day.get('TMIN'),
            "TMax":day.get('TMAX'),
            'TAvg':day.get('TAVG')
        })

    stationTemperatureData = pd.DataFrame(rowsForStation)
    return stationTemperatureData


def main():
    #Define larger data frame which all station data will be collected in
    temperatureDataFrame = pd.DataFrame(columns=["Station", "Elevation","Date","TMin", "TMax", "TAvg"])

    parser = argparse.ArgumentParser(
        description="Fetch NOAA temperature data for all Morris County, NJ weather stations."
    )
    parser.add_argument(
        "--token", required=True,
        help="Your NOAA CDO API token (get one free at https://www.ncdc.noaa.gov/cdo-web/token)"
    )
    parser.add_argument(
        "--start", default=str(date.today() - timedelta(days=30)),
        help="Start date YYYY-MM-DD (default: 30 days ago)"
    )
    parser.add_argument(
        "--end", default=str(date.today() - timedelta(days=1)),
        help="End date YYYY-MM-DD (default: yesterday)"
    )
    args = parser.parse_args()

    print(f"\nNOAA Morris County, NJ — Temperature Data")
    print(f"Date range : {args.start}  →  {args.end}")

    # 1. Get all stations
    stations = get_stations(args.token)
    if not stations:
        print("No stations found. Check your API token or date range.")
        return

    print(f"\nFound {len(stations)} station(s). Fetching temperature data.\n")

    # 2. Fetch + print data for each station
    allStationData = []

    for i, station in enumerate(stations, 1):
        print(f"[{i}/{len(stations)}] {station['name']} ({station['id']})")
        try:
            records = get_temperature_data(args.token, station["id"], args.start, args.end)
            stationDataFrame = createTemperatureDataFrame(station, records)
            if(stationDataFrame is not None):
                allStationData.append(stationDataFrame)
        except requests.HTTPError as e:
            print(f"HTTP error for {station['id']}: {e}")
        sleep(0.25)
    temperatureDataFrame = pd.concat(allStationData, ignore_index=True)
   
    print(f"\n\nDone. Processed {len(stations)} station(s). CSV created.")

    temperatureDataFrame.to_csv('TestingAPITemperature.csv', index=False)


if __name__ == "__main__":
    main()