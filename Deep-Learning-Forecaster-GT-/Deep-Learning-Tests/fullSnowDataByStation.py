import requests
import argparse
from datetime import date, timedelta
from time import sleep
import pandas as pd


# Constants of location and data set access
#Count_id will now be an input when using the terminal to execute the program
BASE_URL   = "https://www.ncei.noaa.gov/cdo-web/api/v2"
DATASET_ID = "GHCND"        # Global Historical Climatology Network — Daily

# Temperature data type IDs given by NOAA
DATATYPES = {
    "TMAX": "Max Temperature (°F × 10)",
    "TMIN": "Min Temperature (°F × 10)",
    "PRCP": "Total percipitation amount",
    "WT18": "Flag for snow and ice",
    "WT17": "Flag for freezing rain",
    "RHMN": "Minimum relative humidity",
    "RHMX": "maximum relative humidity",
    "AWND": "Average wind speed",
    "WDF5": "5 second fastest wind gust",
    "SNOW": "Snowfall measured in inches",
    "SNWD": "Snow depth on ground in inches"
}

def getStations(token: str, counties: list[str]) -> list[dict]:
    headers = {"token": token}
    stations = []
    for county in counties:
        offset = 1
        while True:
            params = {
                "datasetid":  DATASET_ID,
                "locationid": county,
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
            sleep(5)  #Delay before next API request, mindful of daily limits

    return stations

def getStationSnowData(token: str, station_id: str, start: str, end: str) -> list[dict]:
    headers = {"token": token}
    records = []
    offset = 1

    while True:
        params = {
            "datasetid":  DATASET_ID,
            "stationid":  station_id,
            "datatypeid": list(DATATYPES.keys()),
            "startdate":  start,
            "enddate":    end,
            "limit":      1000,
            "offset":     offset,
            "units":      "standard"
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
        sleep(5)

    return records

def createDataFrame(station: dict, records: list[dict]):
    #Define constants for all rows appended for the given station
    stationName = station['name']
    stationElevation = station.get('elevation', 'n/A')

    #Check that there is temperature data to record
    if not records:
        print("No trackable data found for this period.")
        return

    # Group records by date, then append each dated entry into the data frame for the station
    by_date: dict[str, dict] = {}
    for rec in records:
        date_str = rec["date"][:10]  # YYYY-MM-DD
        if date_str not in by_date:
            by_date[date_str] = {}
        by_date[date_str][rec["datatype"]] = rec["value"]

    allStationData = []
    for date_str in sorted(by_date.keys()):
        day=by_date[date_str]
        allStationData.append({
            "Station": stationName,
            "Date": date_str,
            "Elevation": stationElevation,
            "TMin": day.get('TMIN'),
            "TMax":day.get('TMAX'),
            "Percipitation":day.get("PRCP"),
            "Snow/Ice":day.get("WT18"),
            "FreezeRain":day.get("WT17"),
            "HumidMin":day.get("RHMN"),
            "HumidMax":day.get("RHMX"),
            "WindAvg":day.get("AWND"),
            "5SecGust":day.get("WDF5"),
            "Snow":day.get("SNOW"),
            "SnowDepth":day.get("SNWD")
        })

    stationSnowData = pd.DataFrame(allStationData)
    return stationSnowData

#Main function with all parser variables for input of token and county codes
def main():

    parser = argparse.ArgumentParser(
        description="Fetch NOAA temperature data for all weather stations within counties."
    )
    parser.add_argument(
        "--token", required=True,
        help="Your NOAA CDO API token (get one free at https://www.ncdc.noaa.gov/cdo-web/token)"
    )
    parser.add_argument(
        "--start", default=str(date.today() - timedelta(days=100)),
        help="Start date YYYY-MM-DD (default: 100 days ago)"
    )
    parser.add_argument(
        "--end", default=str(date.today() - timedelta(days=1)),
        help="End date YYYY-MM-DD (default: yesterday)"
    )
    parser.add_argument(
    "--counties", nargs="+", default=["FIPS:41005"], #Default set to Clackamas, OR
    help="One or more FIPS county codes"
    )
    args = parser.parse_args()

    print(f"\nNOAA — Snow Data")
    print(f"Date range : {args.start}  →  {args.end}")

    #Access and save all stations within inputted county codes
    stations = getStations(args.token, args.counties)
    if not stations:
        print("No stations found. Check your API token or date range.")
        return
    print(f"\nFound {len(stations)} station(s). Fetching snow data.\n")
    
    #Define selection logic from the stations found
    for i in range(1,len(stations)+1):
        index = i-1
        print(f"{i}. {stations[index]['name']}\n")
    stationSelectionIndex = input("\nWhich station would you like to access snow data from?")
    stationIndex = stationSelectionIndex-1
    stations = [stations[stationIndex]]
        
    #All station data is filled with only the data of a single station, future iterations of this program will include a multi station select, so the array has not been deleted
    allStationData = []
    for i, station in enumerate(stations, 1):
        print(f"[{i}/{len(stations)}] {station['name']} ({station['id']})")
        try:
            records = getStationSnowData(args.token, station["id"], args.start, args.end)
            stationDataFrame = createDataFrame(station, records)
            if(stationDataFrame is not None):
                allStationData.append(stationDataFrame)
        except requests.HTTPError as e:
            print(f"HTTP error for {station['id']}: {e}")
        sleep(0.25)
    #All data collected, this is redundant but will stay for future multi station alterations 
    if(allStationData is not None):
        snowDataFrame = pd.concat(allStationData, ignore_index=True)
   
    print(f"\n\nDone. Processed {len(stations)} station(s). CSV created.")
    
    countiesString = "-".join(args.counties).replace("FIPS:","")
    snowDataFrame.to_csv(f'snowData({countiesString})({args.start}-{args.end}).csv', index=False)

if __name__ == "__main__":
    main()