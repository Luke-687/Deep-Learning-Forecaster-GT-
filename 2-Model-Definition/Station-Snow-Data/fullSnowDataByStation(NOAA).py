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
    "WSF5": "5 second fastest wind gust",
    "SNOW": "Snowfall measured in inches",
    "SNWD": "Snow depth on ground in inches"
}

def getStationById(token: str, station_id: str) -> dict | None:
    headers = {"token": token}
    resp = requests.get(f"{BASE_URL}/stations/{station_id}", headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data if data.get("id") else None

def getStationsByName(token: str, name_query: str) -> list[dict]:
    headers = {"token": token}
    params = {
        "datasetid": DATASET_ID,
        "limit":     1000,
        "offset":    1,
    }
    resp = requests.get(f"{BASE_URL}/stations", headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    query = name_query.lower()
    return [s for s in results if query in s["name"].lower()]

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
            "5SecGust":day.get("WSF5"),
            "Snow":day.get("SNOW"),
            "SnowDepth":day.get("SNWD")
        })

    stationSnowData = pd.DataFrame(allStationData)
    return stationSnowData

def main():

    parser = argparse.ArgumentParser(
        description="Fetch NOAA snow data for a single weather station by ID or name."
    )
    parser.add_argument(
        "--token", required=True,
        help="Your NOAA CDO API token"
    )
    parser.add_argument(
        "--start", default=str(date.today() - timedelta(days=100)),
        help="Start date YYYY-MM-DD (default: 100 days ago)"
    )
    parser.add_argument(
        "--end", default=str(date.today() - timedelta(days=1)),
        help="End date YYYY-MM-DD (default: yesterday)"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--id",
        help="Exact NOAA station ID (e.g. GHCND:USC00358534)"
    )
    group.add_argument(
        "--name",
        help="Case-insensitive substring of the station name (e.g. 'TIMBERLINE LODGE')"
    )
    args = parser.parse_args()

    print(f"\nNOAA — Snow Data")
    print(f"Date range : {args.start}  →  {args.end}")

    if args.id:
        print(f"Looking up station ID: {args.id}")
        station = getStationById(args.token, args.id)
        if not station:
            print(f"No station found with ID '{args.id}'. Exiting.")
            return
        stations = [station]
        print(f"Found: {station['name']} ({station['id']})\n")
    else:
        print(f"Searching for stations matching: '{args.name}'")
        matches = getStationsByName(args.token, args.name)
        if not matches:
            print(f"No station name contains '{args.name}'. Exiting.")
            return
        if len(matches) > 1:
            print(f"Multiple matches for '{args.name}':")
            for s in matches:
                print(f"  {s['name']} ({s['id']})")
            print("Using the first match.")
        stations = [matches[0]]
        print(f"Selected: {stations[0]['name']} ({stations[0]['id']})\n")
        
    #All station data is filled with only the data of a single station, future iterations of this program will include a multi station select, so the array has not been deleted
    for i, station in enumerate(stations, 1):
        print(f"[{i}/{len(stations)}] {station['name']} ({station['id']})")
        try:
            records = getStationSnowData(args.token, station["id"], args.start, args.end)
            stationDataFrame = createDataFrame(station, records)
        except requests.HTTPError as e:
            print(f"HTTP error for {station['id']}: {e}")
        sleep(0.25)
    #All data collected, this is redundant but will stay for future multi station alterations 
    if(stationDataFrame):
        snowDataFrame = pd.concat(stationDataFrame, ignore_index=True)
   
    print(f"\n\nDone. Processed {len(stations)} station(s). CSV created.")
    
    stationName = stations[0]['name'].replace(' ', '-')
    snowDataFrame.to_csv(f'snowData({stationName})({args.start}-{args.end}).csv', index=False)

if __name__ == "__main__":
    main()