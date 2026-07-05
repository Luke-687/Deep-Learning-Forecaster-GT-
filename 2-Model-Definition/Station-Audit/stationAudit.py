import requests
import argparse
import csv
import os
from datetime import datetime, date, timedelta
from time import sleep

BASE_URL   = "https://www.ncei.noaa.gov/cdo-web/api/v2"
DATASET_ID = "GHCND"

MIN_ELEVATION_M = 1000

# Snow-relevant datatypes and their weights in the composite score.
# Higher weight = more important for snow prediction.
DATATYPES = {
    "SNOW": ("Snowfall (in)",             3.0),
    "SNWD": ("Snow depth (in)",           3.0),
    "TMAX": ("Max temperature (°F)",      2.0),
    "TMIN": ("Min temperature (°F)",      2.0),
    "PRCP": ("Precipitation (in)",        1.5),
    "AWND": ("Average wind speed",        1.0),
    "WSF5": ("5-sec peak gust",           0.5),
    "RHMX": ("Max relative humidity",     0.5),
    "RHMN": ("Min relative humidity",     0.5),
    "WT18": ("Snow/ice flag",             0.5),
    "WT17": ("Freezing rain flag",        0.5),
}

US_STATES = [
    "FIPS:01","FIPS:02","FIPS:04","FIPS:05","FIPS:06","FIPS:08","FIPS:09",
    "FIPS:10","FIPS:12","FIPS:13","FIPS:15","FIPS:16","FIPS:17","FIPS:18",
    "FIPS:19","FIPS:20","FIPS:21","FIPS:22","FIPS:23","FIPS:24","FIPS:25",
    "FIPS:26","FIPS:27","FIPS:28","FIPS:29","FIPS:30","FIPS:31","FIPS:32",
    "FIPS:33","FIPS:34","FIPS:35","FIPS:36","FIPS:37","FIPS:38","FIPS:39",
    "FIPS:40","FIPS:41","FIPS:42","FIPS:44","FIPS:45","FIPS:46","FIPS:47",
    "FIPS:48","FIPS:49","FIPS:50","FIPS:51","FIPS:53","FIPS:54","FIPS:55",
    "FIPS:56",
]

CSV_COLUMNS = [
    "station_id", "name", "state", "elevation_ft", "latitude", "longitude",
    "data_start", "data_end", "sample_days",
    *[f"has_{dt}" for dt in DATATYPES],
    *[f"pct_{dt}" for dt in DATATYPES],
    "composite_score",
]


def loadScoredIds(csv_path: str) -> set[str]:
    """Read the existing CSV and return the set of station IDs already scored."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return {row["station_id"] for row in reader}


def appendRow(csv_path: str, row: dict, write_header: bool):
    """Append a single scored row to the CSV, writing the header only when needed."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def fetchStationsForState(token: str, state_id: str) -> list[dict]:
    headers = {"token": token}
    stations = []
    offset   = 1

    while True:
        params = {
            "datasetid":  DATASET_ID,
            "locationid": state_id,
            "limit":      1000,
            "offset":     offset,
        }
        resp = requests.get(f"{BASE_URL}/stations", headers=headers, params=params)
        resp.raise_for_status()
        data    = resp.json()
        results = data.get("results", [])
        if not results:
            break

        stations.extend(results)
        total = data.get("metadata", {}).get("resultset", {}).get("count", len(stations))
        if len(stations) >= total:
            break
        offset += len(results)
        sleep(0.5)

    return stations


def fetchSampleData(token: str, station_id: str, start: str, end: str) -> list[dict]:
    headers = {"token": token}
    records = []
    offset  = 1

    while True:
        params = {
            "datasetid":  DATASET_ID,
            "stationid":  station_id,
            "datatypeid": list(DATATYPES.keys()),
            "startdate":  start,
            "enddate":    end,
            "limit":      1000,
            "offset":     offset,
            "units":      "standard",
        }
        resp = requests.get(f"{BASE_URL}/data", headers=headers, params=params)
        if resp.status_code == 429:
            print("    Rate limited — waiting 30 s")
            sleep(30)
            continue
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
        sleep(1)

    return records


def scoreStation(station: dict, records: list[dict], sample_days: int) -> dict:
    counts: dict[str, int] = {dt: 0 for dt in DATATYPES}
    for rec in records:
        dt = rec.get("datatype")
        if dt in counts:
            counts[dt] += 1

    has = {dt: int(counts[dt] > 0)                         for dt in DATATYPES}
    pct = {dt: round(counts[dt] / max(sample_days, 1), 4) for dt in DATATYPES}

    composite = round(sum(pct[dt] * DATATYPES[dt][1] for dt in DATATYPES), 4)

    return {
        "station_id":      station["id"],
        "name":            station["name"],
        "state":           station.get("locationid", ""),
        "elevation_ft":    station.get("elevation", None),
        "latitude":        station.get("latitude",  None),
        "longitude":       station.get("longitude", None),
        "data_start":      station.get("mindate",   ""),
        "data_end":        station.get("maxdate",   ""),
        "sample_days":     sample_days,
        **{f"has_{dt}": has[dt] for dt in DATATYPES},
        **{f"pct_{dt}": pct[dt] for dt in DATATYPES},
        "composite_score": composite,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Audit NOAA GHCND stations nationwide for snow-prediction data quality."
    )
    parser.add_argument("--token",        required=True, help="NOAA CDO API token")
    parser.add_argument("--sample-start", default="2022-10-01",
                        help="Start of sample window (default: 2022-10-01)")
    parser.add_argument("--sample-end",   default="2023-03-31",
                        help="End of sample window (default: 2023-03-31)")
    parser.add_argument("--csv",          default="station_rankings.csv",
                        help="Output CSV path (default: station_rankings.csv)")
    args = parser.parse_args()

    d1 = datetime.strptime(args.sample_start, "%Y-%m-%d").date()
    d2 = datetime.strptime(args.sample_end,   "%Y-%m-%d").date()
    sample_days = (d2 - d1).days + 1

    # Load already-scored IDs from an existing CSV so runs can be resumed
    already_scored = loadScoredIds(args.csv)
    file_is_new    = len(already_scored) == 0

    print(f"Station Audit — NOAA GHCND")
    print(f"Sample window : {args.sample_start} → {args.sample_end} ({sample_days} days)")
    print(f"Min elevation : {MIN_ELEVATION_M} m")
    print(f"States        : {len(US_STATES)}")
    print(f"CSV           : {args.csv}  ({len(already_scored)} stations already saved)\n")

    total_scored            = 0
    total_skipped_elevation = 0
    total_skipped_cached    = 0
    header_written          = not file_is_new  # don't rewrite header if file already has rows

    for state_idx, state_id in enumerate(US_STATES, 1):
        print(f"[{state_idx}/{len(US_STATES)}] State {state_id}")

        try:
            stations = fetchStationsForState(args.token, state_id)
        except requests.HTTPError as e:
            print(f"  Error fetching stations: {e}")
            sleep(2)
            continue

        sleep(0.5)

        for station in stations:
            elev = station.get("elevation")

            if elev is None or elev < MIN_ELEVATION_M:
                total_skipped_elevation += 1
                continue

            sid = station["id"]

            if sid in already_scored:
                total_skipped_cached += 1
                continue

            print(f"  Scoring {sid} — {station['name']} ({elev} ft)")

            try:
                records = fetchSampleData(args.token, sid, args.sample_start, args.sample_end)
            except requests.HTTPError as e:
                print(f"    HTTP error: {e}")
                sleep(2)
                continue

            row = scoreStation(station, records, sample_days)

            appendRow(args.csv, row, write_header=not header_written)
            header_written = True
            already_scored.add(sid)
            total_scored += 1

            snow_present = "SNOW" if row["has_SNOW"] else "----"
            snwd_present = "SNWD" if row["has_SNWD"] else "----"
            print(f"    score={row['composite_score']:.3f}  {snow_present}  {snwd_present}  "
                  f"({len(records)} records)")

            sleep(1)

    print(f"\nAudit complete.")
    print(f"  Scored this run     : {total_scored}")
    print(f"  Skipped (elevation) : {total_skipped_elevation}")
    print(f"  Skipped (cached)    : {total_skipped_cached}")
    print(f"  Total in CSV        : {len(already_scored)}")
    print(f"  File                : {args.csv}")


if __name__ == "__main__":
    main()
