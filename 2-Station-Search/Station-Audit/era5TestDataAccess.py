"""
ERA5 atmospheric column data fetcher for Mt. Washington, NH.

Mt. Washington peak : 44.2705 N, 71.3033 W  (44°16'13.8"N  71°18'11.7"W)
ERA5 resolution     : 0.25° × 0.25°
Target column       : 44.25 N, 71.25 W  (covers summit + Tuckerman Ravine)

Requires a CDS API key. On first use, create ~/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: <YOUR-API-KEY>
Obtain credentials at: https://cds.climate.copernicus.eu/user/register
Accept dataset licences before first use:
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
"""

import subprocess
import sys
import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path


# ── Grid / region constants ────────────────────────────────────────────────────

PEAK_LAT   =  44.2705
PEAK_LON   = -71.3033

# ERA5 0.25° grid point nearest to the peak (covers summit + Tuckerman Ravine)
BEST_LAT   =  44.25
BEST_LON   = -71.25

# Single-point area: [North, West, South, East] — all four values identical
# forces CDS to return exactly one grid node rather than a regional subset
AREA = [44.25, -71.25, 44.25, -71.25]

# ── Variable definitions ───────────────────────────────────────────────────────
PRESSURE_LEVEL_VARS = {
    "temperature":        ["T_850", "T_700", "T_500"],
    "relative_humidity":  ["RH_700"],
    "specific_humidity":  ["Q_700"],
    "geopotential":       ["Z_500"],
    "vertical_velocity":  ["W_700"],
    "u_component_of_wind": ["U_850"],
    "v_component_of_wind": ["V_850"],
}

PRESSURE_LEVELS = ["500", "700", "850"]

SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_column_water_vapour",
]

HOURS = [f"{h:02d}:00" for h in range(24)]  # all 24 hours — full hourly resolution


# ── Helpers ────────────────────────────────────────────────────────────────────
def ensureCdsapi():
    try:
        import cdsapi
        return cdsapi
    except ImportError:
        print("cdsapi not found — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cdsapi"])
        import cdsapi
        return cdsapi


def dateRange(start: str, end: str):
    """Yield (year, month, days_list) tuples spanning the requested range."""
    d1 = datetime.strptime(start, "%Y-%m-%d").date()
    d2 = datetime.strptime(end,   "%Y-%m-%d").date()

    from collections import defaultdict
    by_ym = defaultdict(list)
    current = d1
    while current <= d2:
        by_ym[(str(current.year), f"{current.month:02d}")].append(f"{current.day:02d}")
        current += timedelta(days=1)

    for (year, month), days in sorted(by_ym.items()):
        yield year, month, days


def buildRequest(year: str, month: str, days: list[str], area: list) -> dict:
    return {
        "product_type": "reanalysis",
        "year":   year,
        "month":  month,
        "day":    days,
        "time":   HOURS,
        "area":   area,
        "format": "netcdf",
    }


def fetchPressureLevels(client, year: str, month: str, days: list[str], out_path: str):
    req = buildRequest(year, month, days, AREA)
    req["variable"]       = list(PRESSURE_LEVEL_VARS.keys())
    req["pressure_level"] = PRESSURE_LEVELS
    client.retrieve("reanalysis-era5-pressure-levels", req, out_path)


def fetchSingleLevels(client, year: str, month: str, days: list[str], out_path: str):
    req = buildRequest(year, month, days, AREA)
    req["variable"] = SINGLE_LEVEL_VARS
    client.retrieve("reanalysis-era5-single-levels", req, out_path)


def printSummary(output_dir: Path):
    files = sorted(output_dir.glob("*.nc"))
    if not files:
        print("No output files found.")
        return
    total = sum(f.stat().st_size for f in files)
    print(f"\n{'─'*60}")
    print(f"  Output directory : {output_dir}")
    print(f"  Files written    : {len(files)}")
    for f in files:
        print(f"    {f.name:50s}  {f.stat().st_size / 1e6:6.2f} MB")
    print(f"  Total size       : {total / 1e6:.2f} MB")
    print(f"{'─'*60}")
    print(f"\n  ERA5 column : {BEST_LAT}°N, {abs(BEST_LON)}°W")
    print(f"  Covers      : Mt. Washington summit + Tuckerman Ravine")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch ERA5 atmospheric column data for the Mt. Washington region.\n"
            "Pressure-level and single-level variables are saved as separate\n"
            "NetCDF files, one per calendar month in the requested date range.\n\n"
            "Prerequisite: create ~/.cdsapirc with your Copernicus CDS credentials.\n"
            "Register at: https://cds.climate.copernicus.eu/user/register"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start", required=True,
        help="Start date  YYYY-MM-DD  (e.g. 2020-10-01)"
    )
    parser.add_argument(
        "--end", required=True,
        help="End date    YYYY-MM-DD  (e.g. 2021-03-31)"
    )
    parser.add_argument(
        "--output", default="era5_mtwashington",
        help="Output directory for NetCDF files (default: era5_mtwashington)"
    )
    args = parser.parse_args()

    # Validate dates
    try:
        d1 = datetime.strptime(args.start, "%Y-%m-%d")
        d2 = datetime.strptime(args.end,   "%Y-%m-%d")
    except ValueError as e:
        parser.error(f"Invalid date format: {e}")
    if d2 < d1:
        parser.error("--end must be on or after --start")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cdsapi = ensureCdsapi()
    client = cdsapi.Client()

    print(f"\nERA5 — Mt. Washington Atmospheric Column")
    print(f"Peak         : {PEAK_LAT}°N, {abs(PEAK_LON)}°W")
    print(f"Best column  : {BEST_LAT}°N, {abs(BEST_LON)}°W  (nearest 0.25° grid point)")
    print(f"Bounding box : {AREA[0]}–{AREA[2]}°N  /  {abs(AREA[1])}–{abs(AREA[3])}°W")
    print(f"Date range   : {args.start} → {args.end}")
    print(f"Output dir   : {output_dir}\n")

    months = list(dateRange(args.start, args.end))
    total  = len(months)

    for i, (year, month, days) in enumerate(months, 1):
        tag = f"{year}-{month}"
        print(f"[{i}/{total}]  {tag}")

        pl_path = output_dir / f"pressure_levels_{tag}.nc"
        sl_path = output_dir / f"single_levels_{tag}.nc"

        if not pl_path.exists():
            print(f"  → Fetching pressure-level variables...")
            fetchPressureLevels(client, year, month, days, str(pl_path))
        else:
            print(f"  → Pressure-level file already exists, skipping.")

        if not sl_path.exists():
            print(f"  → Fetching single-level variables...")
            fetchSingleLevels(client, year, month, days, str(sl_path))
        else:
            print(f"  → Single-level file already exists, skipping.")

    printSummary(output_dir)


if __name__ == "__main__":
    main()
