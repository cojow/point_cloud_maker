"""
Match drone-derived building measurements (measurements.csv) to their
corresponding Utah County parcel record (parcel_data_cleaned_provo.csv) by
nearest latitude/longitude, so drone-measured stats (e.g. footprint area)
can be compared against the assessor's records for the same house.

Each measurement row is matched to its single nearest parcel. A
match_distance_ft column is included so bad matches (drone point far from
any parcel centroid) can be spotted and filtered out.

Usage:
    python3 match_measurements_to_parcels.py [measurements.csv] [parcels.csv] [output.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_FT = 20_925_646.3

DEFAULT_MEASUREMENTS = Path("../data/fir/analysis_fir_v7b/measurements.csv")
DEFAULT_PARCELS = Path("parcel_data_cleaned_provo.csv")
DEFAULT_OUTPUT = Path("../data/fir/analysis_fir_v7b/measurements_with_parcels.csv")


def haversine_ft(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_FT * np.arcsin(np.sqrt(a))


def main():
    args = sys.argv[1:]
    measurements_path = Path(args[0]) if len(args) > 0 else DEFAULT_MEASUREMENTS
    parcels_path = Path(args[1]) if len(args) > 1 else DEFAULT_PARCELS
    output_path = Path(args[2]) if len(args) > 2 else DEFAULT_OUTPUT

    measurements = pd.read_csv(measurements_path)
    parcels = pd.read_csv(parcels_path).dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    parcel_lat = parcels["latitude"].to_numpy()
    parcel_lon = parcels["longitude"].to_numpy()

    best_idx = np.empty(len(measurements), dtype=int)
    best_dist = np.empty(len(measurements))
    for i, (lat, lon) in enumerate(zip(measurements["Latitude"], measurements["Longitude"])):
        dists = haversine_ft(lat, lon, parcel_lat, parcel_lon)
        j = int(np.argmin(dists))
        best_idx[i] = j
        best_dist[i] = dists[j]

    matched_parcels = parcels.iloc[best_idx].reset_index(drop=True).add_prefix("parcel_")
    matched_parcels["match_distance_ft"] = best_dist

    combined = pd.concat([measurements.reset_index(drop=True), matched_parcels], axis=1)
    combined.to_csv(output_path, index=False)

    dup_parcels = matched_parcels["parcel_parcel_id"][matched_parcels["parcel_parcel_id"].duplicated(keep=False)]

    print(f"Wrote {len(combined)} rows to {output_path}")
    print(f"Match distance (ft): min={best_dist.min():.1f} median={np.median(best_dist):.1f} max={best_dist.max():.1f}")
    far = combined[best_dist > 100][["house_ID", "match_distance_ft"]]
    if not far.empty:
        print(f"\n{len(far)} row(s) matched a parcel more than 100 ft away -- verify these:")
        print(far.to_string(index=False))
    if not dup_parcels.empty:
        print(f"\n{dup_parcels.nunique()} parcel(s) matched by more than one measurement row.")


if __name__ == "__main__":
    main()
