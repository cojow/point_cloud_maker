"""
Reads Utah County's raw TaxParcel export (TaxParcel_July21_2026_all.csv) and
writes a cleaned, renamed CSV with address/location, bedroom/bathroom counts,
and building square footage.

This export has no coordinates of its own (only SHAPE_Length/SHAPE_Area,
which are polygon perimeter/area, not a point). Latitude/longitude are
instead joined in from parcel_coordinates.csv (a parcel_id -> lat/long
lookup built from an earlier Utah County export that did include point
geometry) by PARCELID.

Usage:
    python3 clean_parcel_data.py [--city CITY] [input.csv] [output.csv]

    python3 clean_parcel_data.py --city Provo
"""

import argparse
import csv
from pathlib import Path

DEFAULT_INPUT = "TaxParcel_July21_2026_all.csv"
DEFAULT_COORDS = "parcel_coordinates.csv"

# Output column order: identifiers -> location -> rooms/size -> financial -> misc
OUTPUT_FIELDS = [
    "parcel_id",
    "parcel_id_formatted",
    "owner_name",
    "street_address",
    "house_number",
    "street_direction",
    "street_name",
    "street_type",
    "street_post_direction",
    "unit_number",
    "city",
    "zip_code",
    "latitude",
    "longitude",
    "acreage",
    "bedrooms",
    "bathrooms",
    "above_grade_sqft",
    "basement_sqft",
    "basement_finished_sqft",
    "garage_sqft",
    "total_sqft",
    "year_built",
    "house_style",
    "quality",
    "condition",
    "unit_count",
    "improvement_count",
    "market_value_current",
    "market_value_previous",
    "current_year_taxes",
    "previous_year_taxes",
    "greenbelt_status",
    "tax_district",
    "tax_district_description",
    "subdivision_name",
    "parcel_create_date",
]


def clean(value):
    if value is None:
        return None
    value = value.strip()
    return value or None


def to_float(value):
    value = clean(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value):
    num = to_float(value)
    return int(num) if num is not None else None


def load_coordinates(coords_path: Path):
    lookup = {}
    with coords_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["parcel_id"]] = (row["latitude"], row["longitude"])
    return lookup


def convert(input_path: Path, output_path: Path, coords_path: Path, city: str = None):
    coords = load_coordinates(coords_path)
    city_filter = city.strip().upper() if city else None

    seen_parcels = set()
    rows_written = 0
    rows_read = 0
    rows_missing_coords = 0

    with input_path.open(encoding="utf-8-sig") as fin, \
         output_path.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for row in reader:
            rows_read += 1
            parcel_id = clean(row["PARCELID"])

            # A handful of parcels have multiple rows (multi-part shapes).
            # Attributes are duplicated across those rows, so keep the first.
            if parcel_id in seen_parcels:
                continue
            seen_parcels.add(parcel_id)

            if city_filter is not None:
                row_city = clean(row["SITE_CITY"])
                if not row_city or row_city.upper() != city_filter:
                    continue

            lat, lon = coords.get(parcel_id, (None, None))
            if lat is None:
                rows_missing_coords += 1

            above_grade = to_float(row["GLA_RES"]) or 0
            basement = to_float(row["BASEMENT_RES"]) or 0

            writer.writerow({
                "parcel_id": parcel_id,
                "parcel_id_formatted": clean(row["PARCELID_LABEL"]),
                "owner_name": clean(row["OWNER_NAME"]),
                "street_address": clean(row["SITE_FULL_ADDRESS"]),
                "house_number": clean(row["SITE_HOUSE_NUM"]),
                "street_direction": clean(row["SITE_PRE_DIR"]),
                "street_name": clean(row["SITE_STREET_NAME"]),
                "street_type": clean(row["SITE_STREET_TYPE"]),
                "street_post_direction": clean(row["SITE_POST_DIR"]),
                "unit_number": clean(row["SITE_STREET_UNIT"]),
                "city": clean(row["SITE_CITY"]),
                "zip_code": clean(row["SITE_ZIP5"]),
                "latitude": lat,
                "longitude": lon,
                "acreage": to_float(row["ACREAGE"]),
                "bedrooms": to_int(row["GLA_BEDROOMS_RES"]),
                "bathrooms": to_float(row["BATHROOMS_RES"]),
                "above_grade_sqft": above_grade,
                "basement_sqft": basement,
                "basement_finished_sqft": to_float(row["BSMT_FINISH_RES"]),
                "garage_sqft": to_float(row["ATT_GARAGE_SQFT_RES"]),
                "total_sqft": above_grade + basement,
                "year_built": to_int(row["YEARBLT_RES"]),
                "house_style": clean(row["STYLE_DESCR_RES"]),
                "quality": clean(row["QUALITY_DESCR_RES"]),
                "condition": clean(row["CONDITION_DESCR_RES"]),
                "unit_count": to_int(row["TOTAL_UNIT_COUNT"]),
                "improvement_count": to_int(row["TOTAL_IMP_COUNT"]),
                "market_value_current": to_float(row["MKT_CUR_VALUE"]),
                "market_value_previous": to_float(row["MKT_PRV_VAL"]),
                "current_year_taxes": to_float(row["TOT_CUR_TAXES"]),
                "previous_year_taxes": to_float(row["TOT_PRV_TAXES"]),
                "greenbelt_status": clean(row["GREENBELT"]),
                "tax_district": clean(row["TAX_DISTRICT"]),
                "tax_district_description": clean(row["TAX_DISTRICT_DESCR"]),
                "subdivision_name": clean(row["SUB_NAME"]),
                "parcel_create_date": clean(row["PARCEL_CREATE_DATE"]),
            })
            rows_written += 1

    print(f"Read {rows_read} rows, wrote {rows_written} unique parcels to {output_path}")
    if rows_missing_coords:
        pct = rows_missing_coords / rows_written * 100
        print(f"Warning: {rows_missing_coords} parcels ({pct:.1f}%) had no matching coordinates in {coords_path.name}")


def default_output_name(city: str = None) -> str:
    if not city:
        return "parcel_data_cleaned.csv"
    slug = city.strip().lower().replace(" ", "_")
    return f"parcel_data_cleaned_{slug}.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=None, help="Path to TaxParcel CSV export")
    parser.add_argument("output", nargs="?", default=None, help="Path to output CSV")
    parser.add_argument("--city", default=None, help="Only include parcels in this city (matches SITE_CITY, case-insensitive)")
    parser.add_argument("--coords-file", default=None, help="Path to parcel_id -> lat/long lookup CSV")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else Path(__file__).parent / DEFAULT_INPUT
    output_path = Path(args.output) if args.output else Path(__file__).parent / default_output_name(args.city)
    coords_path = Path(args.coords_file) if args.coords_file else Path(__file__).parent / DEFAULT_COORDS
    convert(input_path, output_path, coords_path, city=args.city)
