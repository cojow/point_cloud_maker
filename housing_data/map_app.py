"""
Streamlit app: pick a parcel CSV (produced by clean_parcel_data.py) and view
each house as a pinpoint on a map, using its latitude/longitude columns.

Usage:
    streamlit run map_app.py
"""

from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

DATA_DIR = Path(__file__).parent

st.set_page_config(page_title="Utah County Parcel Map", layout="wide")
st.title("Utah County Parcel Map")


@st.cache_data
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)


# --- File selection -----------------------------------------------------
# Only list outputs of clean_parcel_data.py (parcel_data_cleaned*.csv) --
# raw county exports and parcel_coordinates.csv (an internal lat/long
# lookup) aren't meant to be browsed here and lack the expected columns.
csv_files = sorted(DATA_DIR.glob("parcel_data_cleaned*.csv"))

with st.sidebar:
    st.header("Data")
    source = st.radio("CSV source", ["Choose from folder", "Upload a file"])

    if source == "Choose from folder":
        if not csv_files:
            st.error(f"No CSV files found in {DATA_DIR}")
            st.stop()
        choice = st.selectbox("File", csv_files, format_func=lambda p: p.name)
        df = load_csv(choice)
    else:
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded is None:
            st.info("Upload a CSV to continue.")
            st.stop()
        df = load_csv(uploaded)

# --- Column mapping -------------------------------------------------------
cols = list(df.columns)


def guess(names, default_idx=0):
    for n in names:
        if n in cols:
            return cols.index(n)
    return default_idx


with st.sidebar:
    st.header("Columns")
    lat_col = st.selectbox("Latitude column", cols, index=guess(["latitude", "lat"]))
    lon_col = st.selectbox("Longitude column", cols, index=guess(["longitude", "lon", "long"]))

df = df.dropna(subset=[lat_col, lon_col]).copy()
df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
df = df.dropna(subset=[lat_col, lon_col])

if df.empty:
    st.error("No rows with valid coordinates in this file.")
    st.stop()

# --- Filters ----------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    if "city" in cols:
        cities = sorted(df["city"].dropna().unique())
        picked_cities = st.multiselect("City", cities)
        if picked_cities:
            df = df[df["city"].isin(picked_cities)]

    if "total_sqft" in cols:
        sqft = pd.to_numeric(df["total_sqft"], errors="coerce")
        valid = sqft[sqft > 0]
        if not valid.empty:
            lo, hi = int(valid.min()), int(valid.max())
            if lo < hi:
                sel = st.slider("Total sq ft", lo, hi, (lo, hi))
                df = df[(sqft >= sel[0]) & (sqft <= sel[1])]

    search = st.text_input("Search address / owner contains")
    if search:
        mask = pd.Series(False, index=df.index)
        for c in ("street_address", "owner_name"):
            if c in cols:
                mask |= df[c].astype(str).str.contains(search, case=False, na=False)
        df = df[mask]

st.caption(f"{len(df):,} houses plotted")

# --- Tooltip fields -----------------------------------------------------
tooltip_candidates = [
    "street_address", "city", "bedrooms", "bathrooms", "total_sqft",
    "above_grade_sqft", "basement_sqft", "year_built",
    "market_value_current", "owner_name",
]
tooltip_fields = [c for c in tooltip_candidates if c in cols]

tooltip_html = "<br/>".join(f"<b>{c}:</b> {{{c}}}" for c in tooltip_fields) or "Lat: {%s}<br/>Lon: {%s}" % (lat_col, lon_col)

# Only ship the columns the map actually needs -- sending the full
# dataframe (all 26 columns) to the browser blows past pydeck's message
# size limit once there are more than ~50k rows.
map_cols = [lat_col, lon_col] + [c for c in tooltip_fields if c not in (lat_col, lon_col)]
map_df = df[map_cols]

# --- Map ------------------------------------------------------------
view_state = pdk.ViewState(
    latitude=float(df[lat_col].mean()),
    longitude=float(df[lon_col].mean()),
    zoom=10,
    pitch=0,
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position=f"[{lon_col}, {lat_col}]",
    get_radius=15,
    get_fill_color=[255, 90, 0, 160],
    pickable=True,
    radius_min_pixels=2,
    radius_max_pixels=8,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style="light",
    tooltip={"html": tooltip_html, "style": {"backgroundColor": "steelblue", "color": "white"}},
)

st.pydeck_chart(deck, use_container_width=True, height=650)

with st.expander("Show data table"):
    st.dataframe(df, use_container_width=True)
