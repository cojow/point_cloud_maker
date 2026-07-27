"""
Streamlit app: pick a parcel CSV (produced by clean_parcel_data.py) and view
each house as a pinpoint on a map, using its latitude/longitude columns.

Usage:
    streamlit run map_app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
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

st.caption(f"{len(df):,} houses match filters")

map_tab, compare_tab = st.tabs(["Map", "Compare columns"])

# --- Map tab --------------------------------------------------------------
with map_tab:
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

# --- Compare tab ------------------------------------------------------
# Chart types fall into three shapes:
#   "xy"      needs two numeric columns (x and y)
#   "grouped" summarizes one numeric column, optionally grouped by a category
#   "single"  summarizes the distribution of one column on its own
CHART_TYPES = {
    "Scatter": "xy",
    "Line": "xy",
    "Density heatmap": "xy",
    "Box": "grouped",
    "Violin": "grouped",
    "Histogram": "single",
}

with compare_tab:
    st.subheader("Plot one column against another")

    chart_type = st.selectbox("Chart type", list(CHART_TYPES), key="compare_chart_type")
    shape = CHART_TYPES[chart_type]

    if shape == "xy":
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("X axis", cols, index=guess(["total_sqft"]), key="compare_x")
        with c2:
            y_col = st.selectbox(
                "Y axis", cols, index=guess(["market_value_current"], default_idx=min(1, len(cols) - 1)), key="compare_y"
            )
        numeric_cols = [x_col, y_col]
    elif shape == "grouped":
        c1, c2 = st.columns(2)
        with c1:
            y_col = st.selectbox(
                "Value column (numeric)", cols,
                index=guess(["market_value_current"], default_idx=min(1, len(cols) - 1)),
                key="compare_y",
            )
        with c2:
            x_choice = st.selectbox("Group by (optional)", ["None"] + cols, key="compare_x")
        x_col = None if x_choice == "None" else x_choice
        numeric_cols = [y_col]
    else:  # single
        y_col = st.selectbox("Column", cols, index=guess(["total_sqft"]), key="compare_y")
        x_col = None
        numeric_cols = [y_col]

    supports_color = chart_type != "Density heatmap"
    color_col = None
    if supports_color:
        color_choice = st.selectbox("Color by (optional)", ["None"] + cols, key="compare_color")
        color_col = None if color_choice == "None" else color_choice

    bins = None
    if chart_type == "Histogram":
        bins = st.slider("Bins", 10, 200, 50, key="compare_bins")

    trendline_choice = "None"
    ref_lines = []
    if shape == "xy":
        trendline_choice = st.selectbox(
            "Trendline", ["None", "Linear (OLS)", "LOWESS (smoothed)"], key="compare_trendline"
        )
    else:
        ref_lines = st.multiselect("Reference lines", ["Mean", "Median"], key="compare_reflines")

    plot_cols = list(dict.fromkeys([c for c in (x_col, y_col, color_col) if c]))
    hover_cols = [c for c in ("street_address", "city") if c in cols and c not in plot_cols]

    plot_df = df[plot_cols + hover_cols].copy()
    for c in numeric_cols:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

    required_cols = [c for c in (x_col, y_col) if c]
    plot_df = plot_df.dropna(subset=required_cols)

    if plot_df.empty:
        st.warning("No rows have valid values for the selected column(s).")
    else:
        if chart_type == "Scatter":
            fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, hover_data=hover_cols or None, opacity=0.6)
        elif chart_type == "Line":
            fig = px.line(plot_df.sort_values(x_col), x=x_col, y=y_col, color=color_col, hover_data=hover_cols or None)
        elif chart_type == "Density heatmap":
            fig = px.density_heatmap(plot_df, x=x_col, y=y_col)
        elif chart_type == "Histogram":
            fig = px.histogram(plot_df, x=y_col, color=color_col, nbins=bins)
        elif chart_type == "Box":
            fig = px.box(plot_df, x=x_col, y=y_col, color=color_col, hover_data=hover_cols or None)
        elif chart_type == "Violin":
            fig = px.violin(plot_df, x=x_col, y=y_col, color=color_col, box=True, hover_data=hover_cols or None)

        if trendline_choice != "None":
            # Statsmodels trendlines (especially LOWESS) get very slow on 100k+ rows,
            # so fit on a capped random sample and draw that over the full plot.
            TREND_MAX_ROWS = 5000
            trend_key = {"Linear (OLS)": "ols", "LOWESS (smoothed)": "lowess"}[trendline_choice]
            trend_source = plot_df
            if len(trend_source) > TREND_MAX_ROWS:
                trend_source = trend_source.sample(n=TREND_MAX_ROWS, random_state=0)
                st.caption(f"Trendline fit on a random sample of {TREND_MAX_ROWS:,} rows for performance.")
            trend_fig = px.scatter(trend_source, x=x_col, y=y_col, color=color_col, trendline=trend_key)
            for trace in trend_fig.data:
                if "trendline" in (trace.hovertemplate or "").lower():
                    fig.add_trace(trace)

        ref_line_specs = [
            ("Mean", "dash", "black", "top"),
            ("Median", "dot", "gray", "bottom"),
        ]
        for label, dash, color, side in ref_line_specs:
            if label in ref_lines:
                value = float(plot_df[y_col].mean() if label == "Mean" else plot_df[y_col].median())
                if shape == "single":
                    fig.add_vline(
                        x=value, line_dash=dash, line_color=color,
                        annotation_text=f"{label}: {value:,.0f}", annotation_position=f"{side} right",
                    )
                else:
                    fig.add_hline(
                        y=value, line_dash=dash, line_color=color,
                        annotation_text=f"{label}: {value:,.0f}", annotation_position=f"{side} right",
                    )

        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(plot_df):,} rows used")
