# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Page config ----------
st.set_page_config(page_title="Air Quality Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #000000; font-weight: bold; }
    .kpi {background-color: #f8fafb; padding: 12px; border-radius:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);}
    .small {font-size:0.9rem; color:#000000;}
    </style>
""", unsafe_allow_html=True)

st.title("🌍 Air Quality Dashboard")

# ---------- Load data ----------
from pathlib import Path

# Use a path relative to this app file; also search workspace for fallback
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Air_quality_cleaned.csv"

@st.cache_data
def load_data(path=DATA_PATH):
    p = Path(path)
    # If default location is missing, try repository root and then a workspace-wide search
    if not p.exists():
        repo_root = Path("/workspaces/streamlit")
        repo_candidate = repo_root / "Air_quality_cleaned.csv"
        if repo_candidate.exists():
            p = repo_candidate
        else:
            matches = list(repo_root.rglob("Air_quality_cleaned*.csv"))
            if matches:
                p = matches[0]
            else:
                st.error(
                    f"CSV file not found. Tried: {path}, {repo_candidate}.\n"
                    "Place 'Air_quality_cleaned.csv' in the same folder as this app or update DATA_PATH."
                )
                st.stop()
    df = pd.read_csv(p)
    # Ensure lat/lng are numeric
    for c in ["lat","lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()

if df.shape[0] == 0:
    st.error("Dataframe is empty. Check your CSV file.")
    st.stop()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

country_col = "Country" if "Country" in df.columns else None
city_col = "City" if "City" in df.columns else None

# Country selection
if country_col:
    countries = sorted(df[country_col].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Select Country", ["All"] + countries, index=0)
else:
    selected_country = "All"

# City selection
if city_col:
    if selected_country != "All" and country_col:
        cities = sorted(df[df[country_col] == selected_country][city_col].dropna().unique().tolist())
        cities = ["All"] + cities
    else:
        cities = ["All"] + sorted(df[city_col].dropna().unique().tolist())
    selected_city = st.sidebar.selectbox("Select City", cities, index=0)
else:
    selected_city = "All"

# Pollutant selection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
pollutant_cols = [c for c in numeric_cols if c not in ("lat","lng")]
if not pollutant_cols:
    st.error("No numeric pollutant columns found.")
    st.stop()

selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutant_cols, index=0)

# AQI threshold
aqi_threshold = st.sidebar.number_input("AQI red threshold (unhealthy >)", min_value=0, max_value=1000, value=150, step=10)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by Clement — Clean dataset used: **Air_quality_cleaned.csv**")

# ---------- Apply filters ----------
filtered = df.copy()
if selected_country != "All" and country_col:
    filtered = filtered[filtered[country_col] == selected_country]
if selected_city != "All" and city_col:
    filtered = filtered[filtered[city_col] == selected_city]

# ---------- KPIs ----------
kpi1 = filtered[selected_pollutant].mean()
kpi2 = filtered[selected_pollutant].max()
kpi3 = filtered.shape[0]

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(f"<div class='kpi'><h3 style='margin:2px'>{kpi1:.1f}</h3><p class='small'>Average {selected_pollutant}</p></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><h3 style='margin:2px'>{kpi2:.1f}</h3><p class='small'>Max {selected_pollutant}</p></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><h3 style='margin:2px'>{k3}</h3><p class='small'>Records</p></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- Bar chart: top cities ----------
st.subheader("Top 20 Cities by Average Pollutant")
if city_col and selected_city == "All":
    avg_city = filtered.groupby(city_col)[selected_pollutant].mean().reset_index().sort_values(by=selected_pollutant, ascending=False).head(20)
    fig_bar = px.bar(avg_city, x=city_col, y=selected_pollutant, color=selected_pollutant, color_continuous_scale="Viridis")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown(f"**Explanation:** Bar chart shows top 20 cities by average {selected_pollutant}.")
else:
    st.info("Select 'All' cities to see this chart.")

# ---------- Pie chart: unhealthy vs healthy ----------
st.subheader("Share of Unhealthy vs Healthy/Moderate Readings")
unhealthy_count = filtered[filtered[selected_pollutant] > aqi_threshold].shape[0]
healthy_count = filtered.shape[0] - unhealthy_count
pie_df = pd.DataFrame({"Status":["Unhealthy","Healthy/Moderate"], "Count":[unhealthy_count, healthy_count]})
fig_pie = px.pie(pie_df, names="Status", values="Count", hole=0.4,
                 color="Status", color_discrete_map={"Unhealthy":"red","Healthy/Moderate":"green"})
st.plotly_chart(fig_pie, use_container_width=True)
st.markdown(f"**Explanation:** Pie chart shows proportion of readings above AQI threshold ({aqi_threshold}) vs safe readings.")

# ---------- Scatter plot ----------
st.subheader(f"{selected_pollutant} vs AQI")
fig_scatter = px.scatter(filtered, x=selected_pollutant, y=selected_pollutant,
                         color=filtered[selected_pollutant] > aqi_threshold,
                         color_discrete_map={True:"red", False:"green"},
                         labels={selected_pollutant:selected_pollutant, selected_pollutant:"AQI"})
st.plotly_chart(fig_scatter, use_container_width=True)
st.markdown(f"**Explanation:** Scatter shows relationship of {selected_pollutant} and AQI threshold. Red points are above threshold.")

# ---------- Correlation heatmap ----------
st.subheader("Correlation Heatmap")
num_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) >= 2:
    corr = filtered[num_cols].corr()
    fig_heat, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig_heat)
    st.markdown("**Explanation:** Heatmap shows correlation between numeric variables.")
else:
    st.info("Not enough numeric columns for heatmap.")

# ---------- Map ----------
st.subheader("Air Quality Map")
if "lat" in filtered.columns and "lng" in filtered.columns:
    map_df = filtered.dropna(subset=["lat","lng"])
    fig_map = px.scatter_mapbox(
        map_df,
        lat="lat", lon="lng",
        hover_name=city_col if city_col else None,
        hover_data=[selected_pollutant],
        color=map_df[selected_pollutant] > aqi_threshold,
        color_discrete_map={True:"red", False:"green"},
        zoom=2,
        height=500
    )
    fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown(f"**Explanation:** Map shows city locations. Points are **green** if AQI ≤ {aqi_threshold}, **red** if AQI > {aqi_threshold}.")
else:
    st.info("No latitude/longitude columns found for mapping.")

# ---------- Insights ----------
st.subheader("Insights & Summary")
if city_col:
    top_cities = filtered[[city_col, selected_pollutant]].sort_values(by=selected_pollutant, ascending=False).head(10)
    bottom_cities = filtered[[city_col, selected_pollutant]].sort_values(by=selected_pollutant, ascending=True).head(10)
    st.markdown("**Top 10 most polluted cities:**")
    st.dataframe(top_cities.reset_index(drop=True))
    st.markdown("**Top 10 least polluted cities:**")
    st.dataframe(bottom_cities.reset_index(drop=True))

percent_unhealthy = (unhealthy_count / filtered.shape[0] * 100) if filtered.shape[0] > 0 else 0
st.markdown(f"**Unhealthy readings:** {unhealthy_count} of {filtered.shape[0]} records ({percent_unhealthy:.1f}%)")
