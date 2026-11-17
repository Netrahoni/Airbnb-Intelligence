from __future__ import annotations

from datetime import date
from pathlib import Path
import html

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

st.set_page_config(
    page_title="Kathmandu Stay Intelligence",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
      --bg: #0B1220; --panel: #111C2E; --panel2: #16243A;
      --line: #26364D; --muted: #91A4BC; --text: #E8F0FA;
      --blue: #38BDF8; --teal: #2DD4BF; --amber: #FBBF24;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(circle at 80% 0%, #14243A 0, #0B1220 42%); color: var(--text); }
    [data-testid="stHeader"] { background: rgba(11,18,32,.82); }
    [data-testid="stSidebar"] { background: #0A1424; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
    h1,h2,h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; color: #F4F8FD; }
    h1 { font-size: 2.25rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.45rem !important; font-weight: 750 !important; }
    h3 { font-size: 1.08rem !important; font-weight: 700 !important; }
    p, label, span, div { }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMetric"] { background: linear-gradient(145deg,#14243A,#101B2D); border: 1px solid var(--line); padding: 17px 19px; border-radius: 16px; min-height: 112px; }
    [data-testid="stMetricLabel"] { color: #A9BCD2; font-size: .82rem; }
    [data-testid="stMetricValue"] { color: #F5FAFF; font-family: 'Manrope',sans-serif; font-size: 1.7rem; }
    [data-testid="stMetricDelta"] { font-size: .78rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line) !important; border-radius: 16px; }
    div[data-testid="stForm"] { background: #111C2E; border: 1px solid var(--line); padding: 22px; border-radius: 18px; }
    .stButton > button, .stFormSubmitButton > button { border: 0; border-radius: 10px; background: linear-gradient(100deg,#0284C7,#0D9488); color: white; font-weight: 700; padding: .62rem 1.1rem; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { border: 0; filter: brightness(1.12); color: white; }
    .stTextInput input, .stNumberInput input, .stDateInput input, [data-baseweb="select"] > div { background: #142238 !important; border-color: #30445E !important; border-radius: 10px !important; color: #EDF6FF !important; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    [data-testid="stTabs"] button { color: #A9BCD2; font-weight: 600; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--blue); }
    hr { border-color: var(--line) !important; }
    .eyebrow { color: var(--teal); font-size: .72rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .4rem; }
    .hero { background: linear-gradient(115deg,rgba(18,43,68,.98),rgba(15,35,55,.76)); border: 1px solid #29445F; border-radius: 22px; padding: 28px 30px; margin: 5px 0 22px; }
    .hero-title { font-family: 'Manrope',sans-serif; color: #F5FAFF; font-size: 2rem; font-weight: 800; letter-spacing: -.04em; line-height: 1.15; }
    .hero-copy { color: #B7C8DA; font-size: .98rem; margin-top: 10px; max-width: 760px; }
    .pill { display: inline-block; background: rgba(45,212,191,.12); border: 1px solid rgba(45,212,191,.3); color: #69E6D5; border-radius: 999px; padding: 5px 10px; font-size: .72rem; font-weight: 700; margin-bottom: 12px; }
    .notice { background: rgba(251,191,36,.08); border: 1px solid rgba(251,191,36,.28); color: #F5D88B; border-radius: 12px; padding: 12px 15px; font-size: .84rem; }
    .info-panel { background: #111C2E; border: 1px solid var(--line); border-radius: 15px; padding: 17px 19px; }
    .info-label { color: #91A4BC; text-transform: uppercase; letter-spacing: .1em; font-size: .68rem; font-weight: 800; }
    .info-value { color: #F2F7FD; font-size: 1.05rem; font-weight: 700; margin-top: 5px; }
    .property-card { background: linear-gradient(150deg,#14243A,#101A2B); border: 1px solid #2A3D56; border-radius: 16px; padding: 17px; height: 100%; }
    .property-name { color: #F4F8FD; font-family: 'Manrope',sans-serif; font-size: 1rem; font-weight: 800; }
    .property-meta { color: #91A4BC; font-size: .78rem; margin: 5px 0 12px; }
    .property-price { color: #67E8D4; font-size: 1.16rem; font-weight: 800; margin-top: 12px; }
    .score { color: #7DD3FC; font-size: .78rem; font-weight: 700; }
    .sidebar-brand { font-family: 'Manrope',sans-serif; font-weight: 800; font-size: 1.12rem; color: #F5FAFF; }
    .sidebar-sub { color: #8FA5BE; font-size: .78rem; margin-top: 4px; }
    .section-kicker { color: #38BDF8; font-size: .73rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# DATA + MODEL LOADING
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    rec = joblib.load(MODEL_DIR / "property_recommender.joblib")
    price = joblib.load(MODEL_DIR / "price_predictor.joblib")
    return rec, price


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_report(filename: str) -> pd.DataFrame:
    path = REPORT_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


try:
    recommendation_model, price_model = load_models()
    properties = load_csv("properties_clean.csv")
    availability = load_csv("availability_clean.csv")
    price_observations = load_csv("price_observations_clean.csv")
    metrics = load_report("price_model_metrics.csv")
    comparison = load_report("price_model_comparison.csv")
    cross_validation = load_report("price_model_cross_validation.csv")
    error_categories = load_report("price_error_by_category.csv")
    recommendation_properties = recommendation_model["properties"].reset_index(drop=True)
    similarity_matrix = recommendation_model["similarity_matrix"]
except Exception as exc:
    st.error(f"Project resources could not be loaded: {exc}")
    st.stop()

if properties.empty:
    st.error("The properties dataset is empty. Check data/processed/properties_clean.csv.")
    st.stop()


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def npr(value):
    return f"NPR {float(value):,.0f}"


def metric_value(name: str, default=np.nan):
    if metrics.empty or not {"metric", "value"}.issubset(metrics.columns):
        return default
    row = metrics.loc[metrics["metric"].astype(str).str.lower() == name.lower(), "value"]
    return float(row.iloc[0]) if not row.empty else default


def published_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "published" not in frame.columns:
        return frame.copy()
    mask = frame["published"].astype(str).str.lower().isin(["true", "1"])
    return frame.loc[mask].copy()


def recommendations_for(index: int, count: int = 5) -> pd.DataFrame:
    scores = similarity_matrix[index]
    order = np.argsort(scores)[::-1]
    rows = []
    for candidate in order:
        if candidate == index:
            continue
        item = recommendation_properties.iloc[candidate]
        rows.append({
            "property_id": item["property_id"],
            "property_name": item["property_name"],
            "neighbourhood": item["neighbourhood"],
            "property_type": item["property_type"],
            "room_type": item["room_type"],
            "nightly_price_npr": float(item["nightly_price_npr"]),
            "rating": float(item["rating"]),
            "review_count": int(item["review_count"]),
            "similarity": float(scores[candidate]) * 100,
            "description": item.get("description", ""),
        })
        if len(rows) >= count:
            break
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🏔️ Kathmandu Stay</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">INTELLIGENCE • ML ANALYTICS</div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Explore Properties", "Recommendations", "Price Intelligence", "Data Explorer", "Model Performance"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown('<div class="info-label">PROJECT STATUS</div>', unsafe_allow_html=True)
    st.markdown("**2 ML models** · 5 datasets")
    st.caption("Educational / Internship Project")
    st.caption("Kathmandu, Nepal")
    st.divider()
    st.caption("Data snapshot · November 2026")
    if st.button("↻ Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ---------------------------------------------------------------------
# SHARED HEADER
# ---------------------------------------------------------------------
st.markdown('<div class="eyebrow">KATHMANDU • ACCOMMODATION ANALYTICS</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------
if page == "Overview":
    st.markdown(
        """
        <div class="hero">
          <span class="pill">MACHINE LEARNING WORKSPACE</span>
          <div class="hero-title">Understand the stay market.<br>Make smarter accommodation decisions.</div>
          <div class="hero-copy">Explore Kathmandu property samples, discover similar stays, estimate nightly prices, and inspect model performance in one analytical workspace.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="notice">DATA NOTE · Property, availability, and price records contain synthetic sample data. Outputs are experimental and must not be interpreted as verified live Airbnb listings or market prices.</div>', unsafe_allow_html=True)
    pub = published_frame(properties)
    avg_price = numeric(properties["nightly_price_npr"]).mean()
    avg_rating = numeric(properties["rating"]).mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Property records", f"{len(properties):,}", help="All records in the cleaned property dataset.")
    c2.metric("Published samples", f"{len(pub):,}", help="Records marked as published in the sample dataset.")
    c3.metric("Average nightly price", npr(avg_price))
    c4.metric("Average sample rating", f"{avg_rating:.2f} / 5")
    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Property landscape")
        st.caption("How the sample inventory is distributed across accommodation types.")
        counts = properties["property_type"].value_counts().rename_axis("Property type").reset_index(name="Properties")
        chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=6, color="#38BDF8").encode(
            x=alt.X("Properties:Q", title="Number of properties"),
            y=alt.Y("Property type:N", sort="-x", title=None),
            tooltip=["Property type", "Properties"],
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown("### Price by neighbourhood")
        st.caption("Average listed nightly price in the synthetic property sample.")
        area = properties.groupby("neighbourhood", as_index=False)["nightly_price_npr"].mean().nlargest(9, "nightly_price_npr")
        chart = alt.Chart(area).mark_bar(cornerRadiusEnd=6, color="#2DD4BF").encode(
            x=alt.X("nightly_price_npr:Q", title="Average price (NPR)"),
            y=alt.Y("neighbourhood:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("neighbourhood:N", title="Neighbourhood"), alt.Tooltip("nightly_price_npr:Q", format=",.0f", title="Average NPR")],
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
    st.markdown("### Workspace modules")
    modules = st.columns(3)
    module_info = [
        ("01", "Explore properties", "Filter the sample inventory by area, type, rating, and budget."),
        ("02", "Recommendations", "Use the trained TF-IDF similarity model to find comparable stays."),
        ("03", "Price intelligence", "Estimate nightly price using the saved Gradient Boosting pipeline."),
    ]
    for col, (num, title, desc) in zip(modules, module_info):
        with col:
            st.markdown(f'<div class="info-panel"><div class="section-kicker">{num} / MODULE</div><h3>{title}</h3><div style="color:#91A4BC;font-size:.86rem">{desc}</div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("### Recent sample inventory")
    preview_cols = ["property_name", "neighbourhood", "property_type", "room_type", "nightly_price_npr", "rating"]
    st.dataframe(properties[preview_cols].head(8).rename(columns={
        "property_name": "Property", "neighbourhood": "Neighbourhood", "property_type": "Type",
        "room_type": "Room", "nightly_price_npr": "Nightly price (NPR)", "rating": "Rating",
    }), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# PROPERTY EXPLORER
# ---------------------------------------------------------------------
elif page == "Explore Properties":
    st.title("Explore properties")
    st.caption("Browse and narrow down the Kathmandu accommodation sample using practical filters.")
    base = published_frame(properties)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        areas = ["All neighbourhoods"] + sorted(base["neighbourhood"].dropna().unique().tolist())
        area_filter = st.selectbox("Neighbourhood", areas)
    with f2:
        types = ["All property types"] + sorted(base["property_type"].dropna().unique().tolist())
        type_filter = st.selectbox("Property type", types)
    with f3:
        room_types = ["All room types"] + sorted(base["room_type"].dropna().unique().tolist())
        room_filter = st.selectbox("Room type", room_types)
    with f4:
        min_rating = st.selectbox("Minimum rating", [0, 3.0, 3.5, 4.0, 4.5], index=0)
    price_max = int(numeric(base["nightly_price_npr"]).max())
    budget = st.slider("Maximum nightly budget (NPR)", 0, max(price_max, 1000), min(8000, max(price_max, 1000)), step=250)
    filtered = base.copy()
    if area_filter != "All neighbourhoods":
        filtered = filtered[filtered["neighbourhood"] == area_filter]
    if type_filter != "All property types":
        filtered = filtered[filtered["property_type"] == type_filter]
    if room_filter != "All room types":
        filtered = filtered[filtered["room_type"] == room_filter]
    filtered = filtered[numeric(filtered["nightly_price_npr"]) <= budget]
    filtered = filtered[numeric(filtered["rating"]) >= float(min_rating)]
    a, b, c = st.columns(3)
    a.metric("Matching properties", f"{len(filtered):,}")
    b.metric("Average filtered price", npr(numeric(filtered["nightly_price_npr"]).mean()) if not filtered.empty else "—")
    c.metric("Average filtered rating", f"{numeric(filtered['rating']).mean():.2f} / 5" if not filtered.empty else "—")
    st.divider()
    if filtered.empty:
        st.info("No properties match these filters. Increase the budget or relax one of the filters.")
    else:
        display = filtered[["property_id", "property_name", "neighbourhood", "property_type", "room_type", "capacity", "nightly_price_npr", "rating", "review_count"]].copy()
        display.columns = ["ID", "Property", "Neighbourhood", "Type", "Room", "Capacity", "Price (NPR)", "Rating", "Reviews"]
        st.dataframe(display.sort_values("Price (NPR)"), use_container_width=True, hide_index=True)
        if {"latitude", "longitude"}.issubset(filtered.columns):
            st.markdown("### Approximate property locations")
            map_data = filtered[["latitude", "longitude"]].rename(columns={"latitude": "lat", "longitude": "lon"}).dropna()
            if not map_data.empty:
                st.map(map_data, zoom=11)
                st.caption("Map points represent coordinates in the sample dataset, not verified live listing locations.")

# ---------------------------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------------------------
elif page == "Recommendations":
    st.title("Similar-stay recommendations")
    st.caption("Choose a sample property and discover comparable options using TF-IDF text features and cosine similarity.")
    names = recommendation_properties["property_id"].astype(str) + " · " + recommendation_properties["property_name"].astype(str)
    selected_label = st.selectbox("Starting property", names.tolist())
    idx = names.tolist().index(selected_label)
    selected = recommendation_properties.iloc[idx]
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(f'<div class="info-panel"><div class="info-label">SELECTED PROPERTY</div><h2 style="margin:8px 0">{html.escape(str(selected["property_name"]))}</h2><div style="color:#91A4BC">{html.escape(str(selected["neighbourhood"]))} · {html.escape(str(selected["property_type"]))} · {html.escape(str(selected["room_type"]))}</div><div class="property-price">{npr(selected["nightly_price_npr"])} / night</div></div>', unsafe_allow_html=True)
    with right:
        st.metric("Sample rating", f'{float(selected["rating"]):.1f} / 5')
        st.metric("Review count", f'{int(selected["review_count"]):,}')
    st.write("")
    with st.expander("View property description and amenities"):
        st.write(selected.get("description", "No description available."))
        st.caption(str(selected.get("amenities", "")).replace(";", " · "))
    count = st.slider("Number of similar properties", 3, 10, 6)
    max_budget = st.number_input("Optional maximum recommended price (NPR)", min_value=0, value=0, step=500, help="Set to 0 to show recommendations at any price.")
    recs = recommendations_for(idx, 10)
    if max_budget > 0:
        recs = recs[recs["nightly_price_npr"] <= max_budget]
    recs = recs.head(count)
    st.markdown("### Recommended alternatives")
    if recs.empty:
        st.info("No similar properties match this budget. Increase the price limit.")
    else:
        card_cols = st.columns(2)
        for i, (_, item) in enumerate(recs.iterrows()):
            with card_cols[i % 2]:
                st.markdown(
                    f'<div class="property-card"><div class="score">MATCH SCORE · {item["similarity"]:.1f}%</div><div class="property-name">{html.escape(str(item["property_name"]))}</div><div class="property-meta">{html.escape(str(item["neighbourhood"]))} · {html.escape(str(item["property_type"]))} · {html.escape(str(item["room_type"]))}</div><div style="color:#B7C8DA;font-size:.82rem">Capacity: {int(item["review_count"]*0 + selected["capacity"])} guests &nbsp; · &nbsp; Rating: {item["rating"]:.1f}/5</div><div class="property-price">{npr(item["nightly_price_npr"])} / night</div></div>',
                    unsafe_allow_html=True,
                )
                st.write("")
    st.caption("Similarity is a content-based score from the trained TF-IDF/cosine-similarity model. It is not a booking probability, review score, or guarantee of suitability.")

# ---------------------------------------------------------------------
# PRICE INTELLIGENCE
# ---------------------------------------------------------------------
elif page == "Price Intelligence":
    st.title("Nightly price intelligence")
    st.caption("Estimate a sample property's nightly price for a planned stay date and booking lead time.")
    st.markdown('<div class="notice">MODEL INPUT NOTE · The saved model uses the selected property’s existing baseline nightly price as one of its 13 input features. This is a conditional estimate, not a price forecast independent of a known baseline.</div>', unsafe_allow_html=True)
    base = published_frame(properties)
    labels = base["property_id"].astype(str) + " · " + base["property_name"].astype(str)
    with st.form("price_intelligence_form"):
        chosen = st.selectbox("Choose a property", labels.tolist())
        pos = labels.tolist().index(chosen)
        item = base.iloc[pos]
        st.markdown(f"**Baseline nightly price:** {npr(item['nightly_price_npr'])}  ·  **{item['neighbourhood']}**  ·  **{item['property_type']}**")
        col1, col2 = st.columns(2)
        with col1:
            lead = st.number_input("Booking lead time (days)", 0, 730, 30, 1)
        with col2:
            stay = st.date_input("Planned stay date", value=date.today())
        submit = st.form_submit_button("Estimate nightly price", use_container_width=True)
    if submit:
        model_input = pd.DataFrame([{
            "city": item["city"], "neighbourhood": item["neighbourhood"],
            "property_type": item["property_type"], "room_type": item["room_type"],
            "capacity": item["capacity"], "bedrooms": item["bedrooms"], "bathrooms": item["bathrooms"],
            "lead_time_days": int(lead), "stay_month": stay.month, "stay_day": stay.day,
            "stay_day_of_week": stay.weekday(), "nightly_price_npr": item["nightly_price_npr"],
            "amenities": item["amenities"],
        }])
        try:
            prediction = float(price_model.predict(model_input)[0])
            if not np.isfinite(prediction):
                raise ValueError("Model returned a non-finite value.")
            baseline = float(item["nightly_price_npr"])
            delta = prediction - baseline
            st.divider()
            st.markdown("### Estimate")
            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated nightly price", npr(prediction), delta=npr(delta))
            c2.metric("Existing baseline", npr(baseline))
            c3.metric("Difference", f"{delta / baseline * 100:+.1f}%" if baseline else "—")
            compare = pd.DataFrame({"Price basis": ["Baseline", "Model estimate"], "NPR": [baseline, prediction]})
            chart = alt.Chart(compare).mark_bar(cornerRadiusEnd=7).encode(
                x=alt.X("NPR:Q", title="Nightly price (NPR)"),
                y=alt.Y("Price basis:N", title=None, sort=None),
                color=alt.Color("Price basis:N", scale=alt.Scale(domain=["Baseline", "Model estimate"], range=["#64748B", "#2DD4BF"]), legend=None),
                tooltip=[alt.Tooltip("NPR:Q", format=",.0f")],
            ).properties(height=120)
            st.altair_chart(chart, use_container_width=True)
            st.caption("Experimental estimate from synthetic sample data. Not a live quote or verified market price.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ---------------------------------------------------------------------
# DATA EXPLORER
# ---------------------------------------------------------------------
elif page == "Data Explorer":
    st.title("Data explorer")
    st.caption("Inspect the datasets used in the analysis and understand their role and limitations.")
    tabs = st.tabs(["Properties", "Availability", "Price observations", "Tourism context", "Dataset notes"])
    with tabs[0]:
        st.metric("Property records", f"{len(properties):,}")
        if "neighbourhood" in properties:
            dist = properties["neighbourhood"].value_counts().rename_axis("Neighbourhood").reset_index(name="Records")
            st.altair_chart(alt.Chart(dist).mark_bar(color="#38BDF8", cornerRadiusEnd=5).encode(
                x=alt.X("Records:Q"), y=alt.Y("Neighbourhood:N", sort="-x", title=None), tooltip=["Neighbourhood", "Records"]
            ).properties(height=320), use_container_width=True)
        st.dataframe(properties, use_container_width=True, hide_index=True)
    with tabs[1]:
        st.metric("Availability records", f"{len(availability):,}")
        if not availability.empty:
            st.dataframe(availability.head(250), use_container_width=True, hide_index=True)
            st.caption("Showing up to the first 250 records to keep the page responsive.")
        else:
            st.info("Availability data is not available.")
    with tabs[2]:
        st.metric("Price observations", f"{len(price_observations):,}")
        if not price_observations.empty:
            st.dataframe(price_observations, use_container_width=True, hide_index=True)
        else:
            st.info("Price observation data is not available.")
    with tabs[3]:
        tourism = load_csv("nepal_monthly_tourist_arrivals_clean.csv")
        st.metric("Monthly tourism records", f"{len(tourism):,}")
        if not tourism.empty:
            st.dataframe(tourism, use_container_width=True, hide_index=True)
            st.caption("National tourism arrivals provide broad context only; they are not direct Kathmandu Airbnb occupancy or booking records.")
    with tabs[4]:
        st.markdown("#### Data provenance and limitations")
        st.markdown("""
        - Property, availability, and price-observation datasets are synthetic sample data.
        - The booking-cancellations benchmark is a separate hotel dataset and is not joined to Kathmandu property IDs.
        - Nepal monthly tourist arrivals are contextual national-level data, not direct accommodation demand measurements.
        - Reviews and monthly occupancy files were not populated with usable records at the time of analysis.
        - Sample listings are not verified live Airbnb listings.
        """)

# ---------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model performance")
    st.caption("Review held-out price prediction metrics, model comparisons, cross-validation, and error patterns.")
    mae = metric_value("MAE")
    rmse = metric_value("RMSE")
    r2 = metric_value("R2")
    improvement = metric_value("MAE Improvement Percentage")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean absolute error", npr(mae))
    c2.metric("Root mean squared error", npr(rmse))
    c3.metric("R² score", f"{r2:.4f}")
    c4.metric("MAE improvement vs baseline", f"{improvement:.2f}%")
    st.caption(f"Evaluation split: {int(metric_value('Training rows', 0)):,} training rows / {int(metric_value('Testing rows', 0)):,} test rows, grouped by property ID with no property overlap.")
    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("### Model comparison")
        if not comparison.empty:
            long = comparison.melt(id_vars="model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Error (NPR)")
            chart = alt.Chart(long).mark_bar(cornerRadiusEnd=5).encode(
                x=alt.X("Error (NPR):Q", title="Error (NPR)"),
                y=alt.Y("model:N", sort="-x", title=None),
                color=alt.Color("Metric:N", scale=alt.Scale(range=["#38BDF8", "#2DD4BF"])),
                yOffset="Metric:N",
                tooltip=["model", "Metric", alt.Tooltip("Error (NPR):Q", format=",.2f")],
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(comparison, use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Five-fold cross-validation")
        if not cross_validation.empty:
            cv = cross_validation[["fold", "GB_MAE", "Baseline_MAE"]].melt("fold", var_name="Model", value_name="MAE")
            cv["Model"] = cv["Model"].map({"GB_MAE": "Gradient Boosting", "Baseline_MAE": "Baseline"})
            chart = alt.Chart(cv).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("fold:O", title="Fold"),
                y=alt.Y("MAE:Q", title="Mean absolute error (NPR)"),
                color=alt.Color("Model:N", scale=alt.Scale(range=["#38BDF8", "#94A3B8"])),
                tooltip=["fold", "Model", alt.Tooltip("MAE:Q", format=",.2f")],
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(cross_validation, use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("### Error by price segment")
    if not error_categories.empty:
        chart_data = error_categories.copy()
        chart = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=6, color="#FBBF24").encode(
            x=alt.X("mean_absolute_error:Q", title="Mean absolute error (NPR)"),
            y=alt.Y("price_category:N", sort="-x", title=None),
            tooltip=["price_category", "sample_count", alt.Tooltip("mean_absolute_error:Q", format=",.0f"), alt.Tooltip("mean_percentage_error:Q", format=".1f")],
        ).properties(height=220)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(chart_data, use_container_width=True, hide_index=True)
    st.markdown("### How to interpret these results")
    st.markdown("""
    - **MAE** is the average absolute difference between predictions and observed sample prices, in NPR.
    - **RMSE** penalizes larger errors more strongly than MAE.
    - **R²** measures fit relative to the test-set mean; it does not establish real-world validity.
    - The split keeps properties separate between training and test sets, reducing property-level leakage.
    - The baseline nightly price is an input feature. The model therefore estimates an adjustment conditional on an existing listed baseline.
    """)
    st.markdown('<div class="notice">IMPORTANT · The very high R² and low error are measured on synthetic observations that are strongly correlated with the supplied baseline price. They should not be presented as evidence of real-world Airbnb price accuracy. Luxury and mid-range samples show larger absolute errors.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.divider()
left, right = st.columns([3, 1])
with left:
    st.caption("KATHMANDU STAY INTELLIGENCE  ·  Machine Learning Internship Project")
with right:
    st.caption("Prototype • Synthetic sample data")
