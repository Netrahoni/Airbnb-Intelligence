
from pathlib import Path
from datetime import date
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

st.set_page_config(
    page_title="Airbnb Intelligence | Kathmandu Analytics",
    page_icon=str(ROOT / "assets" / "airbnb-intelligence-icon.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
  --canvas: #080f1b;
  --surface: #101d2d;
  --surface-raised: #14243a;
  --surface-hover: #192d45;
  --line: rgba(148, 171, 195, .16);
  --line-strong: rgba(111, 190, 218, .30);
  --ink: #edf4fb;
  --muted: #91a5bb;
  --blue: #59c8f2;
  --teal: #45d5c2;
}

html, body, [class*="css"], button, input, textarea {
  font-family: 'DM Sans', sans-serif;
}
.stApp {
  color: var(--ink);
  background:
    radial-gradient(ellipse at 82% 0%, rgba(37, 119, 151, .14), transparent 34%),
    radial-gradient(ellipse at 4% 38%, rgba(43, 83, 132, .10), transparent 30%),
    var(--canvas);
}
.block-container {
  max-width: 1580px;
  padding: 1.55rem clamp(1rem, 2.5vw, 2.5rem) 3.5rem;
}
h1, h2, h3, h4 {
  color: #f5f9fd !important;
  font-family: 'Manrope', 'DM Sans', sans-serif !important;
  font-weight: 750 !important;
  letter-spacing: -.045em !important;
}
p, li, label { color: var(--ink); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }

/* Sidebar frame */
section[data-testid="stSidebar"] {
  width: 292px !important;
  min-width: 292px !important;
  background: linear-gradient(180deg, #0d1929 0%, #0a1422 100%) !important;
  border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] > div:first-child { width: 292px !important; }
section[data-testid="stSidebar"] .block-container {
  padding: 1.15rem .95rem 1.7rem !important;
  gap: .85rem;
}

/* Brand: the icon and wordmark are intentionally separate */
.brand-shell {
  display: grid;
  grid-template-columns: 62px minmax(0, 1fr);
  align-items: center;
  column-gap: 13px;
  padding: 14px 12px;
  margin: 0 0 1.1rem;
  border: 1px solid var(--line-strong);
  border-radius: 16px;
  background:
    radial-gradient(circle at 100% 0%, rgba(69, 213, 194, .10), transparent 42%),
    linear-gradient(145deg, #15283d, #101d2e);
  box-shadow: 0 12px 30px rgba(0,0,0,.20);
}
.brand-logo-wrap {
  width: 62px;
  height: 62px;
  display: grid;
  place-items: center;
  padding: 9px;
  border-radius: 14px;
  background: linear-gradient(145deg, #f8fbfd, #e6f0f5);
  border: 1px solid rgba(255,255,255,.78);
  box-shadow: 0 5px 15px rgba(0,0,0,.16), inset 0 1px 0 white;
}
.brand-logo-wrap img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.brand-copy { min-width: 0; }
.brand-title {
  margin: 0 0 5px;
  color: #f5f9fd;
  font-family: 'Manrope', sans-serif;
  font-size: 1rem;
  font-weight: 800;
  line-height: 1.18;
  letter-spacing: -.045em;
  overflow-wrap: anywhere;
}
.brand-tagline {
  margin: 0;
  color: #a9bbcd;
  font-size: .68rem;
  font-weight: 500;
  line-height: 1.45;
  text-align: left;
  letter-spacing: 0;
}
.brand-divider {
  grid-column: 1 / -1;
  height: 1px;
  margin: 12px 0 9px;
  background: linear-gradient(90deg, rgba(89,200,242,.45), rgba(148,163,184,.08));
}
.brand-kicker {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  gap: 8px;
  color: #91acc1;
  font-size: .58rem;
  font-weight: 800;
  line-height: 1.4;
  letter-spacing: 1.25px;
  text-transform: uppercase;
}
.brand-kicker-dot {
  flex: 0 0 6px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--teal);
  box-shadow: 0 0 0 4px rgba(69,213,194,.11);
}

/* Workspace navigation */
section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
  margin: .15rem 0 .55rem;
  color: #8ba3ba !important;
  font-size: .66rem !important;
  font-weight: 800 !important;
  letter-spacing: 1.6px !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
  gap: .38rem !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
  min-height: 43px;
  display: flex;
  align-items: center;
  padding: .58rem .72rem !important;
  border: 1px solid rgba(148,163,184,.09) !important;
  border-radius: 11px !important;
  background: rgba(255,255,255,.018) !important;
  transition: background .18s ease, border-color .18s ease, transform .18s ease;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: rgba(85,199,243,.075) !important;
  border-color: rgba(85,199,243,.24) !important;
  transform: translateX(2px);
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: linear-gradient(100deg, rgba(56,189,248,.16), rgba(45,212,191,.085)) !important;
  border-color: rgba(85,199,243,.34) !important;
  box-shadow: inset 3px 0 0 var(--teal);
}
.sidebar-project-meta {
  padding: .85rem .9rem;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: linear-gradient(145deg, rgba(19,36,56,.82), rgba(12,25,41,.82));
}
.sidebar-project-meta .meta-heading {
  margin-bottom: .4rem;
  color: #b5cadd;
  font-size: .62rem;
  font-weight: 800;
  letter-spacing: 1.3px;
  text-transform: uppercase;
}
.sidebar-project-meta .meta-copy {
  color: var(--muted);
  font-size: .73rem;
  line-height: 1.55;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }

/* Page hero and section hierarchy */
.hero {
  position: relative;
  overflow: hidden;
  padding: clamp(1.35rem, 3vw, 2.15rem);
  margin-bottom: 1.4rem;
  border: 1px solid rgba(85,199,243,.22);
  border-radius: 20px;
  background:
    radial-gradient(circle at 90% 20%, rgba(66,214,195,.14), transparent 28%),
    linear-gradient(115deg, #142b45 0%, #10263a 55%, #102e35 100%);
  box-shadow: 0 18px 48px rgba(0,0,0,.18);
}
.hero-eyebrow, .section-label {
  color: #78ddec;
  font-size: .69rem;
  font-weight: 800;
  letter-spacing: 1.7px;
  text-transform: uppercase;
}
.hero-title {
  margin: .65rem 0 .55rem;
  color: #f8fbff;
  font-family: 'Manrope', sans-serif;
  font-size: clamp(1.8rem, 3.4vw, 2.55rem);
  font-weight: 800;
  line-height: 1.12;
  letter-spacing: -.055em;
}
.hero-description {
  max-width: 760px;
  margin-bottom: 0;
  color: #bfd0df;
  font-size: .94rem;
  line-height: 1.65;
}
.section-label { margin: .25rem 0 .8rem; }

/* KPI cards, panels and data displays */
[data-testid="stMetric"] {
  min-height: 116px;
  padding: 1.05rem 1.1rem;
  border: 1px solid var(--line);
  border-radius: 15px;
  background: linear-gradient(145deg, rgba(20,36,57,.98), rgba(13,25,42,.98));
  box-shadow: 0 9px 25px rgba(0,0,0,.12);
  transition: transform .18s ease, border-color .18s ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  border-color: rgba(66,214,195,.40);
}
[data-testid="stMetricLabel"] { color: #a6b8cc !important; font-size: .78rem !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #f5faff !important; font-family: 'Manrope', sans-serif; font-size: clamp(1.25rem,2vw,1.75rem) !important; font-weight: 800 !important; letter-spacing: -.04em; }
[data-testid="stMetricDelta"] { font-size: .76rem !important; }
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--line) !important;
  border-radius: 15px !important;
  background: rgba(16,29,45,.62);
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 12px;
}
[data-testid="stTabs"] button { font-weight: 650; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--blue) !important; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] {
  border-color: var(--line-strong);
  border-radius: 10px;
  background: var(--surface-raised);
}
input, textarea { color: var(--ink) !important; }
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  min-height: 2.6rem;
  border: 1px solid rgba(85,199,243,.25);
  border-radius: 10px;
  font-weight: 700;
  box-shadow: 0 5px 16px rgba(2,132,199,.12);
  transition: transform .18s ease, border-color .18s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
  border-color: rgba(66,214,195,.55);
  transform: translateY(-1px);
}
[data-testid="stAlert"], div[data-testid="stExpander"] { border-radius: 12px; }
hr { border-color: var(--line); margin: 1.25rem 0; }
.info-card, .property-card {
  margin-bottom: .8rem;
  padding: 1.05rem 1.15rem;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: linear-gradient(145deg, #14243a, #101c2e);
}
.property-card { transition: transform .18s ease, border-color .18s ease; }
.property-card:hover { transform: translateY(-2px); border-color: rgba(66,214,195,.34); }
.property-card-title { color: #f2f7fc; font-size: 1.02rem; font-weight: 750; }
.property-card-meta { margin-top: .25rem; color: #9baec3; font-size: .79rem; }
.similarity { color: var(--teal); font-weight: 800; }
.footer { margin-top: 2.5rem; padding-top: 1.3rem; border-top: 1px solid var(--line); color: #8297ad; text-align: center; font-size: .8rem; }

@media (max-width: 768px) {
  section[data-testid="stSidebar"] { width: min(88vw, 292px) !important; min-width: min(88vw, 292px) !important; }
  section[data-testid="stSidebar"] > div:first-child { width: min(88vw, 292px) !important; }
  .block-container { padding: .9rem .9rem 2rem; }
  .brand-shell { grid-template-columns: 54px minmax(0,1fr); column-gap: 10px; padding: 11px; }
  .brand-logo-wrap { width: 54px; height: 54px; }
  .brand-title { font-size: .94rem; }
  .brand-tagline { font-size: .65rem; }
  .hero { padding: 1.25rem; border-radius: 15px; }
  .hero-title { font-size: 1.75rem; }
  [data-testid="stMetric"] { min-height: 96px; padding: .85rem; }
}
</style>
""", unsafe_allow_html=True)

# Layer the maintained dashboard theme after Streamlit base styles.
DASHBOARD_THEME = (ROOT / "assets" / "dashboard-theme.css").read_text(encoding="utf-8")
st.markdown(f"<style>{DASHBOARD_THEME}</style>", unsafe_allow_html=True)


# ============================================================
# 3. DATA AND MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    recommender = joblib.load(
        MODEL_DIR / "property_recommender.joblib"
    )
    price_predictor = joblib.load(
        MODEL_DIR / "price_predictor.joblib"
    )
    return recommender, price_predictor


@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_optional_csv(*filenames):
    """Load optional datasets using alternative filenames and locations."""
    search_dirs = [
        DATA_DIR,
        ROOT / "data" / "raw",
        ROOT,
    ]

    for filename in filenames:
        for folder in search_dirs:
            path = folder / filename

            if path.exists():
                try:
                    return pd.read_csv(path)
                except Exception as error:
                    st.warning(f"Could not read {filename}: {error}")

    return pd.DataFrame()

@st.cache_data
def load_report(filename):
    path = REPORT_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


try:
    recommender, price_model = load_models()

    properties = load_csv("properties_clean.csv")
    availability = load_csv("availability_clean.csv")
    price_observations = load_csv("price_observations_clean.csv")

    # Optional context datasets for the executive dashboard.
    tourist_arrivals = load_optional_csv(
    "nepal_monthly_tourist_arrivals_clean.csv",
    "nepal_monthly_tourist_arrivals.csv",
)
    occupancy_context = load_optional_csv(
    "kathmandu_monthly_occupancy_clean.csv",
    "kathmandu_monthly_occupancy.csv",
)

    recommendation_properties = recommender["properties"]
    similarity_matrix = recommender["similarity_matrix"]

    metrics = load_report("price_model_metrics.csv")
    price_comparison = load_report("price_model_comparison.csv")
    cross_validation = load_report("price_model_cross_validation.csv")
    error_categories = load_report("price_error_by_category.csv")

    if properties.empty:
        raise ValueError("The properties dataset is empty.")

except Exception as error:
    st.error(f"Project resources could not be loaded: {error}")
    st.stop()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def money(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"NPR {float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def is_published(series):
    return series.astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )


def page_header(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">{eyebrow}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-description">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def get_recommendations(property_index, number=5):
    scores = similarity_matrix[property_index]
    ranked = np.argsort(scores)[::-1]

    results = []

    for index in ranked:
        if index == property_index:
            continue

        item = recommendation_properties.iloc[index]

        results.append({
            "Property ID": item["property_id"],
            "Property Name": item["property_name"],
            "Neighbourhood": item["neighbourhood"],
            "Property Type": item["property_type"],
            "Room Type": item["room_type"],
            "Nightly Price (NPR)": item["nightly_price_npr"],
            "Rating": item["rating"],
            "Similarity (%)": round(float(scores[index]) * 100, 2),
        })

        if len(results) >= number:
            break

    return pd.DataFrame(results)


def published_data(frame):
    if "published" not in frame.columns:
        return frame.copy()
    return frame[is_published(frame["published"])].copy()


# ============================================================
# 5. SIDEBAR
# ============================================================

logo_mark_svg = (ROOT / "assets" / "airbnb-intelligence-mark.svg").read_text(encoding="utf-8")
with st.sidebar:
    st.markdown(
        f"""
        <div class="brand-shell">
          <div class="brand-logo-wrap">
            <img src="data:image/svg+xml;utf8,{quote(logo_mark_svg)}" alt="Airbnb Intelligence brand mark">
          </div>
          <div class="brand-copy">
            <div class="brand-title">Airbnb Intelligence</div>
            <div class="brand-tagline">Turning accommodation data into intelligent decisions.</div>
          </div>
          <div class="brand-divider"></div>
          <div class="brand-kicker"><span class="brand-kicker-dot"></span> Kathmandu · Accommodation Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Executive Analytics",
            "Property Recommendations",
            "Price Prediction",
            "Data Analysis",
            "Model Evaluation",
        ],
        label_visibility="visible",
        format_func=lambda item: {
            "Executive Analytics": "▦  Executive Analytics",
            "Property Recommendations": "⌕  Property Recommendations",
            "Price Prediction": "↗  Price Prediction",
            "Data Analysis": "◫  Data Analysis",
            "Model Evaluation": "◎  Model Evaluation",
        }.get(item, item),
    )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-project-meta">
          <div class="meta-heading">Project workspace</div>
          <div class="meta-copy">Machine Learning Internship Project<br>Python · Scikit-learn · Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("**DATA STATUS**")
    st.markdown(
        '<span style="color:#FBBF24;font-weight:700;">● '
        'Synthetic sample data</span>',
        unsafe_allow_html=True,
    )

    st.caption("Experimental educational project")


# ============================================================
# 6. OVERVIEW
# ============================================================

if page == "Executive Analytics":

    page_header(
        "EXECUTIVE ANALYTICS",
        "Kathmandu Accommodation Intelligence",
        "A decision-support view of accommodation inventory, sample pricing, "
        "neighbourhood patterns and Nepal tourism context.",
    )

    st.warning(
        "Data transparency: Property and price observations include synthetic "
        "sample data. They are not verified live Airbnb listings, actual market "
        "prices, revenue, or confirmed bookings. Tourist arrivals are Nepal-wide "
        "context and must not be interpreted as Kathmandu-only demand."
    )

    with st.container(border=True):
        section_heading(
            "Executive filters",
            "Focus the property-level indicators and visualisations.",
        )

        neighbourhoods = sorted(properties["neighbourhood"].dropna().astype(str).unique()) if "neighbourhood" in properties else []
        property_types = sorted(properties["property_type"].dropna().astype(str).unique()) if "property_type" in properties else []
        room_types = sorted(properties["room_type"].dropna().astype(str).unique()) if "room_type" in properties else []

        price_floor, price_ceiling = 0, 1
        if "nightly_price_npr" in properties:
            all_prices = numeric(properties["nightly_price_npr"]).dropna()
            if not all_prices.empty:
                price_floor = int(np.floor(all_prices.min()))
                price_ceiling = int(np.ceil(all_prices.max()))
                if price_ceiling <= price_floor:
                    price_ceiling = price_floor + 1

        f1, f2, f3 = st.columns([1.2, 1.2, 1])
        with f1:
            selected_neighbourhoods = st.multiselect(
                "Neighbourhood", neighbourhoods, default=neighbourhoods,
                key="exec_neighbourhood_filter",
            )
        with f2:
            selected_types = st.multiselect(
                "Property type", property_types, default=property_types,
                key="exec_property_type_filter",
            )
        with f3:
            published_only = st.toggle(
                "Published only", value=False, key="exec_published_filter"
            )

        # Additional filters keep the executive view useful without changing source data.
        selected_room_types = st.multiselect(
            "Room type", room_types, default=room_types, key="exec_room_type_filter"
        )
        filter_col1, filter_col2 = st.columns([1.4, 1])
        with filter_col1:
            selected_price_range = st.slider(
                "Sample nightly price range (NPR)", min_value=price_floor,
                max_value=price_ceiling, value=(price_floor, price_ceiling),
                key="exec_price_range_filter",
            )
        with filter_col2:
            minimum_rating = st.slider(
                "Minimum guest rating", min_value=0.0, max_value=5.0,
                value=0.0, step=0.1, key="exec_min_rating_filter",
            )
    executive_properties = properties.copy()
    if "neighbourhood" in executive_properties:
        executive_properties = executive_properties[
            executive_properties["neighbourhood"].astype(str).isin(selected_neighbourhoods)
        ]
    if "property_type" in executive_properties:
        executive_properties = executive_properties[
            executive_properties["property_type"].astype(str).isin(selected_types)
        ]
    if published_only and "published" in executive_properties:
        executive_properties = executive_properties[
            is_published(executive_properties["published"])
        ]

    if "room_type" in executive_properties:
        executive_properties = executive_properties[
            executive_properties["room_type"].astype(str).isin(selected_room_types)
        ]
    if "nightly_price_npr" in executive_properties:
        filter_prices = numeric(executive_properties["nightly_price_npr"])
        executive_properties = executive_properties[
            filter_prices.between(selected_price_range[0], selected_price_range[1])
        ]
    if minimum_rating > 0 and "rating" in executive_properties:
        executive_properties = executive_properties[
            numeric(executive_properties["rating"]).fillna(0) >= minimum_rating
        ]

    prices = numeric(executive_properties["nightly_price_npr"]).dropna() if "nightly_price_npr" in executive_properties else pd.Series(dtype=float)
    ratings = numeric(executive_properties["rating"]).dropna() if "rating" in executive_properties else pd.Series(dtype=float)
    total = len(executive_properties)
    published = len(published_data(executive_properties))
    avg_price = prices.mean() if not prices.empty else np.nan
    median_price = prices.median() if not prices.empty else np.nan
    avg_rating = ratings.mean() if not ratings.empty else np.nan

    st.markdown(
        '<div class="section-label">EXECUTIVE PERFORMANCE SNAPSHOT</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Accommodation Portfolio Overview")
    st.caption(
        "A high-level snapshot of the selected accommodation inventory "
        "and its sample pricing characteristics."
    )
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(
            "Total Properties",
            f"{total:,}",
            help="Total property records matching the selected filters.",
        )
    with k2:
        st.metric(
            "Published Properties",
            f"{published:,}",
            help="Number of published property records in the selection.",
        )
    with k3:
        publication_rate = (published / total * 100) if total else 0
        st.metric(
            "Publication Rate",
            f"{publication_rate:.1f}%",
            help="Published properties divided by filtered property records.",
        )

    st.markdown("")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(
            "Average Nightly Price",
            money(avg_price),
            help="Mean sample nightly price in NPR.",
        )
    with k2:
        st.metric(
            "Median Nightly Price",
            money(median_price),
            help="Median sample nightly price in NPR.",
        )
    with k3:
        st.metric(
            "Average Guest Rating",
            f"{avg_rating:.2f} / 5" if pd.notna(avg_rating) else "N/A",
            help="Mean guest rating among selected properties with a rating.",
        )
    st.caption(
        f"Showing {total:,} of {len(properties):,} property records. "
        "Price and rating values are sample dataset statistics, not verified market benchmarks."
    )

    # ========================================================
    # MARKET INTELLIGENCE
    # ========================================================
    st.divider()
    section_heading(
        "Market Intelligence",
        "Explore neighbourhood competitiveness, property inventory "
        "and accommodation quality.",
    )
    market_data = executive_properties.copy()
    if "nightly_price_npr" in market_data:
        market_data["nightly_price_npr"] = numeric(
            market_data["nightly_price_npr"]
        )
    if "rating" in market_data:
        market_data["rating"] = numeric(market_data["rating"])

    # --------------------------------------------------------
    # NEIGHBOURHOOD PERFORMANCE
    # --------------------------------------------------------
    neighbourhood_summary = pd.DataFrame()
    if not market_data.empty and "neighbourhood" in market_data.columns:
        neighbourhood_groups = market_data.groupby("neighbourhood")
        neighbourhood_summary = neighbourhood_groups.size().to_frame(
            "Property_Count"
        )
        if "nightly_price_npr" in market_data.columns:
            neighbourhood_summary["Average_Price"] = neighbourhood_groups[
                "nightly_price_npr"
            ].mean()
            neighbourhood_summary["Median_Price"] = neighbourhood_groups[
                "nightly_price_npr"
            ].median()
        if "rating" in market_data.columns:
            neighbourhood_summary["Average_Rating"] = neighbourhood_groups[
                "rating"
            ].mean()

    insight_columns = st.columns(3)
    if not neighbourhood_summary.empty and "Median_Price" in neighbourhood_summary:
        median_prices = neighbourhood_summary["Median_Price"].dropna()
        if not median_prices.empty:
            highest_median_area = median_prices.idxmax()
            insight_columns[0].metric(
                "Highest median-price area",
                str(highest_median_area),
                help=(
                    "Median sample nightly price: "
                    f"{money(median_prices.loc[highest_median_area])}."
                ),
            )
        else:
            insight_columns[0].metric("Highest median-price area", "N/A")
    else:
        insight_columns[0].metric("Highest median-price area", "N/A")

    if not market_data.empty and "property_type" in market_data.columns:
        inventory_segments = market_data["property_type"].dropna().value_counts()
        if not inventory_segments.empty:
            largest_segment = inventory_segments.index[0]
            segment_share = inventory_segments.iloc[0] / len(market_data) * 100
            insight_columns[1].metric(
                "Largest inventory segment",
                str(largest_segment),
                help=f"{segment_share:.1f}% of filtered property records.",
            )
        else:
            insight_columns[1].metric("Largest inventory segment", "N/A")
    else:
        insight_columns[1].metric("Largest inventory segment", "N/A")

    if not market_data.empty and "rating" in market_data.columns:
        rating_coverage = market_data["rating"].notna().mean() * 100
        insight_columns[2].metric("Rating coverage", f"{rating_coverage:.1f}%")
    else:
        insight_columns[2].metric("Rating coverage", "N/A")

    left, right = st.columns([1.25, 1])
    with left:
        section_heading("Neighbourhood performance")
        if not neighbourhood_summary.empty:
            if {"Average_Price", "Median_Price"}.issubset(
                neighbourhood_summary.columns
            ):
                price_summary = (
                    neighbourhood_summary.dropna(
                        subset=["Average_Price", "Median_Price"]
                    )
                    .sort_values("Average_Price", ascending=False)
                    .head(12)
                )
                if not price_summary.empty:
                    st.bar_chart(
                        price_summary[["Average_Price", "Median_Price"]],
                        color=["#38BDF8", "#2DD4BF"],
                    )
                    st.caption(
                        "NPR per night. Small neighbourhood samples should be "
                        "interpreted cautiously."
                    )
                else:
                    st.info("No valid neighbourhood price records for this selection.")
            else:
                st.info("Nightly price information is unavailable for this selection.")

            summary_columns = [
                column
                for column in (
                    "Property_Count",
                    "Average_Price",
                    "Median_Price",
                    "Average_Rating",
                )
                if column in neighbourhood_summary.columns
            ]
            st.dataframe(
                neighbourhood_summary[summary_columns]
                .sort_values("Property_Count", ascending=False)
                .reset_index(),
                hide_index=True,
                alt="Property counts, prices, and ratings by neighbourhood",
            )
        else:
            st.info("Neighbourhood information is unavailable for this selection.")

    # Geospatial intelligence: use only coordinates supplied by the property dataset.
    st.divider()
    section_heading(
        "Kathmandu property map",
        "Property-level sample coordinates from the selected dataset. Markers are not verified live Airbnb listings."
    )
    coordinate_columns = {"latitude", "longitude"}
    if coordinate_columns.issubset(executive_properties.columns):
        map_data = executive_properties.copy()
        map_data["latitude"] = numeric(map_data["latitude"])
        map_data["longitude"] = numeric(map_data["longitude"])
        map_data = map_data.dropna(subset=["latitude", "longitude"])
        map_data = map_data[
            map_data["latitude"].between(-90, 90)
            & map_data["longitude"].between(-180, 180)
        ]
        if not map_data.empty:
            st.map(
                map_data[["latitude", "longitude"]],
                latitude="latitude", longitude="longitude",
                color="#42D6C3", size=18, zoom=11,
                width="stretch",
            )
            st.caption(
                f"Mapped {len(map_data):,} of {len(executive_properties):,} selected records with valid coordinates. "
                "Coordinates and property records are part of the project's synthetic sample data."
            )
        else:
            st.info("No valid coordinates are available for the current filter selection.")
    else:
        st.info("The current property dataset does not include latitude/longitude fields, so a map cannot be drawn without inventing locations.")

    with right:
        section_heading("Accommodation mix")
        if not market_data.empty and "property_type" in market_data.columns:
            st.bar_chart(
                market_data["property_type"]
                .fillna("Unspecified")
                .value_counts(),
                color="#818CF8",
            )
        else:
            st.info("Property-type information is unavailable.")
        if not market_data.empty and "room_type" in market_data.columns:
            st.caption("Room-type composition")
            st.bar_chart(
                market_data["room_type"]
                .fillna("Unspecified")
                .value_counts(),
                color="#F59E0B",
            )

    st.divider()
    section_heading(
        "Pricing distribution",
        "The spread of sample nightly prices provides more context than an average alone.",
    )
    if not prices.empty:
        bins = pd.cut(prices, bins=min(10, max(3, prices.nunique())))
        distribution = bins.value_counts().sort_index()
        distribution.index = distribution.index.astype(str)
        st.bar_chart(distribution, color="#2DD4BF")
        p1, p2, p3 = st.columns(3)
        p1.metric("Lowest Sample Price", money(prices.min()))
        p2.metric("Highest Sample Price", money(prices.max()))
        p3.metric("Sample Price Range", money(prices.max() - prices.min()))
    else:
        st.info("No valid nightly-price values match the selected filters.")

    # ========================================================
    # PRICING INTELLIGENCE
    # ========================================================

    st.divider()

    section_heading(
        "Pricing Intelligence",
        "Analyze sample price positioning, accommodation segments, "
        "neighbourhood competitiveness and pricing variation.",
    )

    if not price_observations.empty:

        pricing = price_observations.copy()

        # Prepare numeric and date fields.
        pricing["base_price_npr"] = numeric(
            pricing["base_price_npr"]
        )

        pricing["stay_date"] = pd.to_datetime(
            pricing["stay_date"], errors="coerce", utc=True
        )

        pricing["observation_date"] = pd.to_datetime(
            pricing["observation_date"], errors="coerce", utc=True
        )

        # Apply executive filters only when selections exist.
        if selected_neighbourhoods:
            pricing = pricing[
                pricing["neighbourhood"].isin(selected_neighbourhoods)
            ]
        if selected_types:
            pricing = pricing[
                pricing["property_type"].isin(selected_types)
            ]

        pricing = pricing.dropna(
            subset=["base_price_npr"]
        )

        if not pricing.empty:

            # ------------------------------------------------
            # PRICING KPI SNAPSHOT
            # ------------------------------------------------

            unique_properties = pricing["property_id"].nunique()

            avg_observed = pricing["base_price_npr"].mean()
            median_observed = pricing["base_price_npr"].median()

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "Price Observations",
                f"{len(pricing):,}",
            )

            p2.metric(
                "Unique Properties",
                f"{unique_properties:,}",
            )

            p3.metric(
                "Average Observed Price",
                money(avg_observed),
            )

            p4.metric(
                "Median Observed Price",
                money(median_observed),
            )

            st.caption(
                "All figures represent synthetic sample price observations. "
                "They are not verified live market prices."
            )

            # ------------------------------------------------
            # ACCOMMODATION PRICE SEGMENTATION
            # ------------------------------------------------

            section_heading(
                "Accommodation Price Segmentation",
                "Classify observed prices into four relative sample-price groups.",
            )

            if len(pricing) >= 4:
                pricing["Price Segment"] = pd.qcut(
                    pricing["base_price_npr"].rank(method="first"),
                    q=4,
                    labels=[
                        "Budget",
                        "Mid-range",
                        "Premium",
                        "Luxury",
                    ],
                )

                segment_summary = (
                    pricing.groupby(
                        "Price Segment",
                        observed=False,
                    )
                    .agg(
                        Properties=("property_id", "nunique"),
                        Observations=("base_price_npr", "count"),
                        Average_Price=("base_price_npr", "mean"),
                        Median_Price=("base_price_npr", "median"),
                    )
                )

                segment_columns = st.columns(4)

                for column, segment_name in zip(
                    segment_columns,
                    ["Budget", "Mid-range", "Premium", "Luxury"],
                ):
                    segment = segment_summary.loc[segment_name]
                    column.metric(
                        segment_name,
                        money(segment["Median_Price"]),
                        f"{int(segment['Observations']):,} observations",
                    )
                    column.caption(
                        f"{int(segment['Properties']):,} unique properties"
                    )

                left, right = st.columns(2)

                with left:
                    section_heading(
                        "Price Segment Distribution",
                        "Compare observations across the four price groups.",
                    )
                    st.bar_chart(
                        segment_summary[["Observations"]],
                        color="#55C7F3",
                        alt="Price observation counts by sample-price segment",
                    )

                with right:
                    section_heading(
                        "Segment Price Comparison",
                        "Compare average and median prices.",
                    )
                    st.bar_chart(
                        segment_summary[
                            ["Average_Price", "Median_Price"]
                        ],
                        color=["#818CF8", "#42D6C3"],
                        alt="Average and median prices by sample-price segment",
                    )

                st.caption(
                    "Segments are based on quartiles of the filtered sample. "
                    "They are not official industry price categories."
                )
            else:
                st.info(
                    "At least four price observations are required to create "
                    "four sample-price segments."
                )

            st.divider()

            # ------------------------------------------------
            # NEIGHBOURHOOD PRICE COMPARISON
            # ------------------------------------------------

            left, right = st.columns([1.25, 1])

            with left:

                section_heading(
                    "Neighbourhood Price Comparison",
                    "Compare average and median recorded base prices.",
                )

                if "neighbourhood" in pricing.columns:

                    area_prices = (
                        pricing.groupby("neighbourhood")
                        .agg(
                            Average_Price=("base_price_npr", "mean"),
                            Median_Price=("base_price_npr", "median"),
                            Observations=("base_price_npr", "count"),
                        )
                        .sort_values(
                            "Median_Price", ascending=False
                        )
                        .head(10)
                    )

                    if not area_prices.empty:

                        st.bar_chart(
                            area_prices[
                                ["Average_Price", "Median_Price"]
                            ],
                            color=["#55C7F3", "#42D6C3"],
                            alt="Average and median observed prices by neighbourhood",
                        )

                        st.caption(
                            "Top 10 neighbourhoods by median observed price."
                        )

            # ------------------------------------------------
            # PROPERTY TYPE PRICE COMPARISON
            # ------------------------------------------------

            with right:

                section_heading(
                    "Property-Type Pricing",
                    "Compare observed prices across accommodation types.",
                )

                if "property_type" in pricing.columns:

                    type_prices = (
                        pricing.groupby("property_type")[
                            "base_price_npr"
                        ]
                        .agg(["mean", "median", "count"])
                        .sort_values("mean", ascending=False)
                    )

                    if not type_prices.empty:

                        st.bar_chart(
                            type_prices[["mean", "median"]],
                            color=["#818CF8", "#F59E0B"],
                            alt="Average and median observed prices by property type",
                        )

                        st.caption(
                            "Mean and median base prices by property type."
                        )

            # ------------------------------------------------
            # PRICE VOLATILITY ANALYSIS
            # ------------------------------------------------

            st.divider()

            section_heading(
                "Price Volatility Analysis",
                "Measure relative price variation across neighbourhoods.",
            )

            volatility = (
                pricing.groupby("neighbourhood")["base_price_npr"]
                .agg(
                    Average_Price="mean",
                    Price_Std_Dev="std",
                    Observations="count",
                )
            )

            volatility = volatility[
                volatility["Observations"] >= 3
            ].copy()

            volatility["Price_CV_Percent"] = (
                volatility["Price_Std_Dev"]
                / volatility["Average_Price"].where(
                    volatility["Average_Price"].ne(0)
                )
                * 100
            )

            volatility = volatility.dropna(
                subset=["Price_CV_Percent"]
            )

            if not volatility.empty:

                left, right = st.columns(2)

                with left:

                    section_heading(
                        "Highest Price Variation",
                        "Areas with the greatest relative price dispersion.",
                    )

                    st.bar_chart(
                        volatility.sort_values(
                            "Price_CV_Percent",
                            ascending=False,
                        )[["Price_CV_Percent"]].head(10),
                        color="#F97316",
                        alt="Neighbourhoods with highest relative price variation",
                    )

                with right:

                    section_heading(
                        "Most Consistent Sample Prices",
                        "Areas with the lowest relative price dispersion.",
                    )

                    st.bar_chart(
                        volatility.sort_values(
                            "Price_CV_Percent",
                            ascending=True,
                        )[["Price_CV_Percent"]].head(10),
                        color="#2DD4BF",
                        alt="Neighbourhoods with lowest relative price variation",
                    )

                st.caption(
                    "CV = standard deviation / average price × 100. "
                    "This measures variation across observations, "
                    "not price movement over time."
                )

            else:

                st.info(
                    "Insufficient neighbourhood observations "
                    "for price variation analysis."
                )

            # ------------------------------------------------
            # EXECUTIVE PRICING INSIGHTS
            # ------------------------------------------------

            st.divider()

            section_heading(
                "Executive Pricing Insights",
                "Key observations generated from the selected pricing sample.",
            )

            insight1, insight2, insight3 = st.columns(3)

            lowest_price = pricing["base_price_npr"].min()
            highest_price = pricing["base_price_npr"].max()
            price_spread = (
                f"{highest_price / lowest_price:.1f}x spread"
                if lowest_price > 0
                else None
            )

            insight1.metric(
                "Observed Price Range",
                money(highest_price - lowest_price),
                price_spread,
            )

            insight1.caption(
                "Difference between the highest and lowest sample prices."
            )

            eligible_areas = (
                pricing.groupby("neighbourhood")["base_price_npr"]
                .agg(["median", "count"])
            )

            eligible_areas = eligible_areas[
                eligible_areas["count"] >= 3
            ].sort_values("median", ascending=False)

            if not eligible_areas.empty:

                insight2.metric(
                    "Highest Median Price Area",
                    str(eligible_areas.index[0]),
                    money(eligible_areas.iloc[0]["median"]),
                )

                insight2.caption(
                    f"{int(eligible_areas.iloc[0]['count'])} observations."
                )

            else:

                insight2.metric("Highest Median Price Area", "N/A")

            if not volatility.empty:

                highest_variation = volatility.sort_values(
                    "Price_CV_Percent",
                    ascending=False,
                ).iloc[0]

                insight3.metric(
                    "Highest Price Variation",
                    str(highest_variation.name),
                    f"{highest_variation['Price_CV_Percent']:.1f}% CV",
                )

                insight3.caption(
                    "Based on relative price dispersion."
                )

            else:

                insight3.metric("Highest Price Variation", "N/A")

            st.info(
                "Business interpretation: Price segmentation helps "
                "organize accommodation offerings. Neighbourhood "
                "comparisons support relative positioning, while "
                "price variation indicates differences within the sample. "
                "These findings do not establish profitability, booking "
                "demand, occupancy or future price movements."
            )

            st.divider()

            # ------------------------------------------------
            # STAY MONTH PRICE ANALYSIS
            # ------------------------------------------------

            section_heading(
                "Stay-Date Pricing Analysis",
                "Understand how observed base prices differ by intended stay month.",
            )

            monthly_pricing = pricing.dropna(
                subset=["stay_date"]
            ).copy()

            if not monthly_pricing.empty:

                monthly_pricing["Stay Month"] = (
                    monthly_pricing["stay_date"]
                    .dt.strftime("%b %Y")
                )

                monthly_summary = (
                    monthly_pricing.groupby("Stay Month")
                    .agg(
                        Average_Price=("base_price_npr", "mean"),
                        Median_Price=("base_price_npr", "median"),
                        Observations=("base_price_npr", "count"),
                    )
                )

                st.line_chart(
                    monthly_summary[
                        ["Average_Price", "Median_Price"]
                    ],
                    color=["#55C7F3", "#42D6C3"],
                    alt="Average and median observed prices by stay month",
                )

                st.dataframe(
                    monthly_summary.reset_index(),
                    width="stretch",
                    hide_index=True,
                    alt="Monthly pricing summary",
                )

            else:
                st.info(
                    "No valid stay dates are available for monthly analysis."
                )

            st.divider()

            # ------------------------------------------------
            # PRICE OBSERVATION CAPTURE TIMELINE
            # ------------------------------------------------

            section_heading(
                "Observation Capture Timeline",
                "Number of price records captured during each observation date.",
            )

            capture_data = pricing.dropna(
                subset=["observation_date"]
            ).copy()

            if not capture_data.empty:

                capture_data["Capture Date"] = (
                    capture_data["observation_date"]
                    .dt.strftime("%Y-%m-%d")
                )

                capture_trend = (
                    capture_data.groupby("Capture Date")
                    .size()
                    .rename("Observations")
                )

                st.bar_chart(
                    capture_trend,
                    color="#818CF8",
                    alt="Price observation counts by capture date",
                )

            else:
                st.info(
                    "No valid observation dates are available."
                )

            # ------------------------------------------------
            # DETAILED PRICING TABLE
            # ------------------------------------------------

            with st.expander("Explore Detailed Pricing Records"):

                display_columns = [
                    column for column in [
                        "property_id",
                        "neighbourhood",
                        "property_type",
                        "room_type",
                        "stay_date",
                        "base_price_npr",
                        "currency_original",
                        "source_name",
                    ]
                    if column in pricing.columns
                ]

                st.dataframe(
                    pricing[display_columns].sort_values(
                        "base_price_npr",
                        ascending=False,
                    ),
                    width="stretch",
                    hide_index=True,
                    alt="Detailed filtered pricing records",
                )

                st.download_button(
                    "Download Pricing Records CSV",
                    pricing.to_csv(index=False).encode("utf-8"),
                    file_name="pricing_intelligence.csv",
                    mime="text/csv",
                )

        else:
            st.info(
                "No pricing records match the selected executive filters."
            )

    else:
        st.info(
            "Price observation dataset is not available."
        )

    st.divider()
    section_heading(
        "Tourism demand context",
        "Nepal-wide visitor arrivals provide macro context, not a direct measure of Kathmandu accommodation demand.",
    )
    if not tourist_arrivals.empty and "tourist_arrivals" in tourist_arrivals:
        tourism = tourist_arrivals.copy()
        if "month_date" in tourism:
            tourism["month_date"] = pd.to_datetime(tourism["month_date"], errors="coerce")
        elif {"year", "month"}.issubset(tourism.columns):
            tourism["month_date"] = pd.to_datetime(
                dict(
                    year=numeric(tourism["year"]),
                    month=numeric(tourism["month"]),
                    day=1,
                ),
                errors="coerce",
            )
        tourism["tourist_arrivals"] = numeric(tourism["tourist_arrivals"])
        tourism = tourism.dropna(subset=["month_date", "tourist_arrivals"]).sort_values("month_date")
        if not tourism.empty:
            latest = tourism.iloc[-1]
            t1, t2 = st.columns([1, 2])
            t1.metric("Latest Monthly Arrivals", f"{int(latest['tourist_arrivals']):,}")
            t1.caption(f"Nepal · {latest['month_date'].strftime('%B %Y')}")
            t2.line_chart(
                tourism.set_index("month_date")[["tourist_arrivals"]].tail(36),
                color="#38BDF8",
            )
            if "source_name" in tourism and tourism["source_name"].notna().any():
                st.caption(f"Source: {tourism['source_name'].dropna().iloc[0]}")
            if "is_provisional" in tourism:
                provisional = str(latest.get("is_provisional", "")).lower()
                if provisional in ("true", "1", "yes"):
                    st.caption("The latest tourism observation is marked provisional.")
        else:
            st.info("The tourism file is present but has no usable monthly records.")
    else:
        st.info("Tourist-arrival data was not found in the configured project data folders.")

    st.divider()
    section_heading(
        "Availability and occupancy data coverage",
        "Availability is not the same as occupancy. Occupancy is reported only when valid occupancy observations exist.",
    )
    if not availability.empty:
        status_col = next((c for c in ["available", "is_available"] if c in availability.columns), None)
        if status_col:
            statuses = availability[status_col].astype(str).str.strip().str.lower()
            available_n = statuses.isin(["true", "1", "yes", "available"]).sum()
            unavailable_n = statuses.isin(["false", "0", "no", "unavailable"]).sum()
            status_total = available_n + unavailable_n
            if status_total:
                a1, a2, a3 = st.columns(3)
                a1.metric("Availability Records", f"{status_total:,}")
                a2.metric("Available Share", f"{available_n / status_total * 100:.1f}%")
                a3.metric("Unavailable Share", f"{unavailable_n / status_total * 100:.1f}%")
                st.caption("These are recorded availability statuses, not confirmed occupancy or completed bookings.")
            else:
                st.info("Availability records exist, but status values are not interpretable.")
        else:
            st.info("No recognized availability-status column was found.")
    else:
        st.info("No processed availability records are currently loaded.")

    if not occupancy_context.empty and "occupancy_rate" in occupancy_context:
        occupancy_values = numeric(occupancy_context["occupancy_rate"]).dropna()
        if not occupancy_values.empty:
            rate = occupancy_values.mean()
            if occupancy_values.max() <= 1:
                rate *= 100
            st.metric("Mean Source Occupancy Rate", f"{rate:.1f}%")
        else:
            st.caption("Occupancy file exists but contains no valid occupancy-rate observations.")
    else:
        st.caption(
            "No usable monthly occupancy observations are loaded. "
            "Occupancy is not inferred from availability."
        )

    st.divider()
    section_heading(
        "Featured sample properties",
        "Highest-rated records within the selected property sample.",
    )
    if not executive_properties.empty and "rating" in executive_properties:
        featured = executive_properties.copy()
        featured["rating"] = numeric(featured["rating"])
        featured = featured.sort_values("rating", ascending=False, na_position="last").head(8)
        columns = [
            c for c in [
                "property_id", "property_name", "neighbourhood", "property_type",
                "room_type", "nightly_price_npr", "rating",
            ] if c in featured.columns
        ]
        st.dataframe(featured[columns], width="stretch", hide_index=True)
    else:
        st.info("No featured property records match the selected filters.")


# 7. PROPERTY RECOMMENDATIONS
# ============================================================

elif page == "Property Recommendations":

    page_header(
        "DISCOVERY ENGINE",
        "Find Your Next Stay",
        "Discover accommodation options with similar characteristics "
        "using TF-IDF text features and cosine similarity.",
    )

    names = (
        recommendation_properties["property_id"].astype(str)
        + " — "
        + recommendation_properties["property_name"].astype(str)
    ).tolist()

    with st.container(border=True):
        selected_name = st.selectbox(
            "Select a reference property",
            names,
        )

        selected_index = names.index(selected_name)
        selected = recommendation_properties.iloc[selected_index]

        st.markdown("#### Selected Property")

        a, b, c, d = st.columns(4)

        a.metric("Nightly Price", money(selected["nightly_price_npr"]))
        b.metric("Rating", f"{selected['rating']} / 5")
        c.metric("Reviews", f"{int(selected['review_count']):,}")
        d.metric("Capacity", f"{int(selected['capacity'])} guests")

        st.caption(
            f"📍 {selected['neighbourhood']}  ·  "
            f"{selected['property_type']}  ·  "
            f"{selected['room_type']}"
        )

        st.write(selected["description"])

    st.markdown("#### Refine Recommendations")

    f1, f2, f3 = st.columns(3)

    with f1:
        count = st.slider(
            "Number of recommendations",
            3, 10, 5,
        )

    with f2:
        max_budget = st.number_input(
            "Maximum nightly budget (NPR)",
            min_value=0,
            value=0,
            step=500,
            help="Enter 0 to show all price ranges.",
        )

    with f3:
        min_rating = st.slider(
            "Minimum rating",
            0.0, 5.0, 0.0, 0.5,
        )

    recommendations = get_recommendations(
        selected_index,
        number=len(recommendation_properties) - 1,
    )

    if max_budget > 0:
        recommendations = recommendations[
            recommendations["Nightly Price (NPR)"] <= max_budget
        ]

    recommendations = recommendations[
        recommendations["Rating"] >= min_rating
    ].head(count)

    st.divider()
    section_heading("Recommended Properties")

    if recommendations.empty:
        st.info("No properties match the selected filters.")
    else:
        for _, item in recommendations.iterrows():
            st.markdown(
                f"""
                <div class="property-card">
                    <div class="property-card-title">
                        {item['Property Name']}
                    </div>
                    <div class="property-card-meta">
                        {item['Property ID']} ·
                        {item['Neighbourhood']} ·
                        {item['Property Type']} ·
                        {item['Room Type']}
                    </div>
                    <br>
                    <b>{money(item['Nightly Price (NPR)'])}</b>
                    &nbsp; · &nbsp; ⭐ {item['Rating']} / 5
                    &nbsp; · &nbsp;
                    <span class="similarity">
                        {item['Similarity (%)']}% similarity
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "Similarity is a mathematical text-feature similarity score. "
        "It is not a booking probability or a guarantee of user preference."
    )


# ============================================================
# 8. PRICE PREDICTION
# ============================================================

elif page == "Price Prediction":

    page_header(
        "PREDICTIVE ANALYTICS",
        "Nightly Price Estimator",
        "Estimate an experimental nightly price using the trained "
        "Gradient Boosting regression pipeline.",
    )

    st.info(
        "Important: The model uses the selected property's existing "
        "baseline nightly price as an input feature. This is a "
        "baseline-adjustment experiment, not independent price discovery."
    )

    choices = published_data(properties)

    labels = (
        choices["property_id"].astype(str)
        + " — "
        + choices["property_name"].astype(str)
    ).tolist()

    with st.container(border=True):

        st.markdown("#### Property and Stay Details")

        selected_label = st.selectbox(
            "Choose a property",
            labels,
        )

        selected = choices.iloc[labels.index(selected_label)]

        st.metric(
            "Current Baseline Nightly Price",
            money(selected["nightly_price_npr"]),
        )

        col1, col2 = st.columns(2)

        with col1:
            lead_time = st.number_input(
                "Booking lead time (days)",
                min_value=0,
                max_value=730,
                value=30,
                step=1,
            )

        with col2:
            stay_date = st.date_input(
                "Planned check-in date",
                value=date.today(),
                min_value=date.today(),
            )

        predict = st.button(
            "Generate Price Estimate",
            type="primary",
            width="stretch",
        )

    if predict:

        input_data = pd.DataFrame([{
            "city": selected["city"],
            "neighbourhood": selected["neighbourhood"],
            "property_type": selected["property_type"],
            "room_type": selected["room_type"],
            "capacity": selected["capacity"],
            "bedrooms": selected["bedrooms"],
            "bathrooms": selected["bathrooms"],
            "lead_time_days": int(lead_time),
            "stay_month": stay_date.month,
            "stay_day": stay_date.day,
            "stay_day_of_week": stay_date.weekday(),
            "nightly_price_npr": selected["nightly_price_npr"],
            "amenities": selected["amenities"],
        }])

        try:
            prediction = float(price_model.predict(input_data)[0])
            baseline = float(selected["nightly_price_npr"])

            if not np.isfinite(prediction):
                raise ValueError("The model returned an invalid value.")

            difference = prediction - baseline
            percentage = (
                difference / baseline * 100 if baseline else 0
            )

            st.divider()
            section_heading("Prediction Results")

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Estimated Nightly Price",
                money(prediction),
            )

            p2.metric(
                "Baseline Price",
                money(baseline),
            )

            p3.metric(
                "Difference from Baseline",
                money(difference),
                delta=f"{percentage:+.2f}%",
            )

            comparison = pd.DataFrame({
                "Price Type": ["Baseline", "Model Estimate"],
                "Nightly Price (NPR)": [baseline, prediction],
            }).set_index("Price Type")

            st.bar_chart(
                comparison,
                color="#38BDF8",
            )

            st.success("Prediction generated successfully.")

            st.caption(
                "The estimate is experimental and is based on synthetic "
                "sample observations. It is not a live market quotation."
            )

            with st.expander("View model input features"):
                st.dataframe(
                    input_data,
                    width="stretch",
                    hide_index=True,
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ============================================================
# 9. DATA ANALYSIS
# ============================================================

elif page == "Data Analysis":

    page_header(
        "DATA EXPLORATION",
        "Understand the Dataset",
        "Explore property characteristics, availability records "
        "and nightly price observations.",
    )

    tab1, tab2, tab3 = st.tabs([
        "Property Dataset",
        "Availability",
        "Price Observations",
    ])

    with tab1:

        section_heading("Property Explorer")

        filtered = properties.copy()

        f1, f2, f3 = st.columns(3)

        with f1:
            neighbourhoods = sorted(
                filtered["neighbourhood"].dropna().unique()
            )
            selected_neighbourhoods = st.multiselect(
                "Neighbourhood",
                neighbourhoods,
                default=neighbourhoods,
            )

        with f2:
            property_types = sorted(
                filtered["property_type"].dropna().unique()
            )
            selected_types = st.multiselect(
                "Property Type",
                property_types,
                default=property_types,
            )

        with f3:
            room_types = sorted(
                filtered["room_type"].dropna().unique()
            )
            selected_rooms = st.multiselect(
                "Room Type",
                room_types,
                default=room_types,
            )

        filtered = filtered[
            filtered["neighbourhood"].isin(selected_neighbourhoods)
            & filtered["property_type"].isin(selected_types)
            & filtered["room_type"].isin(selected_rooms)
        ]

        m1, m2, m3 = st.columns(3)

        m1.metric("Filtered Properties", f"{len(filtered):,}")

        m2.metric(
            "Average Price",
            money(numeric(filtered["nightly_price_npr"]).mean()),
        )

        m3.metric(
            "Average Rating",
            f"{numeric(filtered['rating']).mean():.2f} / 5",
        )

        left, right = st.columns(2)

        with left:
            section_heading("Neighbourhood Distribution")
            st.bar_chart(
                filtered["neighbourhood"].value_counts(),
                color="#38BDF8",
            )

        with right:
            section_heading("Price by Property Type")
            price_by_type = (
                filtered.assign(
                    nightly_price_npr=numeric(
                        filtered["nightly_price_npr"]
                    )
                )
                .groupby("property_type")["nightly_price_npr"]
                .mean()
            )
            st.bar_chart(price_by_type, color="#2DD4BF")

        section_heading("Filtered Records")
        st.dataframe(
            filtered,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download Filtered Properties CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_properties.csv",
            mime="text/csv",
        )

    with tab2:

        section_heading("Availability Records")

        if availability.empty:
            st.info("Availability dataset is not available.")
        else:
            st.metric(
                "Total Availability Records",
                f"{len(availability):,}",
            )

            if "available" in availability.columns:
                st.bar_chart(
                    availability["available"].astype(str).value_counts(),
                    color="#38BDF8",
                )

            st.dataframe(
                availability.head(500),
                width="stretch",
                hide_index=True,
            )

            st.caption("Preview limited to the first 500 records.")

    with tab3:

        section_heading("Price Observation Records")

        if price_observations.empty:
            st.info("Price observation dataset is not available.")
        else:
            st.metric(
                "Total Price Observations",
                f"{len(price_observations):,}",
            )

            if "observed_price_npr" in price_observations.columns:
                st.bar_chart(
                    numeric(
                        price_observations["observed_price_npr"]
                    ).dropna(),
                    color="#2DD4BF",
                )

            st.dataframe(
                price_observations.head(500),
                width="stretch",
                hide_index=True,
            )

            st.caption("Preview limited to the first 500 records.")


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    page_header(
        "MODEL PERFORMANCE",
        "Advanced Model Evaluation",
        "Compare prediction models, evaluate cross-validation stability, "
        "and investigate prediction errors.",
    )

    # --------------------------------------------------------
    # PREPARE EVALUATION DATA
    # --------------------------------------------------------

    comparison_df = price_comparison.copy()
    cv_df = cross_validation.copy()
    error_df = error_categories.copy()
    metrics_df = metrics.copy()

    for df in [comparison_df, cv_df, error_df]:
        for col in df.columns:
            if col not in ["model", "price_category"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------------
    # 1. OVERALL MODEL PERFORMANCE
    # --------------------------------------------------------

    section_heading("Overall Model Performance")

    if not metrics_df.empty and {"metric", "value"}.issubset(metrics_df.columns):

        metric_values = {
            str(row["metric"]).strip().lower(): pd.to_numeric(
                row["value"], errors="coerce"
            )
            for _, row in metrics_df.iterrows()
        }

        def get_metric(*names):
            for name in names:
                value = metric_values.get(name.lower())
                if value is not None and pd.notna(value):
                    return float(value)
            return None

        mae = get_metric("mae")
        rmse = get_metric("rmse")
        r2 = get_metric("r2", "r2_score", "r2 score")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Mean Absolute Error",
            money(mae) if mae is not None else "N/A",
            help="Average absolute difference between actual and predicted prices.",
        )

        c2.metric(
            "Root Mean Squared Error",
            money(rmse) if rmse is not None else "N/A",
            help="Penalizes larger prediction errors more heavily.",
        )

        c3.metric(
            "R² Score",
            f"{r2:.4f}" if r2 is not None else "N/A",
            help="Measures how much variation is explained by the model.",
        )

    else:
        st.info("Model performance metrics are unavailable.")

    st.caption(
        "These are saved test-set metrics. Their interpretation depends "
        "on the dataset and evaluation methodology."
    )

    st.divider()

    # --------------------------------------------------------
    # 2. MODEL COMPARISON
    # --------------------------------------------------------

    section_heading("Model Comparison")

    if not comparison_df.empty and "model" in comparison_df.columns:

        st.dataframe(
            comparison_df,
            width="stretch",
            hide_index=True,
        )

        if "MAE" in comparison_df.columns:
            valid_mae = comparison_df.dropna(subset=["MAE"])

            if not valid_mae.empty:
                best_model = valid_mae.loc[valid_mae["MAE"].idxmin()]

                st.success(
                    f"Lowest MAE: {best_model['model']} "
                    f"({money(best_model['MAE'])})"
                )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### MAE Comparison")

            if "MAE" in comparison_df.columns:
                st.bar_chart(
                    comparison_df.set_index("model")[["MAE"]]
                )

        with chart_col2:
            st.markdown("#### RMSE Comparison")

            if "RMSE" in comparison_df.columns:
                st.bar_chart(
                    comparison_df.set_index("model")[["RMSE"]]
                )

        if "R2" in comparison_df.columns:
            st.markdown("#### R² Score Comparison")

            st.bar_chart(
                comparison_df.set_index("model")[["R2"]]
            )

    else:
        st.info("Model comparison report is unavailable.")

    st.divider()

    # --------------------------------------------------------
    # 3. CROSS-VALIDATION ANALYSIS
    # --------------------------------------------------------

    section_heading("Cross-Validation Stability")

    if not cv_df.empty:

        st.markdown(
            "Evaluate how consistently the model performs across "
            "different validation folds."
        )

        st.dataframe(
            cv_df,
            width="stretch",
            hide_index=True,
        )

        required_cv = ["GB_MAE", "GB_RMSE", "GB_R2"]

        if all(col in cv_df.columns for col in required_cv):

            cv_mae_mean = cv_df["GB_MAE"].mean()
            cv_rmse_mean = cv_df["GB_RMSE"].mean()
            cv_r2_mean = cv_df["GB_R2"].mean()

            cv_c1, cv_c2, cv_c3 = st.columns(3)

            cv_c1.metric(
                "Average CV MAE",
                money(cv_mae_mean),
            )

            cv_c2.metric(
                "Average CV RMSE",
                money(cv_rmse_mean),
            )

            cv_c3.metric(
                "Average CV R²",
                f"{cv_r2_mean:.4f}",
            )

            st.markdown("#### Gradient Boosting: Fold Performance")

            chart1, chart2 = st.columns(2)

            with chart1:
                st.markdown("**MAE Across Folds**")
                st.line_chart(cv_df.set_index("fold")[["GB_MAE"]])

            with chart2:
                st.markdown("**RMSE Across Folds**")
                st.line_chart(cv_df.set_index("fold")[["GB_RMSE"]])

            st.markdown("**R² Across Folds**")
            st.line_chart(cv_df.set_index("fold")[["GB_R2"]])

            stability = cv_df["GB_MAE"].std()

            if pd.notna(stability):
                st.caption(
                    f"MAE standard deviation across folds: "
                    f"{money(stability)}. Lower variation indicates "
                    "more consistent fold performance."
                )

        baseline_cols = [
            "Baseline_MAE",
            "Baseline_RMSE",
            "Baseline_R2",
        ]

        if all(col in cv_df.columns for col in baseline_cols):

            st.markdown("#### Gradient Boosting vs Baseline")

            baseline_comparison = pd.DataFrame({
                "Gradient Boosting MAE": cv_df["GB_MAE"],
                "Baseline MAE": cv_df["Baseline_MAE"],
            })

            if "fold" in cv_df.columns:
                baseline_comparison.index = cv_df["fold"]
                baseline_comparison.index.name = "Fold"

            st.line_chart(baseline_comparison)

    else:
        st.info("Cross-validation report is unavailable.")

    st.divider()

    # --------------------------------------------------------
    # 4. PREDICTION ERROR ANALYSIS
    # --------------------------------------------------------

    section_heading("Prediction Error Analysis")

    if not error_df.empty:

        st.dataframe(
            error_df,
            width="stretch",
            hide_index=True,
        )

        error_col1, error_col2 = st.columns(2)

        with error_col1:
            st.markdown("#### Mean Absolute Error by Category")

            if "mean_absolute_error" in error_df.columns:
                st.bar_chart(
                    error_df.set_index("price_category")[
                        ["mean_absolute_error"]
                    ]
                )

        with error_col2:
            st.markdown("#### Mean Percentage Error by Category")

            if "mean_percentage_error" in error_df.columns:
                st.bar_chart(
                    error_df.set_index("price_category")[
                        ["mean_percentage_error"]
                    ]
                )

        # Actual vs predicted category averages
        actual_predicted_cols = [
            "average_actual_price",
            "average_predicted_price",
        ]

        if all(col in error_df.columns for col in actual_predicted_cols):

            st.markdown("#### Actual vs Predicted Average Price")

            comparison_chart = error_df.set_index("price_category")[
                actual_predicted_cols
            ]

            st.bar_chart(comparison_chart)

        # Signed error analysis
        signed_error_col = next(
            (
                col for col in error_df.columns
                if col in [
                    "average_signed_error",
                    "average_signed_error ",
                ]
            ),
            None,
        )

        if signed_error_col:
            st.markdown("#### Prediction Bias")

            st.bar_chart(
                error_df.set_index("price_category")[
                    [signed_error_col]
                ]
            )

            st.caption(
                "Positive signed error indicates average overprediction; "
                "negative signed error indicates average underprediction."
            )

    else:
        st.info("Prediction error category report is unavailable.")

    st.divider()


    # --------------------------------------------------------
    # PREDICTION ERROR KEY TAKEAWAYS
    # --------------------------------------------------------

    if not error_df.empty:

        section_heading("Key Takeaways")

        insight_cols = st.columns(2)

        # Highest percentage error
        if "mean_percentage_error" in error_df.columns:

            percentage_data = error_df.dropna(
                subset=["mean_percentage_error"]
            )

            if not percentage_data.empty:

                highest_percentage = percentage_data.loc[
                    percentage_data["mean_percentage_error"].idxmax()
                ]

                with insight_cols[0]:
                    st.metric(
                        "Highest Percentage Error",
                        f"{highest_percentage['mean_percentage_error']:.2f}%",
                        help="Price category with the highest mean percentage error.",
                    )

                    st.caption(
                        f"Category: {highest_percentage['price_category']}"
                    )

        # Highest absolute error
        if "mean_absolute_error" in error_df.columns:

            absolute_data = error_df.dropna(
                subset=["mean_absolute_error"]
            )

            if not absolute_data.empty:

                highest_absolute = absolute_data.loc[
                    absolute_data["mean_absolute_error"].idxmax()
                ]

                with insight_cols[1]:
                    st.metric(
                        "Highest Absolute Error",
                        money(highest_absolute["mean_absolute_error"]),
                    )

                    st.caption(
                        f"Category: {highest_absolute['price_category']}"
                    )

        # Prediction bias
        if "average_signed_error" in error_df.columns:

            st.markdown("#### Prediction Bias by Category")

            for _, row in error_df.iterrows():

                signed_error = row["average_signed_error"]

                if pd.isna(signed_error):
                    continue

                if signed_error > 0:
                    interpretation = "Overprediction"
                elif signed_error < 0:
                    interpretation = "Underprediction"
                else:
                    interpretation = "Approximately balanced"

                st.write(
                    f"**{row['price_category']}**: "
                    f"{interpretation} "
                    f"({money(abs(signed_error))} average signed error)"
                )

        st.info(
            "The model shows different error patterns across price "
            "categories. These results can help identify where further "
            "model improvements are needed. They are based on synthetic "
            "sample data and should not be treated as verified market "
            "performance."
        )


    # --------------------------------------------------------
    # 5. EVALUATION INTERPRETATION
    # --------------------------------------------------------

    section_heading("Model Evaluation Insights")

    if not comparison_df.empty and "MAE" in comparison_df.columns:

        valid_models = comparison_df.dropna(subset=["MAE"])

        if not valid_models.empty:

            best = valid_models.loc[valid_models["MAE"].idxmin()]

            st.markdown(
                f"""
                - **Best model by MAE:** {best['model']}
                - **MAE:** {money(best['MAE'])}
                - **Models evaluated:** {len(valid_models)}
                """
            )

    if not cv_df.empty and "GB_MAE" in cv_df.columns:

        average_cv_mae = cv_df["GB_MAE"].mean()

        st.markdown(
            f"- **Cross-validation average MAE:** {money(average_cv_mae)}"
        )

    st.warning(
        "Important limitation: The current price prediction experiment "
        "uses the existing baseline nightly price as an input. Strong "
        "evaluation scores should not be interpreted as independent "
        "market-price discovery. Property and price observations are "
        "synthetic sample data, not verified live Airbnb market data."
    )



# ============================================================
# 11. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AIRBNB INTELLIGENCE<br>
        Machine Learning Internship Project · Experimental Dataset
    </div>
    """,
    unsafe_allow_html=True,
)
