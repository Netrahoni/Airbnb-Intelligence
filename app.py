
from pathlib import Path
from datetime import date
from html import escape
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def _file_cache_key(path):
    """Return a lightweight file signature so cached artifacts refresh after replacement."""
    try:
        stat = Path(path).stat()
        return (stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        return None


def _optional_file_cache_keys(filenames):
    """Capture signatures for each optional-data search location."""
    search_dirs = [DATA_DIR, ROOT / "data" / "raw", ROOT]
    return tuple(
        (str(folder / filename), _file_cache_key(folder / filename))
        for filename in filenames
        for folder in search_dirs
    )


@st.cache_resource(
    max_entries=1,
    show_spinner="Loading recommendation and pricing models…",
)
def load_models(recommender_signature=None, price_signature=None):
    recommender = joblib.load(
        MODEL_DIR / "property_recommender.joblib"
    )
    price_predictor = joblib.load(
        MODEL_DIR / "price_predictor.joblib"
    )
    return recommender, price_predictor


@st.cache_data(ttl=1800, max_entries=32, show_spinner=False)
def load_csv(filename, file_signature=None):
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=1800, max_entries=32, show_spinner=False)
def load_optional_csv(*filenames, file_signatures=None):
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


@st.cache_data(ttl=1800, max_entries=32, show_spinner=False)
def load_report(filename, file_signature=None):
    path = REPORT_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


try:
    recommender, price_model = load_models(
        _file_cache_key(MODEL_DIR / "property_recommender.joblib"),
        _file_cache_key(MODEL_DIR / "price_predictor.joblib"),
    )

    properties = load_csv(
        "properties_clean.csv",
        _file_cache_key(DATA_DIR / "properties_clean.csv"),
    )
    availability = load_csv(
        "availability_clean.csv",
        _file_cache_key(DATA_DIR / "availability_clean.csv"),
    )
    price_observations = load_csv(
        "price_observations_clean.csv",
        _file_cache_key(DATA_DIR / "price_observations_clean.csv"),
    )

    # Optional context datasets for the executive dashboard.
    tourist_arrival_files = (
        "nepal_monthly_tourist_arrivals_clean.csv",
        "nepal_monthly_tourist_arrivals.csv",
    )
    tourist_arrivals = load_optional_csv(
        *tourist_arrival_files,
        file_signatures=_optional_file_cache_keys(tourist_arrival_files),
    )
    occupancy_files = (
        "kathmandu_monthly_occupancy_clean.csv",
        "kathmandu_monthly_occupancy.csv",
    )
    occupancy_context = load_optional_csv(
        *occupancy_files,
        file_signatures=_optional_file_cache_keys(occupancy_files),
    )

    recommendation_properties = recommender["properties"]
    similarity_matrix = recommender["similarity_matrix"]

    metrics = load_report(
        "price_model_metrics.csv",
        _file_cache_key(REPORT_DIR / "price_model_metrics.csv"),
    )
    price_comparison = load_report(
        "price_model_comparison.csv",
        _file_cache_key(REPORT_DIR / "price_model_comparison.csv"),
    )
    cross_validation = load_report(
        "price_model_cross_validation.csv",
        _file_cache_key(REPORT_DIR / "price_model_cross_validation.csv"),
    )
    error_categories = load_report(
        "price_error_by_category.csv",
        _file_cache_key(REPORT_DIR / "price_error_by_category.csv"),
    )

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



def render_kpi_card(label, value, context, progress, symbol, accent="cyan"):
    """Render a compact executive KPI card with a contextual indicator."""
    try:
        progress_value = min(100.0, max(0.0, float(progress)))
    except (TypeError, ValueError):
        progress_value = 0.0

    safe_label = escape(str(label))
    safe_value = escape(str(value))
    safe_context = escape(str(context))
    safe_symbol = escape(str(symbol))
    safe_accent = accent if accent in {"cyan", "teal", "blue"} else "cyan"

    st.markdown(
        f"""
        <article class="executive-kpi {safe_accent}">
          <div class="executive-kpi-head">
            <span class="executive-kpi-symbol" aria-hidden="true">{safe_symbol}</span>
            <span class="executive-kpi-label">{safe_label}</span>
            <span class="executive-kpi-indicator" aria-hidden="true"></span>
          </div>
          <div class="executive-kpi-value">{safe_value}</div>
          <div class="executive-kpi-context">{safe_context}</div>
          <div class="executive-kpi-track" role="img" aria-label="{safe_label} indicator: {progress_value:.1f}%">
            <span style="width:{progress_value:.1f}%"></span>
          </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def _style_chart_axis(ax):
    """Apply the shared dark navy / cyan dashboard chart palette."""
    ax.set_facecolor("#101d30")
    ax.tick_params(colors="#b8c9d9", labelsize=9)
    ax.xaxis.label.set_color("#91a8bd")
    ax.yaxis.label.set_color("#91a8bd")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#34475d", alpha=.48, linewidth=.8)
    ax.set_axisbelow(True)


def render_grouped_bar_chart(frame, columns, value_format="price", limit=10):
    """Render a readable horizontal grouped comparison chart."""
    available = [column for column in columns if column in frame.columns]
    if frame.empty or not available:
        st.info("There is not enough data to draw this comparison.")
        return

    chart_data = frame[available].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    chart_data = chart_data.sort_values(available[0], ascending=True).tail(limit)
    if chart_data.empty:
        st.info("There are no valid values for this chart.")
        return

    labels = [str(label) for label in chart_data.index]
    y = np.arange(len(labels))
    height = max(3.1, .43 * len(labels) + 1.1)
    fig, ax = plt.subplots(figsize=(9, height), dpi=120)
    fig.patch.set_facecolor("#101d30")
    _style_chart_axis(ax)
    palette = ["#59c8f2", "#42d6c3", "#82aaff"]
    bar_height = min(.34, .72 / len(available))
    offsets = (np.arange(len(available)) - (len(available) - 1) / 2) * (bar_height + .035)

    for index, column in enumerate(available):
        values = chart_data[column].fillna(0).to_numpy(dtype=float)
        bars = ax.barh(y + offsets[index], values, height=bar_height,
                       color=palette[index % len(palette)], label=column.replace("_", " "),
                       edgecolor="none", alpha=.96)
        for bar, value in zip(bars, values):
            if value > 0:
                label = f"{value:,.0f}" if value_format == "count" else f"{value:,.0f}"
                ax.annotate(label, (bar.get_width(), bar.get_y() + bar.get_height()/2),
                            xytext=(5, 0), textcoords="offset points",
                            va="center", color="#dce8f2", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#c3d2df")
    ax.set_xlabel("Properties" if value_format == "count" else "Sample nightly price · NPR", fontsize=8)
    ax.margins(x=.18)
    ax.legend(loc="lower right", frameon=False, labelcolor="#b8c9d9", fontsize=8, ncol=min(2, len(available)))
    fig.tight_layout(pad=1.25)
    st.pyplot(fig, width="stretch", clear_figure=True)
    plt.close(fig)


def render_horizontal_bar_chart(series, value_format="count", limit=10, color="#59c8f2"):
    """Render a compact horizontal ranking chart."""
    values = pd.to_numeric(series, errors="coerce").dropna().sort_values().tail(limit)
    if values.empty:
        st.info("There are no valid values for this chart.")
        return
    fig, ax = plt.subplots(figsize=(9, max(3.0, .42 * len(values) + .9)), dpi=120)
    fig.patch.set_facecolor("#101d30")
    _style_chart_axis(ax)
    bars = ax.barh([str(x) for x in values.index], values.to_numpy(),
                   color=color, height=.58, edgecolor="none")
    for bar, value in zip(bars, values.to_numpy()):
        suffix = "%" if value_format == "percent" else ""
        ax.annotate(f"{value:,.1f}{suffix}" if value_format == "percent" else f"{value:,.0f}",
                    (bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    color="#dce8f2", fontsize=8)
    ax.set_xlabel("Share / variation (%)" if value_format == "percent" else
                  ("Properties" if value_format == "count" else "Sample nightly price · NPR"), fontsize=8)
    ax.margins(x=.18)
    fig.tight_layout(pad=1.2)
    st.pyplot(fig, width="stretch", clear_figure=True)
    plt.close(fig)


def render_donut_chart(series, center_label="LISTINGS"):
    """Render a labelled dark-theme donut chart with a compact legend."""
    values = pd.to_numeric(series, errors="coerce")
    values = values[values > 0]
    if values.empty:
        st.info("There is not enough category data to draw this chart.")
        return
    values = values.sort_values(ascending=False).head(8)
    palette = ["#59c8f2", "#42d6c3", "#82aaff", "#8b9cf6",
               "#54b6a9", "#6e9fc8", "#a2c7dc", "#6d8298"]
    fig, ax = plt.subplots(figsize=(6.2, 4.1), dpi=120)
    fig.patch.set_facecolor("#101d30")
    ax.set_facecolor("#101d30")
    wedges, _, _ = ax.pie(
        values.values, startangle=90, counterclock=False,
        colors=palette[:len(values)], autopct=lambda pct: f"{pct:.0f}%" if pct >= 5 else "",
        pctdistance=.78, wedgeprops={"width": .34, "edgecolor": "#101d30", "linewidth": 2},
        textprops={"color": "#f2f7fc", "fontsize": 8},
    )
    ax.text(0, .06, f"{int(values.sum()):,}", ha="center", va="center",
            color="#f5f9ff", fontsize=18, fontweight="bold")
    ax.text(0, -.19, center_label, ha="center", va="center",
            color="#91a8bd", fontsize=7, fontweight="bold")
    ax.legend(wedges, [str(x) for x in values.index], loc="center left",
              bbox_to_anchor=(.88, .5), frameon=False, labelcolor="#c3d2df", fontsize=8)
    fig.tight_layout(pad=1.1)
    st.pyplot(fig, width="stretch", clear_figure=True)
    plt.close(fig)


def render_price_distribution(prices):
    """Render a histogram and compact boxplot for filtered sample prices."""
    values = pd.to_numeric(prices, errors="coerce").dropna()
    values = values[values >= 0]
    if values.empty:
        st.info("No valid nightly-price values match the selected filters.")
        return
    fig, (hist_ax, box_ax) = plt.subplots(
        2, 1, figsize=(10, 4.8), dpi=120,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#101d30")
    for ax in (hist_ax, box_ax):
        _style_chart_axis(ax)
    hist_ax.hist(values, bins=min(18, max(5, int(np.sqrt(len(values))))),
                 color="#59c8f2", edgecolor="#101d30", alpha=.9)
    hist_ax.axvline(values.median(), color="#42d6c3", linewidth=2,
                    linestyle="--", label=f"Median · NPR {values.median():,.0f}")
    hist_ax.set_ylabel("Observations", fontsize=8)
    hist_ax.legend(frameon=False, labelcolor="#c3d2df", fontsize=8)
    box_ax.boxplot(values, vert=False, patch_artist=True,
                   boxprops={"facecolor": "#17617a", "edgecolor": "#59c8f2"},
                   medianprops={"color": "#42d6c3", "linewidth": 2},
                   whiskerprops={"color": "#8fa8bf"},
                   capprops={"color": "#8fa8bf"},
                   flierprops={"marker": "o", "markersize": 2.5,
                               "markerfacecolor": "#82aaff", "markeredgecolor": "none"})
    box_ax.set_yticks([])
    box_ax.set_xlabel("Sample nightly price · NPR", fontsize=8)
    fig.tight_layout(pad=1.2)
    st.pyplot(fig, width="stretch", clear_figure=True)
    plt.close(fig)


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
    publication_rate = (published / total * 100) if total else 0
    full_inventory = max(len(properties), 1)
    price_scale = max(float(price_ceiling), 1.0)

    kpi_row_one = st.columns(3, gap="medium")
    with kpi_row_one[0]:
        render_kpi_card(
            "Total properties",
            f"{total:,}",
            f"Filtered from {len(properties):,} sample records",
            total / full_inventory * 100,
            "⌂",
            "cyan",
        )
    with kpi_row_one[1]:
        render_kpi_card(
            "Published properties",
            f"{published:,}",
            "Records marked as published",
            published / total * 100 if total else 0,
            "✓",
            "teal",
        )
    with kpi_row_one[2]:
        render_kpi_card(
            "Publication rate",
            f"{publication_rate:.1f}%",
            "Published share of selected inventory",
            publication_rate,
            "%",
            "blue",
        )

    kpi_row_two = st.columns(3, gap="medium")
    with kpi_row_two[0]:
        render_kpi_card(
            "Average nightly price",
            money(avg_price),
            "Mean sample price · NPR per night",
            (avg_price / price_scale * 100) if pd.notna(avg_price) else 0,
            "↗",
            "cyan",
        )
    with kpi_row_two[1]:
        render_kpi_card(
            "Median nightly price",
            money(median_price),
            "Middle value · less sensitive to outliers",
            (median_price / price_scale * 100) if pd.notna(median_price) else 0,
            "◆",
            "teal",
        )
    with kpi_row_two[2]:
        render_kpi_card(
            "Average guest rating",
            f"{avg_rating:.2f} / 5" if pd.notna(avg_rating) else "N/A",
            "Mean rating among records with a rating",
            (avg_rating / 5 * 100) if pd.notna(avg_rating) else 0,
            "★",
            "blue",
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

    # Dynamic findings are calculated from the currently filtered sample only.
    insight_columns = st.columns(3)
    median_prices = (
        neighbourhood_summary["Median_Price"].dropna()
        if "Median_Price" in neighbourhood_summary else pd.Series(dtype=float)
    )
    if not median_prices.empty:
        highest_median_area = str(median_prices.idxmax())
        insight_columns[0].metric(
            "Highest median-price area", highest_median_area,
            help=f"Median sample nightly price: {money(median_prices.max())}.",
        )
        insight_columns[0].caption(f"Median sample price · {money(median_prices.max())}")
    else:
        insight_columns[0].metric("Highest median-price area", "N/A")

    inventory_segments = (
        market_data["property_type"].fillna("Unspecified").value_counts()
        if "property_type" in market_data else pd.Series(dtype=int)
    )
    if not inventory_segments.empty and len(market_data):
        largest_segment = str(inventory_segments.index[0])
        segment_share = float(inventory_segments.iloc[0] / len(market_data) * 100)
        insight_columns[1].metric("Largest inventory segment", largest_segment)
        insight_columns[1].caption(f"{segment_share:.1f}% of filtered sample records")
    else:
        insight_columns[1].metric("Largest inventory segment", "N/A")

    rating_coverage = (
        market_data["rating"].notna().mean() * 100
        if "rating" in market_data else np.nan
    )
    insight_columns[2].metric(
        "Rating coverage", f"{rating_coverage:.1f}%" if pd.notna(rating_coverage) else "N/A"
    )
    insight_columns[2].caption("Share of selected records with a rating")

    # A short narrative insight makes the dashboard useful without implying
    # statistical significance or live-market verification.
    insight_parts = []
    if not median_prices.empty:
        insight_parts.append(
            f"{highest_median_area} has the highest median sample nightly price "
            f"({money(median_prices.max())})."
        )
    if not inventory_segments.empty:
        insight_parts.append(
            f"{str(inventory_segments.index[0])} is the largest accommodation segment "
            f"({segment_share:.1f}% of the selected sample)."
        )
    if pd.notna(rating_coverage):
        insight_parts.append(f"Rating coverage is {rating_coverage:.1f}%.")
    if insight_parts:
        st.markdown(
            '<div class="insight-narrative"><span class="insight-narrative-kicker">'
            'FILTERED SAMPLE INSIGHT</span><div>' + escape(" ".join(insight_parts)) +
            '</div><small>Automatically calculated from the selected synthetic sample records; not a live-market conclusion.</small></div>',
            unsafe_allow_html=True,
        )

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
                    render_grouped_bar_chart(
                        price_summary, ["Average_Price", "Median_Price"], value_format="price", limit=10
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
            render_donut_chart(
                market_data["property_type"].fillna("Unspecified").value_counts(),
                center_label="PROPERTY TYPES",
            )
        else:
            st.info("Property-type information is unavailable.")
        if not market_data.empty and "room_type" in market_data.columns:
            st.caption("Room-type composition")
            render_donut_chart(
                market_data["room_type"].fillna("Unspecified").value_counts(),
                center_label="ROOM TYPES",
            )

    st.divider()
    section_heading(
        "Pricing distribution",
        "The spread of sample nightly prices provides more context than an average alone.",
    )
    if not prices.empty:
        render_price_distribution(prices)
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
                    render_horizontal_bar_chart(
                        segment_summary["Observations"], value_format="count", limit=4, color="#59c8f2"
                    )

                with right:
                    section_heading(
                        "Segment Price Comparison",
                        "Compare average and median prices.",
                    )
                    render_grouped_bar_chart(
                        segment_summary, ["Average_Price", "Median_Price"], value_format="price", limit=4
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

                        render_grouped_bar_chart(
                            area_prices, ["Average_Price", "Median_Price"], value_format="price", limit=10
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

                        render_grouped_bar_chart(
                            type_prices.rename(columns={"mean": "Average_Price", "median": "Median_Price"}),
                            ["Average_Price", "Median_Price"], value_format="price", limit=10
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

                    render_horizontal_bar_chart(
                        volatility.sort_values("Price_CV_Percent", ascending=False)
                        .head(10)["Price_CV_Percent"],
                        value_format="percent", limit=10, color="#59c8f2"
                    )

                with right:

                    section_heading(
                        "Most Consistent Sample Prices",
                        "Areas with the lowest relative price dispersion.",
                    )

                    render_horizontal_bar_chart(
                        volatility.sort_values("Price_CV_Percent", ascending=True)
                        .head(10)["Price_CV_Percent"],
                        value_format="percent", limit=10, color="#42d6c3"
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
                    color=["#59C8F2", "#42D6C3"],
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
                    color="#59C8F2",
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
                color="#59C8F2",
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
        for start in range(0, len(recommendations), 2):
            card_columns = st.columns(2, gap="medium")
            for column, (_, item) in zip(
                card_columns,
                recommendations.iloc[start:start + 2].iterrows(),
            ):
                similarity_score = float(item["Similarity (%)"])
                safe_name = escape(str(item["Property Name"]))
                safe_id = escape(str(item["Property ID"]))
                safe_area = escape(str(item["Neighbourhood"]))
                safe_type = escape(str(item["Property Type"]))
                safe_room = escape(str(item["Room Type"]))
                with column:
                    st.markdown(
                        f"""
                        <article class="discovery-card">
                          <div class="discovery-card-top">
                            <span class="discovery-kicker">MATCHED PROPERTY</span>
                            <span class="discovery-score">{similarity_score:.1f}% match</span>
                          </div>
                          <h3 class="discovery-title">{safe_name}</h3>
                          <div class="discovery-location">{safe_area} · {safe_id}</div>
                          <div class="discovery-tags">
                            <span>{safe_type}</span><span>{safe_room}</span>
                          </div>
                          <div class="discovery-card-bottom">
                            <div><small>Sample nightly price</small><strong>{money(item['Nightly Price (NPR)'])}</strong></div>
                            <div class="discovery-rating"><span>★</span> {item['Rating']} / 5</div>
                          </div>
                          <div class="discovery-track"><span style="width:{min(100, max(0, similarity_score)):.1f}%"></span></div>
                        </article>
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
            section_heading(
                "Prediction Results",
                "A side-by-side view of the model estimate and the selected property's baseline.",
            )

            st.markdown(
                f"""
                <section class="prediction-hero">
                  <div class="prediction-kicker">EXPERIMENTAL MODEL ESTIMATE</div>
                  <div class="prediction-price">{money(prediction)}<span> / night</span></div>
                  <div class="prediction-change {'positive' if difference >= 0 else 'negative'}">
                    {'+' if difference >= 0 else ''}{percentage:.2f}% versus baseline
                  </div>
                </section>
                """,
                unsafe_allow_html=True,
            )

            p1, p2, p3 = st.columns(3)
            p1.metric("Estimated Nightly Price", money(prediction))
            p2.metric("Baseline Nightly Price", money(baseline))
            p3.metric("Absolute Difference", money(abs(difference)))

            comparison = pd.Series(
                {"Baseline": baseline, "Model estimate": prediction},
                dtype=float,
            )
            render_horizontal_bar_chart(
                comparison, value_format="price", limit=2, color="#59c8f2"
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
        "DATA INTELLIGENCE",
        "Explore, Inspect & Understand Your Data",
        "A focused workspace for filtering datasets, checking data quality, "
        "understanding distributions and exporting clean analysis extracts.",
    )

    st.info(
        "Data provenance: the property and price datasets contain synthetic "
        "sample records. Availability and price observations are not live "
        "booking inventory or verified market transactions."
    )

    def render_dataset_workspace(frame, dataset_name, key_prefix, filter_columns=()):
        """Render a consistent, defensive dataset exploration workspace."""
        if frame.empty:
            st.warning(f"{dataset_name} is empty or unavailable in this checkout.")
            return

        working = frame.copy()
        total_rows, total_columns = working.shape
        missing_cells = int(working.isna().sum().sum())
        missing_pct = (missing_cells / max(total_rows * total_columns, 1)) * 100
        duplicate_rows = int(working.duplicated().sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records", f"{total_rows:,}")
        k2.metric("Fields", f"{total_columns:,}")
        k3.metric("Missing cells", f"{missing_cells:,}", f"{missing_pct:.1f}% of all cells")
        k4.metric("Duplicate rows", f"{duplicate_rows:,}")

        with st.container(border=True):
            section_heading("Filter & refine", "Filters apply to the preview, charts and downloaded extract.")
            search_col, filter_col = st.columns([1.2, 1])
            with search_col:
                search_text = st.text_input(
                    "Search across all fields",
                    placeholder="Search neighbourhood, property, ID…",
                    key=f"{key_prefix}_search",
                )
            with filter_col:
                available_filter_columns = [
                    column for column in filter_columns
                    if column in working.columns
                    and working[column].nunique(dropna=True) > 1
                ]
                selected_filter_column = st.selectbox(
                    "Category filter",
                    ["No category filter"] + available_filter_columns,
                    key=f"{key_prefix}_filter_column",
                )

            if search_text.strip():
                searchable = working.astype("string").fillna("")
                mask = searchable.apply(
                    lambda column: column.str.contains(
                        search_text.strip(), case=False, regex=False
                    )
                ).any(axis=1)
                working = working.loc[mask].copy()

            if selected_filter_column != "No category filter":
                choices = sorted(
                    working[selected_filter_column].dropna().astype(str).unique().tolist()
                )
                selected_values = st.multiselect(
                    f"Select {selected_filter_column.replace('_', ' ')}",
                    choices,
                    default=choices,
                    key=f"{key_prefix}_category_values",
                )
                working = working[
                    working[selected_filter_column].astype(str).isin(selected_values)
                ].copy()

        st.caption(
            f"Showing {len(working):,} of {total_rows:,} records · "
            f"{working.shape[1]:,} fields"
        )

        if working.empty:
            st.warning("No records match the current filters. Adjust the filters to continue.")
            return

        section_heading("Explore a field", "Review a distribution from the filtered dataset.")
        numeric_columns = [
            column for column in working.columns
            if pd.api.types.is_numeric_dtype(working[column])
            or pd.to_numeric(working[column], errors="coerce").notna().sum() > 0
        ]
        category_candidates = [
            column for column in working.columns
            if not pd.api.types.is_numeric_dtype(working[column])
            and 1 < working[column].nunique(dropna=True) <= 100
        ]
        chart_left, chart_right = st.columns(2)

        with chart_left:
            if numeric_columns:
                numeric_choice = st.selectbox(
                    "Numeric field", numeric_columns,
                    key=f"{key_prefix}_numeric_chart",
                )
                numeric_values = pd.to_numeric(working[numeric_choice], errors="coerce").dropna()
                if not numeric_values.empty:
                    bins = min(16, max(5, int(np.sqrt(len(numeric_values)))))
                    histogram = pd.cut(numeric_values, bins=bins, duplicates="drop").value_counts().sort_index()
                    histogram.index = histogram.index.astype(str)
                    st.markdown("**Value distribution**")
                    st.bar_chart(histogram, color="#59C8F2")
                else:
                    st.caption("No numeric values are available for this field.")
            else:
                st.caption("No numeric fields are available in this dataset.")

        with chart_right:
            if category_candidates:
                category_choice = st.selectbox(
                    "Category field", category_candidates,
                    key=f"{key_prefix}_category_chart",
                )
                category_counts = (
                    working[category_choice].fillna("Missing").astype(str)
                    .value_counts().head(12)
                )
                st.markdown("**Most common values**")
                st.bar_chart(category_counts, color="#42D6C3")
            else:
                st.caption("No suitable categorical fields are available.")

        with st.expander("Data quality report", expanded=False):
            quality = pd.DataFrame({
                "Field": working.columns,
                "Data type": [str(working[column].dtype) for column in working.columns],
                "Missing values": [int(working[column].isna().sum()) for column in working.columns],
                "Missing (%)": [
                    round(float(working[column].isna().mean() * 100), 2)
                    for column in working.columns
                ],
                "Unique values": [int(working[column].nunique(dropna=True)) for column in working.columns],
            }).sort_values(["Missing values", "Field"], ascending=[False, True])
            st.dataframe(quality, width="stretch", hide_index=True)
            st.markdown("**Numeric summary**")
            numeric_summary = working.select_dtypes(include=np.number).describe().T
            if numeric_summary.empty:
                st.caption("No native numeric columns are available for descriptive statistics.")
            else:
                st.dataframe(numeric_summary, width="stretch")

            st.download_button(
                "Download data quality report",
                quality.to_csv(index=False).encode("utf-8"),
                file_name=f"{key_prefix}_data_quality.csv",
                mime="text/csv",
                key=f"{key_prefix}_quality_download",
            )

        section_heading("Filtered records", "Choose the visible columns and page through the result set.")
        visible_columns = st.multiselect(
            "Visible columns", working.columns.tolist(),
            default=working.columns.tolist(),
            key=f"{key_prefix}_visible_columns",
        )
        if not visible_columns:
            st.warning("Select at least one column to display the table.")
        else:
            page_controls = st.columns([1, 1, 2])
            with page_controls[0]:
                page_size = st.selectbox(
                    "Rows per page", [25, 50, 100, 250, 500],
                    index=1, key=f"{key_prefix}_page_size",
                )
            page_count = max(1, int(np.ceil(len(working) / page_size)))
            with page_controls[1]:
                page_number = st.number_input(
                    "Page", min_value=1, max_value=page_count, value=1,
                    step=1, key=f"{key_prefix}_page_number",
                )
            first_row = (int(page_number) - 1) * page_size
            last_row = min(first_row + page_size, len(working))
            with page_controls[2]:
                st.caption(f"Rows {first_row + 1:,}–{last_row:,} of {len(working):,}")
            st.dataframe(
                working.iloc[first_row:last_row][visible_columns],
                width="stretch", hide_index=True, height=440,
            )

        st.download_button(
            "Download filtered dataset (CSV)",
            working.to_csv(index=False).encode("utf-8"),
            file_name=f"{key_prefix}_filtered.csv",
            mime="text/csv",
            key=f"{key_prefix}_data_download",
            width="stretch",
        )

    tab1, tab2, tab3 = st.tabs([
        "Property Dataset", "Availability", "Price Observations",
    ])

    with tab1:
        section_heading("Property dataset", "Explore listing attributes, pricing, ratings and neighbourhood coverage.")
        render_dataset_workspace(
            properties, "Property dataset", "properties",
            filter_columns=("neighbourhood", "property_type", "room_type", "city", "published"),
        )

    with tab2:
        section_heading("Availability dataset", "Inspect calendar-level records and the fields available in this extract.")
        render_dataset_workspace(
            availability, "Availability dataset", "availability",
            filter_columns=("neighbourhood", "property_type", "room_type", "available", "city"),
        )

    with tab3:
        section_heading("Price observation dataset", "Review sample prices and the property or stay attributes attached to them.")
        render_dataset_workspace(
            price_observations, "Price observation dataset", "price_observations",
            filter_columns=("neighbourhood", "property_type", "room_type", "city", "currency_original"),
        )


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    page_header(
        "MODEL INTELLIGENCE",
        "Performance, Validation & Error Analysis",
        "A transparent view of model quality, cross-validation consistency, "
        "price-band performance and individual prediction behaviour.",
    )

    # Load the row-level holdout report for interactive diagnostics.
    prediction_records = load_report(
        "price_predictions.csv",
        _file_cache_key(REPORT_DIR / "price_predictions.csv"),
    )

    metrics_df = metrics.copy()
    comparison_df = price_comparison.copy()
    cv_df = cross_validation.copy()
    error_df = error_categories.copy()

    def metric_lookup(frame):
        if frame.empty or not {"metric", "value"}.issubset(frame.columns):
            return {}
        values = {}
        for _, row in frame.iterrows():
            key = str(row["metric"]).strip().lower()
            value = pd.to_numeric(row["value"], errors="coerce")
            if key and pd.notna(value):
                values[key] = float(value)
        return values

    metric_values = metric_lookup(metrics_df)

    def evaluation_metric(*names):
        for name in names:
            value = metric_values.get(name.strip().lower())
            if value is not None and np.isfinite(value):
                return float(value)
        return None

    mae = evaluation_metric("mae")
    rmse = evaluation_metric("rmse")
    r2 = evaluation_metric("r2", "r2_score", "r2 score")
    baseline_mae = evaluation_metric("baseline mae")
    baseline_rmse = evaluation_metric("baseline rmse")
    baseline_r2 = evaluation_metric("baseline r2")
    mae_improvement = evaluation_metric("mae improvement percentage")

    if mae_improvement is None and baseline_mae and mae is not None:
        mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100

    training_rows = evaluation_metric("training rows")
    testing_rows = evaluation_metric("testing rows")
    training_properties = evaluation_metric("training properties")
    testing_properties = evaluation_metric("testing properties")
    overlapping_properties = evaluation_metric("overlapping properties")

    # Normalize numeric fields without changing the saved source reports.
    for frame in (comparison_df, cv_df, error_df):
        for column in frame.columns:
            if column not in {"model", "price_category"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    section_heading(
        "Held-out test performance",
        "Saved evaluation metrics from the property-level test split.",
    )

    kpi_columns = st.columns(4)
    with kpi_columns[0]:
        st.metric(
            "Mean absolute error",
            money(mae) if mae is not None else "N/A",
            help="Average absolute difference between the predicted and actual target price.",
        )
    with kpi_columns[1]:
        st.metric(
            "Root mean squared error",
            money(rmse) if rmse is not None else "N/A",
            help="Penalizes larger errors more heavily than MAE.",
        )
    with kpi_columns[2]:
        st.metric(
            "R² score",
            f"{r2:.4f}" if r2 is not None else "N/A",
            help="Explained variance on the saved held-out test set; interpret alongside the feature caveat below.",
        )
    with kpi_columns[3]:
        st.metric(
            "MAE vs baseline",
            f"{mae_improvement:+.2f}%" if mae_improvement is not None else "N/A",
            delta="Lower error than baseline" if mae_improvement is not None and mae_improvement > 0 else (
                "Higher error than baseline" if mae_improvement is not None else None
            ),
            delta_color="off",
            help="Relative MAE reduction compared with using the existing property nightly price as the baseline prediction.",
        )

    st.markdown(
        """
        <div class="evaluation-caveat">
          <span class="evaluation-caveat-kicker">INTERPRETATION NOTE</span>
          The price model includes the property's existing nightly price as an input feature.
          The reported scores therefore measure performance with that baseline information
          available; they should not be described as independent market-price discovery.
          Property and price observations in this project include synthetic sample records.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Test split and evaluation setup", expanded=False):
        split_columns = st.columns(4)
        split_columns[0].metric("Training rows", f"{int(training_rows):,}" if training_rows is not None else "N/A")
        split_columns[1].metric("Test rows", f"{int(testing_rows):,}" if testing_rows is not None else "N/A")
        split_columns[2].metric("Training properties", f"{int(training_properties):,}" if training_properties is not None else "N/A")
        split_columns[3].metric("Test properties", f"{int(testing_properties):,}" if testing_properties is not None else "N/A")

        if overlapping_properties is not None:
            if int(overlapping_properties) == 0:
                st.info("Property-level split check: 0 property IDs overlap between training and test sets.")
            else:
                st.error(
                    f"Split integrity issue: {int(overlapping_properties):,} property IDs overlap."
                )
        st.caption(
            "The saved training workflow uses GroupShuffleSplit with property_id as the grouping key. "
            "A group-based split reduces the risk of the same property appearing in both partitions."
        )

    st.divider()

    # Shared chart helpers keep evaluation charts aligned with the dashboard palette.
    def render_evaluation_bar(series, title, axis_label, color="#59c8f2", value_kind="money"):
        values = pd.to_numeric(series, errors="coerce").dropna().sort_values(ascending=True)
        if values.empty:
            st.caption("No valid values are available for this chart.")
            return

        fig, ax = plt.subplots(figsize=(8.2, max(2.8, .48 * len(values) + .85)), dpi=120)
        fig.patch.set_facecolor("#101d30")
        _style_chart_axis(ax)
        bars = ax.barh(
            [str(label) for label in values.index],
            values.to_numpy(dtype=float),
            color=color,
            height=.58,
            edgecolor="none",
        )
        for bar, value in zip(bars, values.to_numpy(dtype=float)):
            if value_kind == "percent":
                label = f"{value:.1f}%"
            elif value_kind == "score":
                label = f"{value:.4f}"
            else:
                label = f"{value:,.0f}"
            offset = 5 if value >= 0 else -5
            ax.annotate(
                label,
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(offset, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                color="#dce8f2",
                fontsize=8,
            )
        ax.axvline(0, color="#8298ad", linewidth=.8, alpha=.65)
        ax.set_xlabel(axis_label, fontsize=8)
        ax.set_title(title, color="#edf4fb", loc="left", fontsize=10, pad=11, fontweight="bold")
        ax.margins(x=.2)
        fig.tight_layout(pad=1.2)
        st.pyplot(fig, width="stretch", clear_figure=True)
        plt.close(fig)

    def render_evaluation_lines(frame, columns, labels, title, axis_label):
        available = [column for column in columns if column in frame.columns]
        if frame.empty or not available:
            st.info("Cross-validation series are unavailable.")
            return

        x_values = (
            frame["fold"].astype(str).tolist()
            if "fold" in frame.columns
            else [str(index + 1) for index in range(len(frame))]
        )
        palette = ["#59c8f2", "#42d6c3", "#82aaff"]
        fig, ax = plt.subplots(figsize=(8.5, 3.7), dpi=120)
        fig.patch.set_facecolor("#101d30")
        _style_chart_axis(ax)
        ax.grid(axis="y", color="#34475d", alpha=.48, linewidth=.8)
        ax.grid(axis="x", visible=False)

        for index, column in enumerate(available):
            values = pd.to_numeric(frame[column], errors="coerce")
            label = labels[index] if index < len(labels) else column.replace("_", " ")
            ax.plot(
                x_values,
                values,
                color=palette[index % len(palette)],
                marker="o",
                markersize=5,
                linewidth=2,
                label=label,
            )

        ax.set_title(title, color="#edf4fb", loc="left", fontsize=10, pad=11, fontweight="bold")
        ax.set_xlabel("Validation fold", fontsize=8)
        ax.set_ylabel(axis_label, fontsize=8)
        ax.legend(frameon=False, labelcolor="#c3d2df", fontsize=8)
        fig.tight_layout(pad=1.2)
        st.pyplot(fig, width="stretch", clear_figure=True)
        plt.close(fig)

    tab_performance, tab_validation, tab_errors, tab_predictions = st.tabs([
        "Model Comparison",
        "Validation Stability",
        "Error Diagnostics",
        "Prediction Explorer",
    ])

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------
    with tab_performance:
        section_heading(
            "Benchmark comparison",
            "Compare saved model results using consistent error and fit metrics.",
        )

        if comparison_df.empty or "model" not in comparison_df.columns:
            st.info("The model comparison report is unavailable.")
        else:
            model_view = comparison_df.copy()
            for column in ("MAE", "RMSE", "R2"):
                if column in model_view.columns:
                    model_view[column] = pd.to_numeric(model_view[column], errors="coerce")
            if "MAE" in model_view.columns:
                model_view = model_view.sort_values("MAE", ascending=True, na_position="last")

            best_model_name = (
                str(model_view.iloc[0]["model"])
                if not model_view.empty and "model" in model_view.columns
                else "N/A"
            )
            best_model_mae = (
                float(model_view.iloc[0]["MAE"])
                if not model_view.empty and "MAE" in model_view.columns
                and pd.notna(model_view.iloc[0]["MAE"])
                else None
            )

            best_col, count_col = st.columns([1.2, 1])
            with best_col:
                st.markdown(
                    f"""
                    <div class="evaluation-highlight">
                      <span class="evaluation-highlight-label">LOWEST MAE</span>
                      <strong>{escape(best_model_name)}</strong>
                      <small>{money(best_model_mae) if best_model_mae is not None else "Metric unavailable"}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with count_col:
                st.metric("Models benchmarked", f"{len(model_view):,}")

            display_columns = [
                column for column in ["model", "MAE", "RMSE", "R2"]
                if column in model_view.columns
            ]
            st.dataframe(
                model_view[display_columns].rename(columns={
                    "model": "Model",
                    "MAE": "MAE · NPR",
                    "RMSE": "RMSE · NPR",
                    "R2": "R²",
                }),
                width="stretch",
                hide_index=True,
            )

            chart_a, chart_b = st.columns(2)
            with chart_a:
                if "MAE" in model_view.columns:
                    render_evaluation_bar(
                        model_view.set_index("model")["MAE"],
                        "Mean absolute error",
                        "Error · NPR",
                        color="#59c8f2",
                    )
            with chart_b:
                if "RMSE" in model_view.columns:
                    render_evaluation_bar(
                        model_view.set_index("model")["RMSE"],
                        "Root mean squared error",
                        "Error · NPR",
                        color="#42d6c3",
                    )

            if "R2" in model_view.columns:
                render_evaluation_bar(
                    model_view.set_index("model")["R2"],
                    "Explained variance (R²)",
                    "R² score",
                    color="#82aaff",
                    value_kind="score",
                )

        st.divider()
        section_heading("Machine learning vs baseline", "The baseline predicts with the property's existing nightly price.")
        baseline_columns = st.columns(3)
        baseline_columns[0].metric("Model MAE", money(mae) if mae is not None else "N/A")
        baseline_columns[1].metric("Baseline MAE", money(baseline_mae) if baseline_mae is not None else "N/A")
        baseline_columns[2].metric(
            "MAE reduction",
            f"{mae_improvement:.2f}%" if mae_improvement is not None else "N/A",
        )

        if mae is not None and baseline_mae is not None:
            baseline_plot = pd.Series(
                {"Gradient Boosting": mae, "Existing-price baseline": baseline_mae}
            )
            render_evaluation_bar(
                baseline_plot,
                "Held-out MAE: model vs baseline",
                "Mean absolute error · NPR",
                color="#42d6c3",
            )

    # --------------------------------------------------------
    # VALIDATION STABILITY
    # --------------------------------------------------------
    with tab_validation:
        section_heading(
            "Cross-validation consistency",
            "Fold-level metrics show how much performance varies across validation partitions.",
        )

        if cv_df.empty:
            st.info("The cross-validation report is unavailable.")
        else:
            required_cv = ["GB_MAE", "GB_RMSE", "GB_R2"]
            cv_numeric = cv_df.copy()
            for column in cv_numeric.columns:
                if column != "fold":
                    cv_numeric[column] = pd.to_numeric(cv_numeric[column], errors="coerce")

            summary_fields = [
                ("GB_MAE", "Mean CV MAE", money),
                ("GB_RMSE", "Mean CV RMSE", money),
                ("GB_R2", "Mean CV R²", lambda value: f"{value:.4f}"),
            ]
            summary_cols = st.columns(3)
            for index, (column, label, formatter) in enumerate(summary_fields):
                if column in cv_numeric.columns:
                    mean_value = cv_numeric[column].mean()
                    std_value = cv_numeric[column].std()
                    summary_cols[index].metric(
                        label,
                        formatter(mean_value) if pd.notna(mean_value) else "N/A",
                        f"Fold σ · {formatter(std_value)}" if pd.notna(std_value) else None,
                    )

            if all(column in cv_numeric.columns for column in required_cv):
                stability_a, stability_b = st.columns(2)
                with stability_a:
                    render_evaluation_lines(
                        cv_numeric,
                        ["GB_MAE", "Baseline_MAE"],
                        ["Gradient Boosting", "Existing-price baseline"],
                        "MAE across folds",
                        "Error · NPR",
                    )
                with stability_b:
                    render_evaluation_lines(
                        cv_numeric,
                        ["GB_RMSE", "Baseline_RMSE"],
                        ["Gradient Boosting", "Existing-price baseline"],
                        "RMSE across folds",
                        "Error · NPR",
                    )
                render_evaluation_lines(
                    cv_numeric,
                    ["GB_R2", "Baseline_R2"],
                    ["Gradient Boosting", "Existing-price baseline"],
                    "R² across folds",
                    "R² score",
                )

            fold_display = cv_numeric.rename(columns={
                "fold": "Fold",
                "training_properties": "Training properties",
                "testing_properties": "Test properties",
                "GB_MAE": "Model MAE · NPR",
                "GB_RMSE": "Model RMSE · NPR",
                "GB_R2": "Model R²",
                "Baseline_MAE": "Baseline MAE · NPR",
                "Baseline_RMSE": "Baseline RMSE · NPR",
                "Baseline_R2": "Baseline R²",
            })
            st.markdown("#### Fold results")
            st.dataframe(fold_display, width="stretch", hide_index=True)

            if "GB_MAE" in cv_numeric.columns:
                valid_folds = cv_numeric.dropna(subset=["GB_MAE"])
                if not valid_folds.empty:
                    best_fold = valid_folds.loc[valid_folds["GB_MAE"].idxmin()]
                    worst_fold = valid_folds.loc[valid_folds["GB_MAE"].idxmax()]
                    fold_insights = st.columns(2)
                    fold_insights[0].metric(
                        "Lowest-error fold",
                        f"Fold {int(best_fold['fold'])}" if "fold" in best_fold else str(best_fold.name),
                        money(best_fold["GB_MAE"]),
                    )
                    fold_insights[1].metric(
                        "Highest-error fold",
                        f"Fold {int(worst_fold['fold'])}" if "fold" in worst_fold else str(worst_fold.name),
                        money(worst_fold["GB_MAE"]),
                    )

    # --------------------------------------------------------
    # ERROR DIAGNOSTICS
    # --------------------------------------------------------
    with tab_errors:
        section_heading(
            "Where does the model miss?",
            "Compare error size and prediction bias across the saved price bands.",
        )

        if error_df.empty:
            st.info("The price-category error report is unavailable.")
        else:
            if "price_category" in error_df.columns:
                error_sorted = error_df.sort_values(
                    "mean_absolute_error" if "mean_absolute_error" in error_df.columns else "price_category",
                    ascending=False,
                    na_position="last",
                )
                st.dataframe(
                    error_sorted.rename(columns={
                        "price_category": "Price band",
                        "sample_count": "Test records",
                        "average_actual_price": "Actual mean · NPR",
                        "average_predicted_price": "Predicted mean · NPR",
                        "mean_absolute_error": "MAE · NPR",
                        "mean_percentage_error": "MAPE · %",
                        "average_signed_error": "Signed error · NPR",
                    }),
                    width="stretch",
                    hide_index=True,
                )

                err_a, err_b = st.columns(2)
                with err_a:
                    if "mean_absolute_error" in error_df.columns:
                        render_evaluation_bar(
                            error_df.set_index("price_category")["mean_absolute_error"],
                            "Mean absolute error by price band",
                            "Error · NPR",
                            color="#59c8f2",
                        )
                with err_b:
                    if "mean_percentage_error" in error_df.columns:
                        render_evaluation_bar(
                            error_df.set_index("price_category")["mean_percentage_error"],
                            "Mean percentage error by price band",
                            "Mean percentage error",
                            color="#82aaff",
                            value_kind="percent",
                        )

                if "average_signed_error" in error_df.columns:
                    render_evaluation_bar(
                        error_df.set_index("price_category")["average_signed_error"],
                        "Average prediction bias by price band",
                        "Predicted − actual · NPR",
                        color="#42d6c3",
                    )
                    st.caption(
                        "Positive signed error means the model overpredicts on average; "
                        "negative signed error means it underpredicts."
                    )

                insight_candidates = error_df.dropna(
                    subset=["mean_absolute_error"]
                ) if "mean_absolute_error" in error_df.columns else pd.DataFrame()
                if not insight_candidates.empty:
                    hardest_band = insight_candidates.loc[
                        insight_candidates["mean_absolute_error"].idxmax()
                    ]
                    st.markdown(
                        f"""
                        <div class="evaluation-highlight">
                          <span class="evaluation-highlight-label">LARGEST ABSOLUTE ERROR</span>
                          <strong>{escape(str(hardest_band['price_category']))}</strong>
                          <small>MAE · {money(hardest_band['mean_absolute_error'])}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.divider()
        section_heading("Residual sign guide", "Use the sign of predicted − actual to understand direction of error.")
        guide_cols = st.columns(2)
        with guide_cols[0]:
            st.markdown(
                '<div class="evaluation-sign-card positive"><strong>Positive residual</strong><span>Overprediction · estimate is above the actual target</span></div>',
                unsafe_allow_html=True,
            )
        with guide_cols[1]:
            st.markdown(
                '<div class="evaluation-sign-card negative"><strong>Negative residual</strong><span>Underprediction · estimate is below the actual target</span></div>',
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # ROW-LEVEL PREDICTION EXPLORER
    # --------------------------------------------------------
    with tab_predictions:
        section_heading(
            "Held-out prediction explorer",
            "Filter individual test records, inspect residuals and export the selected diagnostic sample.",
        )

        if prediction_records.empty:
            st.info("The row-level price_predictions.csv report is unavailable.")
        else:
            prediction_view = prediction_records.copy()
            numeric_prediction_columns = [
                "actual_price_npr",
                "predicted_price_npr",
                "baseline_price_prediction_npr",
                "absolute_error_npr",
                "signed_error_npr",
                "percentage_error",
            ]
            for column in numeric_prediction_columns:
                if column in prediction_view.columns:
                    prediction_view[column] = pd.to_numeric(
                        prediction_view[column], errors="coerce"
                    )

            if {"actual_price_npr", "predicted_price_npr"}.issubset(prediction_view.columns):
                if "absolute_error_npr" not in prediction_view.columns:
                    prediction_view["absolute_error_npr"] = (
                        prediction_view["predicted_price_npr"]
                        - prediction_view["actual_price_npr"]
                    ).abs()
                if "signed_error_npr" not in prediction_view.columns:
                    prediction_view["signed_error_npr"] = (
                        prediction_view["predicted_price_npr"]
                        - prediction_view["actual_price_npr"]
                    )

                filter_frame = st.container(border=True)
                with filter_frame:
                    filter_cols = st.columns(2)
                    with filter_cols[0]:
                        neighbourhood_values = (
                            sorted(prediction_view["neighbourhood"].dropna().astype(str).unique())
                            if "neighbourhood" in prediction_view.columns else []
                        )
                        selected_neighbourhoods = st.multiselect(
                            "Neighbourhood",
                            neighbourhood_values,
                            default=neighbourhood_values,
                            key="eval_prediction_neighbourhoods",
                        )
                    with filter_cols[1]:
                        property_type_values = (
                            sorted(prediction_view["property_type"].dropna().astype(str).unique())
                            if "property_type" in prediction_view.columns else []
                        )
                        selected_property_types = st.multiselect(
                            "Property type",
                            property_type_values,
                            default=property_type_values,
                            key="eval_prediction_property_types",
                        )

                    max_abs_error = float(
                        prediction_view["absolute_error_npr"].dropna().max()
                    ) if prediction_view["absolute_error_npr"].notna().any() else 0.0
                    if max_abs_error > 0:
                        max_error_filter = st.slider(
                            "Maximum absolute error · NPR",
                            min_value=0.0,
                            max_value=float(np.ceil(max_abs_error)),
                            value=float(np.ceil(max_abs_error)),
                            step=max(1.0, float(np.ceil(max_abs_error / 100))),
                            key="eval_prediction_max_error",
                        )
                    else:
                        max_error_filter = 0.0
                        st.caption("Absolute error range is not available for this report.")

                filtered_predictions = prediction_view.copy()
                if "neighbourhood" in filtered_predictions.columns:
                    filtered_predictions = filtered_predictions[
                        filtered_predictions["neighbourhood"].astype(str).isin(selected_neighbourhoods)
                    ]
                if "property_type" in filtered_predictions.columns:
                    filtered_predictions = filtered_predictions[
                        filtered_predictions["property_type"].astype(str).isin(selected_property_types)
                    ]
                if max_abs_error > 0:
                    filtered_predictions = filtered_predictions[
                        filtered_predictions["absolute_error_npr"].le(max_error_filter)
                        | filtered_predictions["absolute_error_npr"].isna()
                    ]

                if filtered_predictions.empty:
                    st.warning("No test records match the selected filters.")
                else:
                    filtered_mae = filtered_predictions["absolute_error_npr"].mean()
                    filtered_mape = (
                        filtered_predictions["percentage_error"].mean()
                        if "percentage_error" in filtered_predictions.columns else np.nan
                    )
                    filtered_metrics = st.columns(4)
                    filtered_metrics[0].metric("Test records", f"{len(filtered_predictions):,}")
                    filtered_metrics[1].metric("Filtered MAE", money(filtered_mae))
                    filtered_metrics[2].metric(
                        "Median absolute error",
                        money(filtered_predictions["absolute_error_npr"].median()),
                    )
                    filtered_metrics[3].metric(
                        "Mean percentage error",
                        f"{filtered_mape:.2f}%" if pd.notna(filtered_mape) else "N/A",
                    )

                    plot_left, plot_right = st.columns(2)
                    with plot_left:
                        x = filtered_predictions["actual_price_npr"]
                        y = filtered_predictions["predicted_price_npr"]
                        valid_xy = pd.DataFrame({"actual": x, "predicted": y}).dropna()
                        if not valid_xy.empty:
                            fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=120)
                            fig.patch.set_facecolor("#101d30")
                            ax.set_facecolor("#101d30")
                            ax.scatter(
                                valid_xy["actual"], valid_xy["predicted"],
                                s=25, alpha=.78, color="#59c8f2",
                                edgecolors="none",
                            )
                            low = float(min(valid_xy.min().min(), 0))
                            high = float(max(valid_xy.max().max(), 1))
                            ax.plot([low, high], [low, high], color="#42d6c3", linewidth=1.7, linestyle="--", label="Perfect prediction")
                            ax.set_xlim(left=low)
                            ax.set_ylim(bottom=low)
                            ax.set_xlabel("Actual target price · NPR", color="#91a8bd", fontsize=8)
                            ax.set_ylabel("Predicted price · NPR", color="#91a8bd", fontsize=8)
                            ax.set_title("Actual vs predicted", color="#edf4fb", loc="left", fontsize=10, pad=11, fontweight="bold")
                            ax.tick_params(colors="#b8c9d9", labelsize=8)
                            for spine in ax.spines.values():
                                spine.set_visible(False)
                            ax.grid(color="#34475d", alpha=.48, linewidth=.8)
                            ax.legend(frameon=False, labelcolor="#c3d2df", fontsize=8)
                            fig.tight_layout(pad=1.2)
                            st.pyplot(fig, width="stretch", clear_figure=True)
                            plt.close(fig)

                    with plot_right:
                        render_evaluation_bar(
                            filtered_predictions.set_index(
                                filtered_predictions.get(
                                    "property_id",
                                    pd.Series(filtered_predictions.index, index=filtered_predictions.index),
                                ).astype(str)
                            )["absolute_error_npr"].sort_values(ascending=True).tail(12),
                            "Largest absolute errors in the filtered sample",
                            "Absolute error · NPR",
                            color="#82aaff",
                        )

                    residual_columns = [
                        column for column in [
                            "property_id", "neighbourhood", "property_type", "room_type",
                            "actual_price_npr", "predicted_price_npr",
                            "absolute_error_npr", "signed_error_npr", "percentage_error",
                        ]
                        if column in filtered_predictions.columns
                    ]
                    worst_predictions = filtered_predictions.sort_values(
                        "absolute_error_npr", ascending=False, na_position="last"
                    )
                    st.markdown("#### Largest individual prediction errors")
                    st.dataframe(
                        worst_predictions[residual_columns].head(20).rename(columns={
                            "property_id": "Property ID",
                            "neighbourhood": "Neighbourhood",
                            "property_type": "Property type",
                            "room_type": "Room type",
                            "actual_price_npr": "Actual · NPR",
                            "predicted_price_npr": "Predicted · NPR",
                            "absolute_error_npr": "Absolute error · NPR",
                            "signed_error_npr": "Signed error · NPR",
                            "percentage_error": "Percentage error · %",
                        }),
                        width="stretch",
                        hide_index=True,
                    )
                    st.download_button(
                        "Download filtered prediction diagnostics",
                        filtered_predictions.to_csv(index=False).encode("utf-8"),
                        file_name="filtered_prediction_diagnostics.csv",
                        mime="text/csv",
                        key="eval_prediction_export",
                        width="stretch",
                    )
            else:
                st.warning("The row-level report does not contain the required actual and predicted price fields.")

    st.divider()
    st.caption(
        "Evaluation results are saved project reports based on the current prepared datasets. "
        "They are useful for experimentation and comparison, not evidence of live-market performance."
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
