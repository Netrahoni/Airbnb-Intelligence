
from pathlib import Path
from datetime import date

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
    page_title="Kathmandu Airbnb | ML Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0B1220;
    --panel: #111C2E;
    --panel-light: #172438;
    --border: #263449;
    --primary: #38BDF8;
    --accent: #2DD4BF;
    --text: #E2E8F0;
    --muted: #94A3B8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: #0E192A;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 800 !important;
    letter-spacing: -0.6px;
}

p, label, li {
    color: var(--text);
}

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 18px 20px;
    border-radius: 14px;
    min-height: 115px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.85rem;
}

[data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-weight: 800;
}

[data-testid="stMetricDelta"] {
    font-size: 0.8rem;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(100deg, #0284C7, #0D9488);
    color: white;
    border: 0;
    border-radius: 9px;
    padding: 0.6rem 1.1rem;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border: 0;
    color: white;
    filter: brightness(1.15);
}

input, textarea {
    color: var(--text) !important;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] {
    background: var(--panel-light);
    border-color: var(--border);
    border-radius: 9px;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

button[data-baseweb="tab"] {
    color: var(--muted);
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary);
}

hr {
    border-color: var(--border);
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--muted);
}

div[data-testid="stAlert"] {
    border-radius: 11px;
}

.hero {
    background: linear-gradient(120deg, #142B45, #102A35);
    border: 1px solid #24465D;
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 22px;
}

.hero-eyebrow {
    color: #67E8F9;
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    color: #F8FAFC;
    font-size: 2.25rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 8px 0;
}

.hero-description {
    color: #CBD5E1;
    font-size: 1rem;
    margin-bottom: 0;
}

.section-label {
    color: var(--primary);
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 7px;
}

.info-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 18px;
    margin-bottom: 10px;
}

.info-card h4 {
    color: #F8FAFC;
    margin: 0 0 8px 0;
}

.info-card p {
    color: var(--muted);
    margin: 0;
    font-size: 0.9rem;
}

.property-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 17px;
    margin-bottom: 12px;
}

.property-card-title {
    color: #F8FAFC;
    font-size: 1.05rem;
    font-weight: 700;
}

.property-card-meta {
    color: var(--muted);
    font-size: 0.85rem;
}

.similarity {
    color: var(--accent);
    font-size: 1.2rem;
    font-weight: 800;
}

.footer {
    color: var(--muted);
    text-align: center;
    font-size: 0.8rem;
    padding: 24px 0 5px 0;
}


/* ============================================================
   PROFESSIONAL DASHBOARD LAYER
   ============================================================ */

:root {
    --bg: #080F1C;
    --panel: #101B2D;
    --panel-light: #17263B;
    --border: rgba(148, 163, 184, 0.16);
    --primary: #55C7F3;
    --accent: #42D6C3;
    --text: #EAF2FC;
    --muted: #91A4BC;
}

.stApp {
    background:
        radial-gradient(ellipse at 82% 0%, rgba(35, 103, 145, 0.16), transparent 34%),
        radial-gradient(ellipse at 5% 45%, rgba(25, 91, 103, 0.10), transparent 30%),
        var(--bg);
}

.block-container {
    max-width: 1540px;
    padding-top: 1.65rem;
    padding-bottom: 3.5rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D192A 0%, #0A1422 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.13);
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
    letter-spacing: -0.8px;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: #7890AA !important;
    font-size: 0.70rem !important;
    font-weight: 800 !important;
    letter-spacing: 1.5px;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0.35rem;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 0.48rem 0.65rem;
    transition: background 160ms ease, border-color 160ms ease;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(85, 199, 243, 0.07);
    border-color: rgba(85, 199, 243, 0.16);
}

.hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 90% 20%, rgba(66, 214, 195, 0.15), transparent 28%),
        linear-gradient(115deg, #132A43 0%, #10263A 55%, #102E35 100%);
    border: 1px solid rgba(85, 199, 243, 0.22);
    border-radius: 20px;
    padding: clamp(1.35rem, 3vw, 2.15rem);
    margin-bottom: 1.4rem;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.16);
}

.hero::after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    right: -85px;
    bottom: -135px;
    border: 1px solid rgba(85, 199, 243, 0.18);
    border-radius: 50%;
    box-shadow: 0 0 0 24px rgba(85, 199, 243, 0.025),
                0 0 0 48px rgba(85, 199, 243, 0.02);
    pointer-events: none;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    color: #7DE7E0;
    font-size: 0.69rem;
    font-weight: 800;
    letter-spacing: 1.8px;
}

.hero-title {
    font-size: clamp(1.8rem, 3.4vw, 2.55rem);
    letter-spacing: -1.15px;
    line-height: 1.12;
    margin: 0.65rem 0 0.55rem;
}

.hero-description {
    max-width: 760px;
    color: #B8C9DA;
    line-height: 1.65;
    font-size: 0.94rem;
}

.section-label {
    margin: 0.25rem 0 0.8rem;
    color: #70D8F6;
    font-size: 0.68rem;
    letter-spacing: 1.65px;
}

h2, h3, h4 {
    letter-spacing: -0.45px !important;
}

[data-testid="stMetric"] {
    position: relative;
    overflow: hidden;
    background: linear-gradient(145deg, rgba(20, 35, 55, 0.98), rgba(14, 26, 43, 0.98));
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 15px;
    padding: 1.05rem 1.1rem;
    min-height: 112px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.10);
    transition: transform 160ms ease, border-color 160ms ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(85, 199, 243, 0.34);
}

[data-testid="stMetricLabel"] {
    color: #9BAEC3 !important;
    font-size: 0.77rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #F1F7FD !important;
    font-size: clamp(1.25rem, 2vw, 1.75rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.7px;
}

[data-testid="stMetricDelta"] {
    font-size: 0.76rem !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(16, 27, 45, 0.62);
    border-color: rgba(148, 163, 184, 0.15) !important;
    border-radius: 15px !important;
}

.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    min-height: 2.65rem;
    border-radius: 10px;
    font-weight: 700;
    border: 1px solid rgba(85, 199, 243, 0.24);
    box-shadow: 0 5px 16px rgba(2, 132, 199, 0.12);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    border-color: rgba(66, 214, 195, 0.55);
    transform: translateY(-1px);
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] {
    background: #142238;
    border-color: rgba(148, 163, 184, 0.20);
    border-radius: 10px;
}

[data-testid="stTabs"] button {
    font-weight: 650;
    padding-top: 0.7rem;
    padding-bottom: 0.7rem;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #62D1F4 !important;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stAlert"] {
    border-radius: 12px;
    border-width: 1px;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 12px;
    background: rgba(16, 27, 45, 0.45);
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: #8FA3BA !important;
}

hr {
    border-color: rgba(148, 163, 184, 0.14);
    margin: 1.35rem 0;
}

.property-card {
    background: linear-gradient(145deg, #14243A, #101C2E);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 14px;
    padding: 1.05rem 1.15rem;
    margin-bottom: 0.8rem;
    transition: border-color 160ms ease, transform 160ms ease;
}

.property-card:hover {
    border-color: rgba(66, 214, 195, 0.34);
    transform: translateY(-2px);
}

.property-card-title {
    color: #F2F7FC;
    font-size: 1.02rem;
    font-weight: 750;
}

.property-card-meta {
    color: #9BAEC3;
    font-size: 0.79rem;
    margin-top: 0.25rem;
}

.similarity {
    color: #55DCC7;
    font-weight: 800;
}

.footer {
    border-top: 1px solid rgba(148, 163, 184, 0.12);
    margin-top: 2.5rem;
    padding-top: 1.3rem;
    color: #71869F;
    letter-spacing: 0.5px;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }
    .hero {
        border-radius: 15px;
        padding: 1.25rem;
    }
    [data-testid="stMetric"] {
        min-height: 98px;
        padding: 0.85rem;
    }
}

</style>
""", unsafe_allow_html=True)


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

with st.sidebar:
    st.markdown(
        '''
        <div style="padding:0.25rem 0 0.7rem 0;">
          <div style="font-size:1.65rem;line-height:1;">🏡</div>
          <div style="font-size:1.12rem;font-weight:800;color:#F1F7FC;margin-top:0.55rem;letter-spacing:-0.4px;">
            Kathmandu Stays
          </div>
          <div style="font-size:0.64rem;font-weight:800;letter-spacing:1.55px;color:#68D8E8;margin-top:0.25rem;">
            ACCOMMODATION INTELLIGENCE
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Property Recommendations",
            "Price Prediction",
            "Data Analysis",
            "Model Evaluation",
        ],
        label_visibility="visible",
    )

    st.divider()

    st.markdown("**PROJECT INFORMATION**")
    st.caption("Kathmandu Airbnb ML Analysis")
    st.caption("Machine Learning Internship Project")
    st.caption("Python · Scikit-learn · Streamlit")

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

if page == "Overview":

    page_header(
        "MACHINE LEARNING ANALYTICS PLATFORM",
        "Accommodation Intelligence",
        "Explore Kathmandu accommodation sample data, discover similar "
        "properties and investigate experimental nightly price estimates.",
    )

    st.warning(
        "Data transparency: Property, availability and price datasets "
        "contain synthetic sample data. These results do not represent "
        "verified live Airbnb listings or market prices."
    )

    total = len(properties)
    published = len(published_data(properties))

    avg_price = numeric(
        properties["nightly_price_npr"]
    ).mean()

    avg_rating = numeric(
        properties["rating"]
    ).mean()

    st.markdown('<div class="section-label">KEY INDICATORS</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Properties", f"{total:,}")
    c2.metric("Published Properties", f"{published:,}")
    c3.metric("Average Nightly Price", money(avg_price))
    c4.metric("Average Rating", f"{avg_rating:.2f} / 5")

    st.divider()

    left, right = st.columns(2)

    with left:
        section_heading(
            "Property Types",
            "Distribution of accommodation types in the sample dataset.",
        )

        type_data = properties["property_type"].value_counts()
        st.bar_chart(type_data, color="#38BDF8")

    with right:
        section_heading(
            "Neighbourhood Price Comparison",
            "Ten neighbourhoods with the highest average sample prices.",
        )

        neighbourhood_prices = (
            properties.assign(
                nightly_price_npr=numeric(
                    properties["nightly_price_npr"]
                )
            )
            .groupby("neighbourhood")["nightly_price_npr"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(neighbourhood_prices, color="#2DD4BF")

    st.divider()

    section_heading(
        "Property Price Distribution",
        "Explore the distribution of listed sample nightly prices.",
    )

    price_series = numeric(
        properties["nightly_price_npr"]
    ).dropna()

    if not price_series.empty:
        bins = pd.cut(price_series, bins=10)
        distribution = bins.value_counts().sort_index()
        distribution.index = distribution.index.astype(str)
        st.bar_chart(distribution, color="#818CF8")

    st.divider()

    section_heading("Featured Sample Properties")

    featured = properties.sort_values(
        "rating", ascending=False
    ).head(8)

    display_columns = [
        "property_id",
        "property_name",
        "neighbourhood",
        "property_type",
        "room_type",
        "nightly_price_npr",
        "rating",
    ]

    st.dataframe(
        featured[display_columns],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
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
            use_container_width=True,
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
                    use_container_width=True,
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
            use_container_width=True,
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
                use_container_width=True,
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
                use_container_width=True,
                hide_index=True,
            )

            st.caption("Preview limited to the first 500 records.")


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    page_header(
        "MODEL PERFORMANCE",
        "Evaluate the Prediction Model",
        "Review saved test-set metrics, baseline comparison, "
        "cross-validation and prediction error patterns.",
    )

    if not metrics.empty and {"metric", "value"}.issubset(metrics.columns):

        metric_values = {
            str(row["metric"]).strip().lower(): float(row["value"])
            for _, row in metrics.iterrows()
        }

        def metric_lookup(*names):
            for name in names:
                if name.lower() in metric_values:
                    return metric_values[name.lower()]
            return None

        mae = metric_lookup("mae")
        rmse = metric_lookup("rmse")
        r2 = metric_lookup("r2", "r2_score", "r2 score")
        baseline_mae = metric_lookup("baseline mae")
        improvement = metric_lookup("mae improvement percentage")

        st.markdown(
            '<div class="section-label">TEST SET PERFORMANCE</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("MAE", money(mae) if mae is not None else "N/A")
        c2.metric("RMSE", money(rmse) if rmse is not None else "N/A")
        c3.metric("R² Score", f"{r2:.4f}" if r2 is not None else "N/A")
        c4.metric(
            "MAE Improvement",
            f"{improvement:.2f}%" if improvement is not None else "N/A",
        )

        if baseline_mae is not None:
            st.caption(
                f"Direct baseline MAE: {money(baseline_mae)}"
            )

    else:
        st.info("Price model metrics report is unavailable.")

    st.divider()

    section_heading("Model Comparison")

    if not price_comparison.empty:
        st.dataframe(
            price_comparison,
            use_container_width=True,
            hide_index=True,
        )

        numeric_columns = price_comparison.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            st.bar_chart(
                price_comparison.set_index(
                    price_comparison.columns[0]
                )[numeric_columns],
            )
    else:
        st.info("Model comparison report is unavailable.")

    st.divider()

    section_heading("Cross-Validation")

    if not cross_validation.empty:
        st.dataframe(
            cross_validation,
            use_container_width=True,
            hide_index=True,
        )

        numeric_columns = cross_validation.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            st.line_chart(
                cross_validation[numeric_columns],
                color="#38BDF8",
            )
    else:
        st.info("Cross-validation report is unavailable.")

    st.divider()

    section_heading("Prediction Error by Price Category")

    if not error_categories.empty:
        st.dataframe(
            error_categories,
            use_container_width=True,
            hide_index=True,
        )

        numeric_columns = error_categories.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            st.bar_chart(
                error_categories.set_index(
                    error_categories.columns[0]
                )[numeric_columns],
            )
    else:
        st.info("Price error category report is unavailable.")

    st.warning(
        "Interpretation limitation: The model uses the existing baseline "
        "nightly price as an input. Its strong evaluation scores should "
        "not be interpreted as independent price prediction accuracy. "
        "All property and price observations are synthetic sample data."
    )


# ============================================================
# 11. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        KATHMANDU AIRBNB ML ANALYSIS<br>
        Machine Learning Internship Project · Experimental Dataset
    </div>
    """,
    unsafe_allow_html=True,
)
