
from pathlib import Path
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

st.set_page_config(
    page_title="Kathmandu Airbnb ML Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. LOAD MODELS AND DATA
# --------------------------------------------------

@st.cache_resource
def load_models():
    recommendation_model = joblib.load(
        MODEL_DIR / "property_recommender.joblib"
    )
    price_model = joblib.load(
        MODEL_DIR / "price_predictor.joblib"
    )
    return recommendation_model, price_model


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
    recommendation_model, price_model = load_models()

    properties = load_csv("properties_clean.csv")
    availability = load_csv("availability_clean.csv")
    price_observations = load_csv("price_observations_clean.csv")

    recommendation_properties = recommendation_model["properties"]
    similarity_matrix = recommendation_model["similarity_matrix"]

    metrics = load_report("price_model_metrics.csv")
    price_comparison = load_report("price_model_comparison.csv")
    cross_validation = load_report("price_model_cross_validation.csv")
    error_categories = load_report("price_error_by_category.csv")

except Exception as error:
    st.error(f"Unable to load project resources: {error}")
    st.stop()


# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def format_npr(value):
    return f"NPR {value:,.2f}"


def get_recommendations(property_index, number=5):
    scores = similarity_matrix[property_index]

    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indices:
        if index == property_index:
            continue

        property_info = recommendation_properties.iloc[index]

        results.append({
            "Property ID": property_info["property_id"],
            "Property Name": property_info["property_name"],
            "Neighbourhood": property_info["neighbourhood"],
            "Property Type": property_info["property_type"],
            "Room Type": property_info["room_type"],
            "Nightly Price (NPR)": property_info["nightly_price_npr"],
            "Rating": property_info["rating"],
            "Similarity (%)": round(float(scores[index]) * 100, 2),
        })

        if len(results) == number:
            break

    return pd.DataFrame(results)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🏠 Kathmandu Airbnb")
st.sidebar.caption("Machine Learning Analysis Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Property Recommendations",
        "Price Prediction",
        "Data Analysis",
        "Model Evaluation",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Kathmandu Airbnb ML Analysis")
st.sidebar.caption("Educational / Internship Project")


# --------------------------------------------------
# 5. OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    st.title("Kathmandu Airbnb ML Analysis")
    st.subheader("Machine Learning Dashboard")

    st.write(
        "Explore accommodation sample data, discover similar properties, "
        "and estimate nightly accommodation prices using trained ML models."
    )

    st.warning(
        "Data disclaimer: The property, availability, and price datasets "
        "currently contain synthetic sample data. Results are experimental "
        "and do not represent verified live Airbnb market information."
    )

    total_properties = len(properties)

    if "published" in properties.columns:
        published_count = int(
            properties["published"].astype(str).str.lower().isin(
                ["true", "1"]
            ).sum()
        )
    else:
        published_count = total_properties

    average_price = (
        safe_numeric(properties["nightly_price_npr"]).mean()
        if "nightly_price_npr" in properties.columns
        else 0
    )

    average_rating = (
        safe_numeric(properties["rating"]).mean()
        if "rating" in properties.columns
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Properties", f"{total_properties:,}")
    col2.metric("Published Properties", f"{published_count:,}")
    col3.metric("Average Nightly Price", format_npr(average_price))
    col4.metric("Average Rating", f"{average_rating:.2f} / 5")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Properties by Type")

        if "property_type" in properties.columns:
            type_counts = properties["property_type"].value_counts()
            st.bar_chart(type_counts)

    with right:
        st.subheader("Average Price by Neighbourhood")

        if {"neighbourhood", "nightly_price_npr"}.issubset(
            properties.columns
        ):
            neighbourhood_prices = (
                properties.groupby("neighbourhood")["nightly_price_npr"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )

            st.bar_chart(neighbourhood_prices)

    st.divider()

    st.subheader("Sample Properties")

    display_columns = [
        "property_id",
        "property_name",
        "neighbourhood",
        "property_type",
        "room_type",
        "nightly_price_npr",
        "rating",
    ]

    available_columns = [
        column for column in display_columns
        if column in properties.columns
    ]

    st.dataframe(
        properties[available_columns].head(10),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# 6. PROPERTY RECOMMENDATION PAGE
# --------------------------------------------------

elif page == "Property Recommendations":

    st.title("Property Recommendation System")

    st.write(
        "Select a property to discover similar accommodation "
        "using TF-IDF and cosine similarity."
    )

    recommendation_names = (
        recommendation_properties["property_id"].astype(str)
        + " - "
        + recommendation_properties["property_name"].astype(str)
    )

    selected_name = st.selectbox(
        "Choose a property",
        recommendation_names.tolist(),
    )

    selected_index = recommendation_names.tolist().index(selected_name)

    selected_property = recommendation_properties.iloc[selected_index]

    st.subheader("Selected Property")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Nightly Price",
        format_npr(selected_property["nightly_price_npr"]),
    )

    col2.metric(
        "Rating",
        f"{selected_property['rating']} / 5",
    )

    col3.metric(
        "Review Count",
        f"{selected_property['review_count']:,}",
    )

    st.write(
        f"**Neighbourhood:** {selected_property['neighbourhood']}"
    )
    st.write(
        f"**Property Type:** {selected_property['property_type']}"
    )
    st.write(
        f"**Room Type:** {selected_property['room_type']}"
    )
    st.write(
        f"**Description:** {selected_property['description']}"
    )

    number = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5,
    )

    if st.button("Find Similar Properties", type="primary"):

        recommendations = get_recommendations(
            selected_index,
            number,
        )

        st.subheader("Recommended Properties")

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Similarity percentages are calculated from the trained "
            "TF-IDF and cosine similarity model. They are not booking "
            "probabilities or customer preference probabilities."
        )


# --------------------------------------------------
# 7. NIGHTLY PRICE PREDICTION PAGE
# --------------------------------------------------

elif page == "Price Prediction":

    st.title("Nightly Price Prediction")

    st.write(
        "Estimate a property's nightly price using the trained "
        "Gradient Boosting prediction pipeline."
    )

    st.info(
        "The model uses the selected property's existing baseline nightly "
        "price as one of its input features."
    )

    published_properties = properties.copy()

    if "published" in published_properties.columns:
        published_properties = published_properties[
            published_properties["published"].astype(str).str.lower().isin(
                ["true", "1"]
            )
        ]

    property_labels = (
        published_properties["property_id"].astype(str)
        + " - "
        + published_properties["property_name"].astype(str)
    )

    with st.form("price_prediction_form"):

        selected_label = st.selectbox(
            "Select a property",
            property_labels.tolist(),
        )

        selected_position = property_labels.tolist().index(selected_label)
        selected = published_properties.iloc[selected_position]

        st.write(
            f"**Current baseline price:** "
            f"{format_npr(selected['nightly_price_npr'])}"
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
                "Planned stay date",
                value=date.today(),
            )

        submitted = st.form_submit_button(
            "Predict Nightly Price",
            type="primary",
        )

    if submitted:

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

            if not np.isfinite(prediction):
                raise ValueError("The model returned an invalid prediction.")

            baseline = float(selected["nightly_price_npr"])

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Nightly Price",
                format_npr(prediction),
            )

            col2.metric(
                "Baseline Nightly Price",
                format_npr(baseline),
                delta=format_npr(prediction - baseline),
            )

            st.caption(
                "This is an experimental model estimate based on the "
                "available sample data, not a live market quotation."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# 8. DATA ANALYSIS PAGE
# --------------------------------------------------

elif page == "Data Analysis":

    st.title("Accommodation Data Analysis")

    st.write(
        "Explore property characteristics, price distributions, "
        "availability records, and price observations."
    )

    tab1, tab2, tab3 = st.tabs([
        "Properties",
        "Availability",
        "Price Observations",
    ])

    with tab1:

        st.subheader("Property Dataset")

        st.write(f"Total records: {len(properties):,}")

        if "neighbourhood" in properties.columns:
            st.subheader("Properties by Neighbourhood")
            st.bar_chart(properties["neighbourhood"].value_counts())

        if "nightly_price_npr" in properties.columns:
            st.subheader("Nightly Price Distribution")

            price_data = safe_numeric(
                properties["nightly_price_npr"]
            ).dropna()

            st.bar_chart(
                price_data.value_counts(bins=15).sort_index()
            )

        st.dataframe(
            properties,
            use_container_width=True,
            hide_index=True,
        )

    with tab2:

        st.subheader("Availability Dataset")

        st.write(f"Total availability records: {len(availability):,}")

        if not availability.empty:

            st.dataframe(
                availability.head(100),
                use_container_width=True,
                hide_index=True,
            )

            st.caption("Showing the first 100 records.")

        else:
            st.info("Availability dataset is not available.")

    with tab3:

        st.subheader("Price Observation Dataset")

        st.write(
            f"Total price observation records: "
            f"{len(price_observations):,}"
        )

        if not price_observations.empty:

            st.dataframe(
                price_observations.head(100),
                use_container_width=True,
                hide_index=True,
            )

            st.caption("Showing the first 100 records.")

        else:
            st.info("Price observation dataset is not available.")


# --------------------------------------------------
# 9. MODEL EVALUATION PAGE
# --------------------------------------------------

elif page == "Model Evaluation":

    st.title("Model Evaluation")

    st.write(
        "Review the saved evaluation results of the trained "
        "nightly price prediction model."
    )

    st.subheader("Price Prediction Metrics")

    if not metrics.empty:

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True,
        )

        for _, row in metrics.iterrows():

            metric_name = str(row["metric"]).lower()

            if metric_name == "mae":
                st.metric(
                    "Mean Absolute Error",
                    format_npr(float(row["value"])),
                )

            elif metric_name == "rmse":
                st.metric(
                    "Root Mean Squared Error",
                    format_npr(float(row["value"])),
                )

            elif metric_name in ["r2", "r2_score", "r2 score"]:
                st.metric(
                    "R² Score",
                    f"{float(row['value']):.4f}",
                )

    else:
        st.info("Price model metrics report is unavailable.")

    st.divider()

    st.subheader("Model Comparison")

    if not price_comparison.empty:
        st.dataframe(
            price_comparison,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Cross-Validation Results")

    if not cross_validation.empty:
        st.dataframe(
            cross_validation,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Prediction Error by Price Category")

    if not error_categories.empty:
        st.dataframe(
            error_categories,
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "Evaluation results are based on synthetic sample observations. "
        "They do not establish real-world predictive performance."
    )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Kathmandu Airbnb ML Analysis | Machine Learning Internship Project"
)
