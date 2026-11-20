# Airbnb Intelligence

> **Turning Airbnb Data into Intelligent Decisions.**

**Kathmandu Accommodation Analytics & Machine Learning Platform**

An end-to-end machine learning and analytics project exploring accommodation listings, recommendation, price estimation, and market insights for Kathmandu, Nepal. It includes prepared datasets, ML workflows, evaluation reports, and an interactive Streamlit dashboard.

> **Scope:** Academic / internship ML analytics—not a live Airbnb booking platform. Some property and price datasets are synthetic sample data; benchmark datasets are not verified Kathmandu booking records.

## Overview

The project demonstrates how data science and machine learning can support accommodation discovery and pricing analysis.

### Main features
- **Property Recommendation System** — personalized recommendations and similar-property discovery using structured preferences, text features, and hybrid scoring.
- **Nightly Price Prediction** — estimates accommodation prices from listing characteristics and stay-related features.
- **Executive Analytics Dashboard** — interactive market, pricing, neighbourhood, property-type, and tourism-context visualizations.
- **Data Analysis & Model Evaluation** — dataset exploration, preprocessing, model comparisons, validation reports, and performance diagnostics.
- **Booking Cancellation Data Study** — exploratory analysis and preprocessing of a separate hotel-booking benchmark for a cancellation-prediction experiment.

## Dashboard Workspaces

| Workspace | Description |
|---|---|
| **Executive Analytics** | Accommodation KPIs, market and price insights, neighbourhood comparisons, property-type analysis, price distributions, tourism context, and filters |
| **Property Recommendations** | Personalized search, preference filtering, ranked recommendations, similar-property discovery, and comparisons |
| **Price Prediction** | Interactive property/stay inputs and estimated nightly price |
| **Data Analysis** | Dataset exploration, descriptive statistics, and visual analysis |
| **Model Evaluation** | Model metrics, comparisons, recommendation evaluation, and prediction diagnostics |

The dashboard is for exploration and demonstration. It does not process reservations or verify live availability.

## Machine Learning & Analytics

### 1. Property Recommendation System

The recommendation workflow combines listing information and user preferences.

**Techniques and features**
- TF-IDF text vectorization and cosine similarity
- Property descriptions and amenities
- Neighbourhood, property type, and room type matching
- Price / budget compatibility
- Guest capacity, bedrooms, and bathrooms
- Rating and review-count quality signals
- Preference filters and weighted ranking

The prepared dataset contains **400 property records**, including **354 published properties**.

The repository includes scenario-based recommendation evaluation and weight-comparison scripts. Reported scores are heuristic preference-match diagnostics—not ground-truth precision, recall, recommendation accuracy, or user satisfaction.

### 2. Nightly Price Prediction

The current price pipeline uses a **Gradient Boosting Regressor** with categorical, numerical, and amenity-text features.

Features include city, neighbourhood, property type, room type, capacity, bedrooms, bathrooms, amenities, lead time, stay month/day/weekday, and baseline nightly price.

**Target:** <code>base_price_npr</code>

**Reported evaluation:**

| Metric | Result |
|---|---:|
| MAE | NPR 148.33 |
| RMSE | NPR 218.93 |
| R² | 0.9942 |
| Baseline MAE | NPR 158.85 |
| MAE improvement over baseline | 6.62% |

**Property-level split:**
- Training: 808 observations from 320 properties
- Testing: 192 observations from 80 properties
- Property IDs shared between train and test: **0**

These results are based on prepared sample data and do not establish real-market pricing accuracy.

### 3. Booking Cancellation Benchmark — Exploratory Work

Cancellation-data inspection and preprocessing are also included for the **Hotel Booking Demand** benchmark.

- Original dataset: **119,390 records**, **25 columns**
- Source domain: Portugal hotel bookings—not Kathmandu Airbnb bookings
- Exploratory cancellation rate: approximately **37.04%**
- Preprocessing removes logically invalid rows and builds booking-time features
- The checked preprocessing workflow reports **119,209 rows** after invalid-row removal

This is a separate benchmark experiment. It must not be represented as a cancellation model trained on Kathmandu bookings. No real Nepal COD/no-show labels are implied.

### 4. Tourism Context

Monthly Nepal tourist-arrival data is included for context. National arrivals are not equivalent to Kathmandu occupancy, accommodation bookings, or local room-night demand.

Occupancy and review datasets should only be used when valid observations are available; empty or schema-only files are not treated as real data.

## Data

### Main processed datasets

| File | Description |
|---|---|
| <code>properties_clean.csv</code> | Property attributes, prices, ratings, amenities, descriptions, and publication fields |
| <code>price_observations_clean.csv</code> | Price observations |
| <code>availability_clean.csv</code> | Availability/calendar information |
| <code>booking_cancellations_clean.csv</code> | Hotel-booking cancellation benchmark |
| <code>nepal_monthly_tourist_arrivals_clean.csv</code> | Monthly national tourist-arrival context |
| <code>recommendation_features.csv</code> | Features prepared for recommendation |
| <code>price_prediction_features.csv</code> | Features prepared for price prediction |

Processed datasets are stored in <code>data/processed/</code>.

### Data limitations and provenance
- Property records and price observations include **synthetic sample data**.
- Listings are not verified live Airbnb properties.
- Price observations are not verified live market prices.
- The cancellation benchmark is not linked to Kathmandu property IDs.
- Nepal-wide tourist arrivals are not Kathmandu-specific accommodation demand.
- Occupancy and review data currently lack usable observations.
- Results are suitable for development, testing, demonstration, and academic analysis—not operational decisions.

Do not interpret predictions as actual prices, recommendations as verified listings, or availability as live inventory.

## Repository Structure

    Kathmandu-Airbnb-ML-Analysis/
    ├── app.py
    ├── requirements.txt
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── models/
    │   ├── property_recommender.joblib
    │   └── price_predictor.joblib
    ├── notebooks/
    │   ├── Project_Report_and_Visual_Analysis.ipynb
    │   ├── 01_data_audit_preprocessing/
    │   ├── 02_price_prediction/
    │   └── 03_accommodation_recommendation/
    ├── reports/
    │   ├── price_model_metrics.csv
    │   ├── price_predictions.csv
    │   └── analysis and evaluation reports
    ├── scripts/
    │   ├── evaluate_recommendation_quality.py
    │   └── optimize_recommendation_weights.py
    ├── src/
    │   ├── advanced_recommender.py
    │   ├── booking_preprocessing.py
    │   ├── prepare_ml_datasets.py
    │   ├── train_price_model.py
    │   ├── train_recommendation_model.py
    │   └── validation and analysis scripts
    ├── tests/
    │   └── test_recommendations.py
    └── README.md

Raw files are expected in <code>data/raw/</code> for the data-audit workflow. Do not commit private, restricted, or personally identifiable data.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy, SciPy |
| Machine learning | Scikit-learn |
| Model persistence | Joblib |
| Visualization | Matplotlib, Seaborn |
| Interactive dashboard | Streamlit |
| Development | Visual Studio Code |
| Version control | Git, GitHub |
| Data storage | CSV |

## Installation

**Prerequisites:** Python 3.10 or later (check compatibility with pinned dependencies), Git, and optionally Visual Studio Code.

### 1. Clone the repository

    git clone https://github.com/Netrahoni/Kathmandu-Airbnb-ML-Analysis.git
    cd Kathmandu-Airbnb-ML-Analysis

### 2. Create a virtual environment

    python -m venv .venv

### 3. Activate it

**Windows PowerShell:**

    .\.venv\Scripts\Activate.ps1

**Windows Command Prompt:**

    .\.venv\Scripts\activate.bat

**macOS / Linux:**

    source .venv/bin/activate

### 4. Install dependencies

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

### 5. Validate the environment

    python -m pip check

## Project Report Notebook

Open [`notebooks/Project_Report_and_Visual_Analysis.ipynb`](notebooks/Project_Report_and_Visual_Analysis.ipynb) in Jupyter Notebook or VS Code. Run cells from top to bottom to generate a concise project report, workflow diagram, dataset audit, dashboard-style plots, model evaluation visualizations, and outcome/limitations summary from the local processed data and reports. Missing artifacts are handled with notices rather than fabricated chart values.

Install Jupyter if needed:

    python -m pip install notebook

## Run the Dashboard

From the repository root:

    python -m streamlit run app.py

Streamlit will print a local URL. Open it in your browser to explore the application. Using <code>python -m streamlit</code> is recommended on Windows because it avoids stale launchers after moving or recreating a virtual environment.

## Model & Dataset Workflows

Run commands from the project root and only when you intend to regenerate the corresponding artifacts.

### Prepare model features

    python src/prepare_ml_datasets.py

### Train the recommendation model

    python src/train_recommendation_model.py

### Train the price-prediction model

    python src/train_price_model.py

### Validate the price model

    python src/validate_price_model.py

### Evaluate recommendation scenarios

    python scripts/evaluate_recommendation_quality.py

### Compare recommendation weights

    python scripts/optimize_recommendation_weights.py

## Testing and Quality Checks

### Recommendation tests

    python tests/test_recommendations.py

The current repository README reports **16 passing tests**. The suite covers recommendation generation and similar-property discovery, filtering, ranking, duplicate handling, invalid inputs, and property-reference validation. Re-run locally to confirm the result for your checkout.

### Compile Python files

    python -m compileall -q app.py src scripts tests

### Validate dependencies

    python -m pip check

These checks help software quality but do not replace validation using verified real-world accommodation data or actual user feedback.

## Development Status

- [x] Data inspection and preprocessing workflows
- [x] Recommendation feature engineering
- [x] Personalized recommendation system
- [x] Similar-property recommendations
- [x] Recommendation scenario evaluation
- [x] Recommendation weight comparison
- [x] Price-prediction feature engineering
- [x] Gradient Boosting price model
- [x] Property-level train/test split
- [x] Price-model evaluation reports
- [x] Saved-model inference
- [x] Streamlit dashboard
- [x] Executive Analytics workspace
- [x] Data Analysis workspace
- [x] Model Evaluation workspace
- [x] Booking-cancellation benchmark inspection and preprocessing
- [ ] Validation using verified, permitted Kathmandu accommodation data
- [ ] Local occupancy and booking-history collection
- [ ] Real review data and validated sentiment analysis
- [ ] Further user-based recommendation evaluation

## Future Work

- Replace synthetic property and pricing samples with verified, legally collected data.
- Improve recommendation quality with user feedback and unbiased evaluation.
- Validate price estimates against independent real-world observations.
- Add more robust temporal and geographic validation.
- Incorporate Kathmandu-specific occupancy and booking signals when available.
- Develop and evaluate cancellation-risk modelling with appropriate booking-time features and trustworthy labels.
- Add review sentiment and aspect analysis after collecting permitted, validated review data.
- Improve explainability, monitoring, reproducibility, and automated testing.

## Disclaimer

This repository is an academic and internship-oriented project for educational, analytical, and demonstration purposes.

The project uses prepared datasets, including synthetic sample data and external benchmark data. Recommendations, predictions, and visualizations are generated from those data and trained models. They do not represent official Airbnb information, verified live listings, actual transactions, confirmed availability, or guaranteed market prices.

The reported evaluation metrics describe performance on the available prepared datasets and evaluation scenarios. Real-world performance may differ.

This project is not affiliated with, endorsed by, or officially connected to Airbnb.

## Author

**Project:** Airbnb Intelligence  
**Repository:** [Netrahoni/Kathmandu-Airbnb-ML-Analysis](https://github.com/Netrahoni/Kathmandu-Airbnb-ML-Analysis)  
**Purpose:** Machine Learning Internship Project
