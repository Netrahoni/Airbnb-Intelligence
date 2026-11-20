# Kathmandu Airbnb ML Analysis

An end-to-end machine learning project for accommodation recommendation, nightly price prediction, and market analytics using a prepared Kathmandu accommodation dataset.

The project combines machine learning models, data analysis, model evaluation, and an interactive Streamlit dashboard. It is developed as an ML-focused internship project, not as a complete Airbnb booking platform.

---

## 1. Project Overview

Kathmandu Airbnb ML Analysis explores how machine learning can support accommodation discovery and pricing analysis in Kathmandu, Nepal.

The project focuses on three main areas:

1. **Property Recommendation System** – Recommends properties based on user preferences and identifies properties similar to a selected accommodation.
2. **Nightly Price Prediction System** – Estimates accommodation prices using property characteristics and stay-related features.
3. **Executive Analytics Dashboard** – Presents accommodation data, pricing patterns, tourism context, and model insights through an interactive interface.

The primary objective is to demonstrate the application of data preprocessing, feature engineering, machine learning, model evaluation, and data visualization in an accommodation analytics use case.

## 2. Machine Learning Modules

### 2.1 Property Recommendation System

The recommendation system helps users discover accommodation options based on property characteristics and preferences.

**Techniques used:**

- TF-IDF text vectorization
- Cosine similarity
- Text-based property comparison
- Weighted hybrid recommendation scoring
- Preference-based filtering and ranking

**Personalized recommendations consider:**

- Neighbourhood
- Property type
- Room type
- Budget
- Guest capacity
- Property quality

Budget, minimum rating, and guest capacity are applied as filters where supported. Other preferences contribute to recommendation ranking.

**Similar-property recommendations** compare a selected property with other available properties using property descriptions and characteristics.

The current prepared dataset contains 400 property records, of which 354 are published properties.

**Recommendation evaluation:**

The recommendation engine is evaluated using 25 personalized scenarios and 25 similar-property scenarios.

| Evaluation metric | Result |
|---|---:|
| Average personalized recommendation score | 63.28% |
| Budget compliance | 100.00% |
| Neighbourhood match | 77.60% |
| Property type match | 54.00% |
| Room type match | 72.00% |
| Guest capacity sufficiency | 100.00% |
| Average similar-property score | 62.91% |
| Same neighbourhood for similar properties | 87.60% |
| Same room type for similar properties | 78.80% |

These results are heuristic preference-match diagnostics. They are not ground-truth accuracy, precision, recall, or measured user satisfaction.

### 2.2 Nightly Price Prediction System

The price prediction module estimates nightly accommodation prices using a Gradient Boosting Regressor.

**Input features include:**

- City and neighbourhood
- Property type and room type
- Guest capacity
- Bedrooms and bathrooms
- Amenities
- Lead time
- Stay month
- Stay day
- Day of the week
- Baseline nightly price

The model is trained using prepared price observation data and evaluated with a property-level train/test split.

**Current evaluation results:**

| Metric | Result |
|---|---:|
| MAE | NPR 148.33 |
| RMSE | NPR 218.93 |
| R² Score | 0.9942 |
| Baseline MAE | NPR 158.85 |
| Improvement over baseline MAE | 6.62% |

**Dataset split:**

- Training: 808 observations from 320 properties
- Testing: 192 observations from 80 properties
- Overlapping property IDs: 0

The property-level split prevents the same property from appearing in both training and testing sets.

These results describe performance on the prepared sample dataset. They should not be interpreted as proof of real-world pricing accuracy.

## 3. Executive Analytics Dashboard

The project includes an interactive Streamlit dashboard that brings the analytical modules together.

**Dashboard sections:**

### Executive Analytics
- Key accommodation statistics
- Property and pricing indicators
- Market intelligence
- Neighbourhood price comparison
- Property-type pricing analysis
- Price distribution and quartile-based segments
- Price dispersion analysis
- Tourism demand context
- Availability and occupancy data coverage
- Featured property samples
- Interactive filters

### Property Recommendations
- Personalized property recommendations
- Similar-property discovery
- Preference-based filtering
- Recommendation scores
- Property information and comparison

### Price Prediction
- Interactive price prediction inputs
- Property and stay-related features
- Predicted nightly price output

### Data Analysis
- Prepared dataset exploration
- Descriptive statistics
- Data visualization and analytical insights

### Model Evaluation
- Model performance indicators
- Recommendation evaluation summaries
- Price prediction evaluation information

The dashboard is intended for analytical exploration and demonstration rather than live booking or transaction processing.

## 4. Dataset Information

The project uses multiple prepared datasets covering accommodation properties, price observations, availability, cancellations, and tourism indicators.

### Main processed datasets

| Dataset | Description |
|---|---|
| `properties_clean.csv` | Property information and characteristics |
| `price_observations_clean.csv` | Accommodation price observations |
| `availability_clean.csv` | Availability dataset |
| `booking_cancellations_clean.csv` | Booking cancellation benchmark data |
| `nepal_monthly_tourist_arrivals_clean.csv` | Monthly tourist arrivals in Nepal |
| `recommendation_features.csv` | Features prepared for recommendation |
| `price_prediction_features.csv` | Features prepared for price prediction |

The main processed datasets are stored in `data/processed/`.

### Important Data Limitations

The current property and price observation datasets contain synthetic sample data.

Additional limitations:

- The property records are not verified live Airbnb listings.
- The price observations do not represent verified live market prices.
- The booking cancellation dataset is based on the Hotel Booking Demand benchmark and is not directly connected to Kathmandu property IDs.
- The tourist arrivals dataset represents broader Nepal-wide tourism context, not Kathmandu-specific accommodation demand.
- The occupancy dataset currently contains schema information without usable occupancy observations.
- Review data does not currently contain usable observations.

Consequently, the project is intended for machine learning development, experimentation, demonstration, and academic analysis. Its outputs must not be treated as actual booking availability, verified accommodation prices, confirmed revenue, or measured market demand.

## 5. Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy, SciPy |
| Machine learning | Scikit-learn |
| Model serialization | Joblib |
| Data visualization | Matplotlib, Seaborn |
| Interactive dashboard | Streamlit |
| Development environment | Visual Studio Code |
| Version control | Git and GitHub |
| Data format | CSV |

## 6. Project Structure

```text
Kathmandu-Airbnb-ML-Analysis/
│
├── data/
│   └── processed/
│       ├── availability_clean.csv
│       ├── booking_cancellations_clean.csv
│       ├── nepal_monthly_tourist_arrivals_clean.csv
│       ├── price_observations_clean.csv
│       ├── price_prediction_features.csv
│       ├── properties_clean.csv
│       └── recommendation_features.csv
│
├── docs/
│   └── Kathmandu_Airbnb_ML_Presentation.pptx
│
├── models/
│   ├── property_recommender.joblib
│   └── price_predictor.joblib
│
├── reports/
│   ├── price_model_metrics.csv
│   ├── price_predictions.csv
│   └── other evaluation reports
│
├── scripts/
│   ├── evaluate_recommendation_quality.py
│   └── optimize_recommendation_weights.py
│
├── src/
│   ├── advanced_recommender.py
│   ├── booking_preprocessing.py
│   ├── prepare_ml_datasets.py
│   ├── train_price_model.py
│   ├── train_recommendation_model.py
│   └── validation and comparison scripts
│
├── notebooks/
│   ├── 01_data_audit_preprocessing/
│   ├── 02_price_prediction/
│   └── 03_accommodation_recommendation/
│
├── tests/
│   └── test_recommendations.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The numbered notebook folders organize the audit, price-prediction, and recommendation analyses. Shared datasets, source code, trained models, reports, and documentation are kept in their corresponding top-level folders. The data-audit notebook expects original input CSVs in `data/raw/`; those raw files are not included in the repository.

The `.venv/` directory is created locally during environment setup and is not part of the source-code structure committed to Git.

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or later, subject to dependency compatibility
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the repository

```bash
git clone https://github.com/Netrahoni/Kathmandu-Airbnb-ML-Analysis.git
cd Kathmandu-Airbnb-ML-Analysis
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 5: Verify dependencies

```bash
python -m pip check
```

## 8. Running the Project

Run commands from the repository root directory.

### Launch the Streamlit dashboard

```bash
python -m streamlit run app.py
```

Using `python -m streamlit` avoids stale Windows console launchers after a virtual environment is moved. Streamlit will start a local server and provide a browser address for the dashboard.

### Prepare machine learning datasets

```bash
python src/prepare_ml_datasets.py
```

### Train the recommendation model

```bash
python src/train_recommendation_model.py
```

### Train the price prediction model

```bash
python src/train_price_model.py
```

### Run price-model validation

```bash
python src/validate_price_model.py
```

Run the training and preparation scripts only when you intend to regenerate the corresponding data or model artifacts. Existing prepared datasets and saved models may be used for dashboard demonstration.

## 9. Testing and Model Evaluation

The project includes separate testing and evaluation workflows.

### Recommendation behavior tests

```bash
python tests/test_recommendations.py
```

The current recommendation test suite contains 16 tests covering:

- Personalized recommendation generation
- Similar-property recommendation generation
- Budget filtering
- Guest capacity filtering
- Minimum rating filtering
- Recommendation ordering
- Duplicate detection
- Invalid input handling
- Similar-property reference validation

**Latest test result: 16 passed, 0 failed.**

### Recommendation quality evaluation

```bash
python scripts/evaluate_recommendation_quality.py
```

This evaluates personalized and similar-property recommendations across multiple scenarios and saves evaluation reports in `reports/`.

### Recommendation weight optimization

```bash
python scripts/optimize_recommendation_weights.py
```

This script compares recommendation weight configurations using the project's heuristic evaluation approach.

### Python compilation check

```bash
python -m compileall -q app.py src scripts tests
```

### Dependency validation

```bash
python -m pip check
```

These checks help identify syntax issues, dependency conflicts, and unexpected recommendation behavior. They do not replace real-world validation using verified accommodation data and user feedback.

## 10. Current Development Status

- [x] Dataset preparation and cleaning
- [x] Data quality inspection
- [x] Recommendation feature engineering
- [x] Property recommendation system
- [x] Personalized recommendation functionality
- [x] Similar-property recommendation functionality
- [x] Recommendation behavior testing
- [x] Recommendation quality evaluation
- [x] Recommendation weight comparison
- [x] Price prediction feature engineering
- [x] Price prediction model comparison
- [x] Gradient Boosting price prediction model
- [x] Property-level model evaluation
- [x] Saved model inference testing
- [x] Streamlit dashboard implementation
- [x] Executive Analytics dashboard
- [x] Data analysis and visualization
- [x] Model evaluation reporting
- [ ] Further validation using verified real-world accommodation data
- [ ] Additional model improvements and user-based evaluation

## 11. Future Improvements

Potential improvements include:

- Collecting more reliable and verified accommodation data.
- Improving recommendation relevance through user feedback.
- Exploring additional recommendation algorithms.
- Evaluating price prediction on independent real-world observations.
- Adding more detailed temporal pricing analysis.
- Incorporating reliable Kathmandu-specific tourism and occupancy indicators.
- Improving model explainability and monitoring.
- Expanding automated testing and documentation.

## 12. Project Disclaimer

Kathmandu Airbnb ML Analysis is an academic and internship-oriented machine learning project developed for educational, analytical, and demonstration purposes.

The project uses prepared datasets, including synthetic sample data and benchmark datasets. Its recommendations, predictions, and analytical outputs are generated from these available datasets and trained models.

The results do not represent official Airbnb information, verified live accommodation listings, actual booking transactions, confirmed availability, or guaranteed market prices.

Model evaluation metrics reflect performance on the prepared datasets and evaluation scenarios. Real-world performance may differ significantly.

This project is not affiliated with, endorsed by, or officially connected to Airbnb.

---

## Author and Repository

**Project:** Kathmandu Airbnb ML Analysis
**Repository:** https://github.com/Netrahoni/Kathmandu-Airbnb-ML-Analysis

Developed as a machine learning internship project.