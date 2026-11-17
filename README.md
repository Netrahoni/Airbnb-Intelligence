# Kathmandu Airbnb ML Analysis

A machine learning project focused on accommodation recommendation and nightly price prediction for properties in Kathmandu, Nepal.

## 1. Project Overview

This project explores how machine learning can support accommodation discovery and price analysis in Kathmandu.

The current scope focuses on two machine learning modules:

1. Property Recommendation System
2. Nightly Price Prediction System

This is an ML-focused internship project, not a complete Airbnb booking website.

## 2. Machine Learning Modules

### 2.1 Property Recommendation System

The recommendation system uses:

- TF-IDF text vectorization
- Cosine similarity
- Property descriptions and characteristics

It compares property information and identifies similar properties for recommendation.

The current model uses 354 published properties from the prepared sample dataset.

### 2.2 Nightly Price Prediction System

The price prediction model uses a Gradient Boosting Regressor.

Input features include:

- City and neighbourhood
- Property type and room type
- Capacity, bedrooms, and bathrooms
- Amenities
- Lead time
- Stay month, day, and day of the week
- Baseline nightly price

The model estimates a property's nightly accommodation price.

## 3. Price Prediction Evaluation

The current evaluation uses a property-level train/test split to prevent the same property from appearing in both training and testing sets.

| Metric | Result |
|---|---:|
| MAE | NPR 148.33 |
| RMSE | NPR 218.93 |
| R² Score | 0.9942 |
| Baseline MAE | NPR 158.85 |
| Improvement over baseline MAE | 6.62% |

Dataset split:

- Training: 808 observations from 320 properties
- Testing: 192 observations from 80 properties
- Overlapping property IDs: 0

These results describe performance on the prepared sample dataset. They should not be interpreted as proof of real-world pricing accuracy.

## 4. Dataset Information

The project uses multiple prepared datasets:

- Property information
- Availability observations
- Price observations
- Monthly tourist arrivals in Nepal
- Booking cancellation benchmark data

### Important Data Limitations

The current property, availability, and price datasets contain synthetic sample data.

The booking cancellation dataset is based on the Hotel Booking Demand benchmark and is not directly connected to Kathmandu property IDs.

The tourist arrivals dataset provides broader tourism context. It does not directly measure Airbnb occupancy.

Review and occupancy datasets currently do not contain usable observations.

Therefore, the project is intended for machine learning development, experimentation, and demonstration. It does not represent verified live Airbnb market data.

## 5. Technology Stack

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Streamlit (for the planned dashboard)

## 6. Project Structure

```text
Kathmandu-Airbnb-ML-Analysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── property_recommender.joblib
│   └── price_predictor.joblib
│
├── reports/
│
├── src/
│   ├── prepare_ml_datasets.py
│   ├── train_recommendation_model.py
│   ├── train_price_model.py
│   └── test_price_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

Additional directories may contain earlier standalone preprocessing and model development work.

## 7. Installation

Clone the repository:

```bash
git clone https://github.com/Netrahoni/Kathmandu-Airbnb-ML-Analysis.git
cd Kathmandu-Airbnb-ML-Analysis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 8. Running the Project

Prepare the machine learning datasets:

```bash
python src/prepare_ml_datasets.py
```

Train the recommendation model:

```bash
python src/train_recommendation_model.py
```

Train the price prediction model:

```bash
python src/train_price_model.py
```

Test the saved price prediction model:

```bash
python src/test_price_model.py
```

Run these scripts from the repository root directory.

## 9. Current Development Status

- [x] Dataset preparation
- [x] Data quality inspection
- [x] Recommendation feature engineering
- [x] Property recommendation model
- [x] Price prediction feature engineering
- [x] Price prediction model comparison
- [x] Price model training
- [x] Group-based evaluation
- [x] Saved model inference test
- [ ] Streamlit dashboard
- [ ] Further model improvements
- [ ] Final project documentation

## 10. Future Improvements

- Build an interactive Streamlit dashboard.
- Add more reliable real-world accommodation data.
- Improve price prediction for different price categories.
- Add more recommendation filters.
- Expand model evaluation and validation.
- Integrate additional tourism and accommodation indicators.

## 11. Project Disclaimer

This project is developed for