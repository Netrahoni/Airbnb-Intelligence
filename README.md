# Kathmandu Airbnb ML Analysis

A machine learning project focused on accommodation recommendation
and nightly price prediction for properties in Kathmandu, Nepal.

## Project Overview

This project explores how machine learning can support accommodation
discovery and price analysis in Kathmandu.

The current scope focuses on two machine learning modules:

1. Property Recommendation System
2. Nightly Price Prediction System

This is an ML-focused internship project, not a complete Airbnb
booking website.

## Machine Learning Modules

### 1. Property Recommendation

Uses TF-IDF text vectorization and cosine similarity to recommend
similar properties based on property information and descriptions.

### 2. Nightly Price Prediction

Uses a Gradient Boosting Regressor to estimate nightly accommodation
prices using property characteristics, stay information, and
baseline nightly prices.

## Model Evaluation

The current price prediction model achieved:

| Metric | Result |
|---|---:|
| MAE | NPR 148.33 |
| RMSE | NPR 218.93 |
| R² Score | 0.9942 |
| Baseline MAE | NPR 158.85 |
| Improvement over baseline MAE | 6.62% |

The evaluation uses a property-level train/test split to prevent
the same property from appearing in both sets.

## Dataset Information

The project uses multiple datasets for property information,
availability, price observations, cancellations, and tourism trends.

**Important:** The property, availability, and price observation
datasets currently contain synthetic sample data. The cancellation
dataset is a separate hotel-booking dataset and is not directly
merged with Kathmandu property records. Tourism arrivals data is
used only for broader tourism context.

Consequently, the reported model performance demonstrates
performance on the available sample data, not verified real-world
Airbnb prediction accuracy.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Streamlit (planned dashboard)

## Project Structure

```text
Kathmandu_Airbnb_ML_Analysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
├── src/
├── models/
├── reports/
├── app/
│
├── requirements.txt
└── README.md