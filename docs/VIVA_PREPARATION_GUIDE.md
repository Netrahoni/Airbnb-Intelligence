# Airbnb Intelligence — Complete Project Viva Preparation Guide

**Purpose:** 200-mark project viva preparation, from beginner concepts to implementation-level and advanced questions.  
**Project:** Airbnb Intelligence  
**Repository:** https://github.com/Netrahoni/Airbnb-Intelligence  
**Application:** Python + Streamlit  
**Domain:** Kathmandu accommodation analytics and machine learning

> **Read this first — scope and honesty.** This is an academic analytics and demonstration project, not a booking platform. Prepared property and price data include synthetic samples. The dashboard does not show verified live Airbnb listings, prices, bookings, or availability. The cancellation dataset is an external Portugal hotel-booking benchmark, not Kathmandu Airbnb booking data. Nepal-wide tourist arrivals are contextual information, not a direct measurement of Kathmandu occupancy or demand.

---

# Part 1 — Your 60-second project introduction

## 1.1 Memorise this answer

“Good morning respected teachers. My project is **Airbnb Intelligence**, an academic machine-learning and data-analytics application focused on accommodation data for Kathmandu, Nepal.

The main objective is to analyse accommodation-related data and demonstrate how machine learning can support property discovery and nightly-price estimation. I developed the workflow using Python, Pandas, NumPy and Scikit-learn, and presented the results through an interactive Streamlit dashboard.

The application has five workspaces: Executive Analytics, Property Recommendations, Price Prediction, Data Analysis and Model Evaluation.

The recommendation module uses a hybrid approach. It combines TF-IDF and cosine similarity for text relevance with weighted matching signals such as neighbourhood, budget, property type, quality and guest capacity. The price module uses a Gradient Boosting Regressor inside a preprocessing pipeline. I evaluated it using MAE, RMSE and R-squared and compared it with a simple baseline.

The project also includes data preparation, exploratory analysis, evaluation scripts, automated tests and documentation. An important limitation is that prepared property and price records include synthetic samples. Therefore, the results demonstrate the workflow and model behaviour; they should not be treated as verified live market predictions.”

## 1.2 One-line answers

- **Project name:** Airbnb Intelligence.
- **Project type:** Machine-learning and data-analytics application.
- **Main domain:** Kathmandu accommodation data.
- **Main language:** Python.
- **Dashboard framework:** Streamlit.
- **ML library:** Scikit-learn.
- **Data libraries:** Pandas and NumPy.
- **Visualisation:** Matplotlib (and any visualisation utilities actually used in the dashboard).
- **Model persistence:** Joblib.
- **Main ML tasks:** Recommendation/ranking and regression.
- **Recommendation technique:** Hybrid content-based recommendation.
- **Price algorithm:** Gradient Boosting Regressor.
- **Deployment interface:** Interactive Streamlit dashboard.

---

# Part 2 — Understand the complete project

## 2.1 Problem statement

Accommodation decisions depend on several attributes: location, budget, property type, room type, amenities, guest capacity, ratings and descriptions. Manually comparing many properties is time-consuming. Data may also be distributed across different files and require cleaning before meaningful analysis.

The project addresses these issues by preparing accommodation datasets, analysing them, ranking properties according to preferences, estimating a prepared price target and presenting the results in a single dashboard.

## 2.2 Objectives

1. Prepare and validate accommodation-related datasets.
2. Perform exploratory data analysis (EDA).
3. Examine neighbourhood, property-type and price patterns.
4. Build a preference-based property recommendation engine.
5. Build a regression pipeline for the prepared nightly-price target.
6. Evaluate model behaviour with suitable metrics and a baseline.
7. Display insights and model outputs in a Streamlit interface.
8. Practise modular programming, testing, reproducibility and documentation.

## 2.3 Five dashboard workspaces

| Workspace | What to explain in viva |
|---|---|
| Executive Analytics | Summary KPIs, listing inventory, price distributions, neighbourhood comparisons, property-type analysis and tourism context |
| Property Recommendations | Preference-based recommendations, similar-property discovery, filters and ranked results |
| Price Prediction | User inputs, feature preparation and estimated target price |
| Data Analysis | Dataset inspection, descriptive statistics and visual analysis |
| Model Evaluation | Saved regression metrics, baseline comparison, prediction diagnostics and recommendation evaluation |

**How to demonstrate:** Open each workspace, explain its purpose, identify its input data, describe the output, and state one limitation. Do not just click through the dashboard silently.

## 2.4 End-to-end workflow

1. **Data collection / source preparation:** Bring together prepared property, price, availability, benchmark and tourism-context datasets.
2. **Data inspection:** Check columns, data types, missing values, duplicates, invalid values and dataset shapes.
3. **Cleaning:** Convert data types, handle missing values, remove or flag unusable records and standardise fields.
4. **Feature engineering:** Derive date-related variables, prepare recommendation text and create model-ready tables.
5. **EDA:** Use summaries and charts to understand distributions and relationships.
6. **Model training:** Train the recommendation representation and the price-regression pipeline.
7. **Evaluation:** Compare held-out price predictions with actual targets and a baseline; run recommendation scenarios and diagnostics.
8. **Persistence:** Save trained artefacts using Joblib and reports as files.
9. **Application:** Load prepared data and models in Streamlit, accept user inputs and display results.
10. **Testing and documentation:** Run tests, validate important behaviours and document setup, assumptions and limitations.

## 2.5 Architecture you can draw

```text
Prepared / Raw Datasets
        |
        v
Data Loading and Validation
        |
        v
Cleaning and Feature Engineering
        |
        +------------------------------+
        |                              |
        v                              v
Recommendation Features          Price Features
        |                              |
        v                              v
TF-IDF + Hybrid Ranking          Preprocessing Pipeline
        |                              |
        v                              v
Recommendation Engine           Gradient Boosting Regressor
        |                              |
        +---------------+--------------+
                        |
                        v
              Saved Model Artefacts
                  (Joblib)
                        |
                        v
              Streamlit Application
                        |
        +---------------+----------------+
        |               |                |
        v               v                v
 Executive          Recommendations   Price Prediction
 Analytics          and Data Analysis and Evaluation
                        |
                        v
             Results, Charts and Reports
```

---

# Part 3 — Dataset knowledge: know what your data means

## 3.1 Important prepared data

The repository documentation describes these prepared datasets:

| Dataset / file | Purpose |
|---|---|
| `properties_clean.csv` | Property attributes, location, prices, ratings, amenities, descriptions and publication status |
| `price_observations_clean.csv` | Prepared observations of accommodation prices |
| `availability_clean.csv` | Calendar/availability-related records |
| `booking_cancellations_clean.csv` | External hotel-booking cancellation benchmark |
| `nepal_monthly_tourist_arrivals_clean.csv` | National tourism context |
| `recommendation_features.csv` | Model-ready property information for recommendations |
| `price_prediction_features.csv` | Model-ready rows for price regression |

Previously documented dataset scale: 400 property records, including 354 published properties; approximately 1,000 price observations; 36,000 availability records; and 360 monthly tourism-context records. The cancellation benchmark originally contains 119,390 rows and 25 columns, with preprocessing documented separately. Treat these as documented project figures and verify the current local CSVs before quoting any changed counts.

## 3.2 What is a dataset?

A dataset is a structured collection of observations. In a table:
- A **row** is one observation or record.
- A **column** is one variable or feature.
- A **feature** is an input used for analysis or prediction.
- A **target/label** is the value a supervised model learns to predict.
- A **record ID** identifies an entity, such as a property.

Example:

| property_id | neighbourhood | capacity | nightly_price_npr |
|---|---|---:|---:|
| KTM001 | Thamel | 2 | 3500 |
| KTM002 | Boudha | 4 | 5000 |

Here, neighbourhood, capacity and price are columns; each property is a row.

## 3.3 Data types

- **Integer:** whole numbers, e.g. guest capacity.
- **Float:** decimal values, e.g. rating or price.
- **String/object:** text, e.g. neighbourhood or description.
- **Boolean:** true/false, e.g. published status.
- **Datetime:** date/time values, e.g. observation date and stay date.

Why data types matter: arithmetic on strings can fail, text models need text input, and date features must be extracted from valid datetime values.

## 3.4 Data cleaning

Data cleaning improves consistency and usability. Common operations include:
- `isna()` / `isnull()`: identify missing values.
- `dropna()`: remove rows with missing values where appropriate.
- `fillna()`: replace missing values with a chosen value or statistic.
- `drop_duplicates()`: remove duplicate records when they are truly duplicates.
- `astype()`: convert a column's data type.
- `pd.to_numeric(errors="coerce")`: convert invalid numeric text to missing values.
- `pd.to_datetime(errors="coerce")`: parse dates and mark invalid ones as missing.
- Range checks: detect impossible values, such as negative prices or ratings outside 0–5.

**Important viva point:** Missing values should not automatically be replaced with zero. Zero can mean a real measurement. The replacement strategy depends on what the field means.

## 3.5 Feature engineering in this project

In `src/prepare_ml_datasets.py`, the workflow prepares:
- Recommendation features by selecting property identifiers, location, property/room types, capacity, bedrooms, bathrooms, amenities, nightly price, rating, review count, description, verification and publication fields.
- Price features by parsing observation and stay dates, then deriving:
  - `lead_time_days`: days between observation date and stay date.
  - `stay_month`: month of the stay.
  - `stay_day`: day of the month.
  - `stay_day_of_week`: weekday index.
- It retains `base_price_npr` as the regression target.

**Lead time:** the number of days between when a price is observed and the intended stay date.  
**Feature engineering:** transforming raw columns into useful model inputs.

## 3.6 Exploratory Data Analysis (EDA)

EDA is the process of exploring data before or alongside modelling. It helps identify:
- Distribution and spread of values.
- Missing values and unusual observations.
- Category frequencies.
- Relationships between variables.
- Possible data quality issues.
- Patterns that may influence model design.

Examples relevant to this project:
- Histogram: distribution of nightly prices.
- Bar chart: listing counts by neighbourhood.
- Bar chart: median price by neighbourhood.
- Bar chart: property-type composition.
- Line chart: monthly tourist arrivals.
- Actual-vs-predicted plot: visual assessment of regression predictions.

**Mean:** sum of values divided by count.  
**Median:** middle value after sorting. It is less affected by extreme values than the mean.  
**Mode:** most frequent value.  
**Standard deviation:** typical spread around the mean.  
**Outlier:** an unusually distant observation; it should be investigated, not automatically deleted.

---

# Part 4 — Machine learning from beginner to advanced

## 4.1 What is Artificial Intelligence?

Artificial Intelligence (AI) is the broad field of building systems that perform tasks associated with human intelligence, such as reasoning, language understanding, planning or perception.

## 4.2 What is Machine Learning?

Machine Learning (ML) is a part of AI in which algorithms learn patterns from data and use those patterns to make predictions, classifications or decisions without being explicitly programmed with every rule.

**Project example:** Instead of writing a fixed formula for every accommodation price, a regression model learns relationships between input features and the prepared target price from training examples.

## 4.3 AI vs ML vs Deep Learning

- **AI:** broad umbrella of intelligent computer systems.
- **ML:** algorithms learn patterns from data.
- **Deep Learning:** ML based on multi-layer neural networks.

This project uses conventional machine-learning methods from Scikit-learn; it is not a deep-learning project.

## 4.4 Types of machine learning

1. **Supervised learning:** training data contains inputs and known target labels.
   - Regression predicts continuous numerical values.
   - Classification predicts discrete categories.
2. **Unsupervised learning:** data has no supplied target; the algorithm discovers structure, such as clusters.
3. **Reinforcement learning:** an agent learns actions through rewards and penalties.

**Which types are used here?** The price prediction module is supervised regression. The recommendation module is primarily content-based similarity and rule/score-based ranking; it does not learn user preferences from historical clicks or booking feedback.

## 4.5 Regression vs classification

| Regression | Classification |
|---|---|
| Predicts a continuous number | Predicts a class/category |
| Example: price in NPR | Example: cancellation yes/no |
| Metrics: MAE, RMSE, R² | Metrics: accuracy, precision, recall, F1 |

The price module is regression because its target, `base_price_npr`, is numeric.

## 4.6 Training, validation and testing

- **Training set:** used to fit model parameters.
- **Validation set:** used to compare models or tune hyperparameters.
- **Test set:** held aside for final evaluation.

A model should not be evaluated on the same examples it memorised during training. Otherwise the evaluation may be overly optimistic.

## 4.7 Overfitting and underfitting

- **Overfitting:** model learns training-specific noise and performs poorly on new data.
- **Underfitting:** model is too simple to learn important patterns.
- **Generalisation:** ability to perform on unseen examples.

Ways to reduce overfitting include more representative data, simpler models, regularisation, sensible hyperparameters, cross-validation and leakage-free splitting.

## 4.8 Data leakage

Data leakage occurs when information unavailable at prediction time—or information from the test set—incorrectly influences training. It produces misleadingly strong results.

**Project-specific example:** Price observations from the same property can be highly related. If some observations of one property enter training and other observations of the same property enter testing, the model may effectively see property-specific information in both sets.

Your price training script uses `GroupShuffleSplit` with `property_id` as the group. This keeps all observations belonging to one property on one side of the split. The documented split has 320 training properties and 80 testing properties, with zero overlapping property IDs.

## 4.9 Feature vs target

For the price model:
- Inputs include city, neighbourhood, property type, room type, capacity, bedrooms, bathrooms, lead time, stay month/day/weekday, nightly price and amenities.
- Target: `base_price_npr`.

A model learns a mapping:

[
hat{y}=f(X)
]

where (X) is the input feature set, (y) is the actual target and (hat y) is the model prediction.

---

# Part 5 — Recommendation system: the most important project module

## 5.1 What is a recommendation system?

A recommendation system suggests items that may be relevant to a user. Examples include films, music, products and accommodation.

Your project recommends properties based on property content and the preferences supplied by the user.

## 5.2 Main recommendation approaches

- **Content-based filtering:** recommends items similar to the content or attributes a user likes.
- **Collaborative filtering:** uses interactions or preferences from multiple users.
- **Hybrid recommendation:** combines two or more recommendation signals or approaches.

Your project is a **hybrid content-based / weighted scoring recommender**. It combines text similarity with structured attribute matching. Do not claim collaborative filtering: the current implementation does not train from a user-item interaction matrix.

## 5.3 What is TF-IDF?

TF-IDF means **Term Frequency–Inverse Document Frequency**. It converts text into numerical vectors that reflect how informative words are across a collection of documents.

- **Term Frequency (TF):** how frequently a term appears in a document.
- **Inverse Document Frequency (IDF):** reduces the influence of terms that appear in many documents.
- A term frequent in one description but uncommon across all descriptions can receive greater weight.

In this project, amenities and descriptions are combined into recommendation text. The vectorizer uses:
- `stop_words="english"`: removes common English words that often add little meaning.
- `ngram_range=(1, 2)`: considers single words and two-word phrases.
- `max_features=5000`: caps the vocabulary size in the advanced recommender.

Example: “free wifi near airport” can produce tokens such as “wifi”, “airport”, “free wifi” and “near airport”.

**Why TF-IDF?** It is a simple, interpretable and efficient way to represent descriptions and amenities numerically without requiring labelled training data.

## 5.4 What are n-grams?

An n-gram is a sequence of n consecutive tokens:
- Unigram: one token, e.g. “mountain”.
- Bigram: two tokens, e.g. “mountain view”.
- Trigram: three tokens, e.g. “near city centre”.

Using unigrams and bigrams helps preserve short phrases such as “free parking”.

## 5.5 What is cosine similarity?

Cosine similarity measures the angle-based similarity between two vectors:

[
operatorname{cosine}(A,B)=
rac{Acdot B}{|A||B|}
]

For non-negative TF-IDF vectors, scores commonly range from 0 to 1:
- Near 1: vectors point in a similar direction.
- Near 0: little or no shared weighted direction.

It is not a probability and does not mean “this property has a 90% chance of being suitable.” It is a similarity score.

## 5.6 How similar-property recommendation works

The `recommend_similar()` method:
1. Finds the selected property by its property ID.
2. Compares its TF-IDF feature vector with all property vectors using cosine similarity.
3. Scores neighbourhood match.
4. Compares relative prices.
5. Scores property type and room type similarity.
6. Calculates quality and guest-capacity similarity.
7. Combines component scores using configured weights.
8. Excludes the selected property itself.
9. Applies optional budget and minimum-rating filters.
10. Sorts by recommendation score, rating, review count and property ID, then returns the top N.

## 5.7 How personalised recommendation works

The `recommend_personalized()` method accepts a dictionary of preferences. Examples include:
- `neighbourhood`
- `property_type`
- `room_type`
- `amenities`
- `description`
- `max_price`
- `guests`

It then:
1. Builds a text query from amenities and description preferences.
2. Calculates text similarity against property descriptions/amenities.
3. Calculates categorical neighbourhood matching.
4. Validates the maximum budget.
5. Calculates property-type and room-type matching.
6. Calculates quality scores.
7. Calculates capacity suitability.
8. Redistributes weights among explicitly active ranking preferences.
9. Applies strict budget, rating and capacity constraints.
10. Sorts and returns the top recommendations.

## 5.8 Six recommendation components

The current `AdvancedPropertyRecommender` class defines these weights:

| Component | Weight | Meaning |
|---|---:|---|
| Text similarity | 0.20 | Relevance of amenities and description |
| Neighbourhood match | 0.22 | Whether the location matches |
| Budget match | 0.15 | Budget component |
| Property-type match | 0.18 | Property/room category match |
| Quality score | 0.10 | Rating and review-count signal |
| Capacity match | 0.15 | Suitability for guest count |
| **Total** | **1.00** | **100%** |

Weighted score:

[
S=sum_{i=1}^{n}w_i s_i
]

where (w_i) is a component weight and (s_i) is its component score.

The final displayed recommendation score is the combined score multiplied by 100.

**Important nuance:** Personalised recommendations can redistribute weights over the signals explicitly supplied by the user. A maximum budget is also enforced as a hard ceiling; it is not merely a soft ranking preference. The code comments explain that an affordability ceiling should not reward an over-budget listing.

## 5.9 Quality score

The code combines rating and review-count evidence:

[
Q=0.70R+0.30V
]

where:
- (R) is rating divided by 5 and clipped to the 0–1 range.
- (V) is a log-normalised review-count score.

The review component uses (log(1+	ext{reviews})), scaled by the maximum review count. The logarithm reduces the influence of very large review counts.

## 5.10 Capacity score

For personalised recommendations, the capacity component is based on:

[
C=minleft(rac{	ext{property capacity}}{	ext{required guests}},1ight)
]

A property that accommodates the requested number of guests receives full capacity suitability. A property with insufficient capacity receives a lower score and is excluded by the strict minimum-capacity filter.

## 5.11 Hard filter vs ranking score

This distinction is a likely viva question:
- **Ranking score:** orders eligible properties by relevance.
- **Hard constraint:** removes properties that violate a requirement.

Example: If the user's maximum budget is NPR 4,000, a property costing NPR 5,000 must be excluded, even if its text similarity is excellent. The implementation applies the maximum budget as a filter before selecting top results.

## 5.12 Recommendation evaluation — what it does and does not prove

The evaluation script samples up to 25 published properties using a fixed random seed, creates preference scenarios, and measures outputs such as:
- Budget compliance percentage.
- Neighbourhood match percentage.
- Property-type and room-type match.
- Guest-capacity sufficiency.
- Average recommendation score.
- Similar-property price difference and component diagnostics.
- Execution success/failure information.

The weight-optimisation script compares multiple predefined configurations, checks that weights sum to 1 and are non-negative, and records scenario outcomes.

**Do not call these scores ground-truth recommendation accuracy, precision, recall or user satisfaction.** The project does not have verified user-click, booking or relevance labels for those claims. These are scenario-based heuristic diagnostics.

---

# Part 6 — Price prediction: Gradient Boosting Regressor

## 6.1 What is price prediction?

Price prediction is a regression task. A model receives prepared input features and estimates a continuous numerical target. In this project the target is `base_price_npr`.

## 6.2 What is Gradient Boosting?

Gradient Boosting is an ensemble learning method. It builds a sequence of relatively small decision trees. Each new tree is trained to improve on errors made by the existing ensemble.

Simplified process:
1. Start with an initial prediction.
2. Calculate the residual/error.
3. Fit a small decision tree to the residual-related information.
4. Add the new tree's contribution to the current model.
5. Repeat for the chosen number of estimators.
6. Use the combined ensemble to make predictions.

The model is an additive ensemble:

[
F_m(x)=F_{m-1}(x)+eta h_m(x)
]

where (h_m(x)) is the new tree's contribution and (eta) is the learning rate.

## 6.3 Why Gradient Boosting?

It can learn non-linear relationships and interactions between structured features. It is suitable for tabular data and often provides strong performance without requiring a neural network.

## 6.4 Hyperparameters in your code

The training script uses:
- `n_estimators=150`: number of boosting stages/trees.
- `learning_rate=0.05`: contribution scale of each tree.
- `max_depth=3`: limits individual tree depth.
- `random_state=42`: makes stochastic aspects reproducible.

**Trade-off:** More trees or deeper trees can increase model complexity. A smaller learning rate often requires more estimators. Hyperparameters should be evaluated rather than assumed optimal.

## 6.5 Preprocessing pipeline

The project uses Scikit-learn's `ColumnTransformer` and `Pipeline`.

**Categorical features:** city, neighbourhood, property type and room type are processed with `OneHotEncoder(handle_unknown="ignore")`.

**Numerical features:** capacity, bedrooms, bathrooms, lead time, stay month, stay day, weekday and nightly price are passed through.

**Text feature:** amenities are processed using TF-IDF with unigrams and bigrams.

The transformed feature blocks are passed to the Gradient Boosting Regressor.

## 6.6 What is One-Hot Encoding?

One-Hot Encoding converts a categorical variable into binary indicator columns.

Example:

| Property type | Apartment | Hotel | House |
|---|---:|---:|---:|
| Apartment | 1 | 0 | 0 |
| Hotel | 0 | 1 | 0 |
| House | 0 | 0 | 1 |

Why use it? Most numerical ML estimators cannot directly interpret arbitrary category names as meaningful numbers. One-hot encoding avoids falsely implying that one category is numerically greater than another.

`handle_unknown="ignore"` prevents an error if a category appears at prediction time that was not seen during fitting.

## 6.7 What is a Pipeline?

A Pipeline chains preprocessing and model training into one object. This helps ensure that the same transformations are applied consistently during training and prediction. It also helps avoid fitting transformations separately on the test data.

## 6.8 GroupShuffleSplit

The price dataset has multiple observations for a property. A random row split could place observations of the same property in both training and testing.

`GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` splits by property ID groups. The documented evaluation has:
- Training: 808 rows from 320 properties.
- Testing: 192 rows from 80 properties.
- Shared property IDs: zero.

This is a group-based holdout split, not k-fold cross-validation.

## 6.9 Critical limitation: baseline price input

Your training code merges `nightly_price_npr` from the property table into the price observations and uses it as an input feature. The target is `base_price_npr`.

The baseline prediction is simply the existing `nightly_price_npr` value. Therefore, the model is learning to estimate a prepared observed price conditional on a known listing price and other features.

**Say this clearly if asked:** “This is a baseline-adjustment experiment on prepared sample data. It is not a pure price-discovery model that predicts an unknown market price solely from property characteristics. The baseline already performs strongly, so I report the modest improvement honestly.”

This is a methodological limitation and a good future-work opportunity.

---

# Part 7 — Evaluation metrics and how to explain your results

## 7.1 MAE — Mean Absolute Error

[
MAE=rac{1}{n}sum_{i=1}^{n}|y_i-hat y_i|
]

It is the average absolute difference between actual and predicted values. It is easy to interpret in the target's unit.

Project result: **NPR 148.33**. This means the average absolute prediction error on the documented held-out test observations is about NPR 148.

## 7.2 RMSE — Root Mean Squared Error

[
RMSE=sqrt{rac{1}{n}sum_{i=1}^{n}(y_i-hat y_i)^2}
]

It penalises larger errors more strongly because errors are squared before averaging.

Project result: **NPR 218.93**.

## 7.3 R-squared

[
R^2=1-rac{sum_i(y_i-hat y_i)^2}
{sum_i(y_i-ar y)^2}
]

It compares the model's residual error with the error from predicting the target mean. A value closer to 1 indicates a better fit relative to that reference on the evaluated data. It can be negative on poor predictions and does not prove causation or real-world validity.

Project result: **0.9942**.

## 7.4 Baseline comparison

| Metric | Gradient Boosting | Baseline |
|---|---:|---:|
| MAE | NPR 148.33 | NPR 158.85 |
| RMSE | NPR 218.93 | Refer to saved report for exact value |
| R² | 0.9942 | Refer to saved report for exact value |

MAE improvement:

[
rac{158.85-148.33}{158.85}	imes100
approx 6.62%
]

The model reduces MAE by approximately **6.62%** relative to the existing-price baseline in the documented experiment.

**Interpretation:** This is a modest improvement on prepared sample data. The high R² is partly explained by the baseline price feature being closely related to the target. It should not be presented as proof of real-market accuracy.

## 7.5 Actual vs predicted chart

- X-axis commonly represents actual target values.
- Y-axis represents predicted values.
- A perfect prediction lies on the diagonal (y=x).
- Points close to the diagonal indicate smaller errors.
- Systematic distance above/below the diagonal can indicate over/under-prediction.

## 7.6 Bias and variance

- **Bias:** error from overly simplistic assumptions.
- **Variance:** sensitivity to changes in training data.
- High bias can lead to underfitting.
- High variance can lead to overfitting.

---

# Part 8 — Python, libraries and important code concepts

## 8.1 Python

Python is a high-level, general-purpose programming language. It is widely used for data science because of its readable syntax and ecosystem.

## 8.2 Pandas

Pandas provides Series and DataFrame structures for tabular data.

Common operations:
- `pd.read_csv()`: load a CSV.
- `df.head()`: inspect first rows.
- `df.shape`: number of rows and columns.
- `df.info()`: column types and non-null counts.
- `df.describe()`: numerical summary statistics.
- `df.groupby()`: group and aggregate.
- `df.merge()`: join datasets.
- `df.sort_values()`: sort records.
- `df.to_csv()`: save a DataFrame.

## 8.3 NumPy

NumPy provides efficient arrays and numerical operations. Your recommender uses it for numerical scores, clipping, logarithms, arrays and vectorised calculations.

## 8.4 Scikit-learn

Scikit-learn provides preprocessing, feature extraction, model training, model selection and evaluation tools. Examples used in this project include:
- `TfidfVectorizer`
- `cosine_similarity`
- `ColumnTransformer`
- `OneHotEncoder`
- `Pipeline`
- `GradientBoostingRegressor`
- `GroupShuffleSplit`
- `mean_absolute_error`
- `mean_squared_error`
- `r2_score`

## 8.5 Joblib

Joblib saves and loads Python objects, including fitted ML pipelines and vectorizers.

- `joblib.dump(model, path)`: save an object.
- `joblib.load(path)`: load it later.

The dashboard loads the recommendation and price model artefacts instead of retraining them on every user interaction.

Only load trusted Joblib/pickle files: deserialisation can execute unsafe code if the file is malicious.

## 8.6 Streamlit

Streamlit is a Python framework for building interactive data applications. It provides widgets, layout, charts, tables and caching without requiring a separate frontend framework.

The project uses Streamlit for navigation and user interaction. The app loads saved model artefacts and processed datasets, then displays analytics and model outputs.

## 8.7 Caching

Caching stores reusable results to avoid repeating expensive operations.
- `st.cache_data`: appropriate for data-returning operations such as reading datasets.
- `st.cache_resource`: appropriate for shared resources such as loaded models.

Caching improves responsiveness, but stale cache data must be considered when underlying files change.

---

# Part 9 — Repository files you should know

| File | Responsibility |
|---|---|
| `app.py` | Main Streamlit application, page navigation, model/data loading and dashboard interface |
| `src/advanced_recommender.py` | Hybrid recommendation engine and scoring logic |
| `src/train_recommendation_model.py` | TF-IDF recommendation artefact preparation and saving |
| `src/train_price_model.py` | Price feature preprocessing, group split, model training and evaluation |
| `src/prepare_ml_datasets.py` | Creates recommendation and price model-ready feature datasets |
| `src/model_evaluation.py` | Regression metrics, price-band diagnostics and data/group validation helpers |
| `scripts/evaluate_recommendation_quality.py` | Scenario-based recommendation evaluation |
| `scripts/optimize_recommendation_weights.py` | Compares predefined recommendation-weight configurations |
| `models/` | Saved model artefacts |
| `data/raw/` | Raw input data, where included |
| `data/processed/` | Cleaned and model-ready datasets |
| `reports/` | Metrics, prediction outputs and analysis reports |
| `notebooks/` | Exploratory analysis and project-report workflow |
| `tests/` | Automated tests |
| `requirements.txt` | Python dependency versions |

## 9.1 Why modular code?

Separating data preparation, model training, recommendation logic, evaluation and UI makes the project easier to test, maintain, debug and extend. The dashboard does not need to contain every algorithm implementation in one giant function.

## 9.2 How to run the project

From the repository root, activate your virtual environment and install the dependencies if needed:

```powershell
# Activate your existing environment (if it exists)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the dashboard from the repository root
streamlit run app.py
```

Use the project's own virtual environment and Python version. Do not switch interpreters immediately before the viva if the application already runs successfully.

---

# Part 10 — Testing and software quality

## 10.1 What is software testing?

Testing checks whether the application behaves as expected and helps identify defects before delivery.

Types relevant to this project:
- **Unit testing:** tests a small function or class in isolation.
- **Integration testing:** checks whether components work together.
- **Smoke testing:** checks whether the app starts and basic functionality is available.
- **Data validation:** checks required columns, data types, valid values and usable targets.
- **Model evaluation:** checks prediction errors and recommendation behaviour.

## 10.2 What should you demonstrate?

- A valid property ID returns similar properties.
- An unknown property ID produces a clear error.
- A maximum budget excludes over-budget properties.
- Guest capacity is respected.
- An invalid recommendation count is rejected.
- Missing or invalid model inputs are handled safely.
- Regression metrics are calculated on aligned actual/predicted arrays.
- The dashboard loads the expected model and dataset artefacts.

The repository contains recommendation tests and a dashboard smoke test. Describe only tests that you have actually run. Do not claim all tests passed unless you can show the latest result.

## 10.3 Reproducibility

A reproducible experiment can be repeated with the same code, data and configuration to obtain consistent results. Fixed random seeds such as `random_state=42`, dependency versions and documented preprocessing help reproducibility.

---

# Part 11 — Limitations, ethics and future scope

## 11.1 Current limitations — memorise these

1. Property and price data include synthetic sample records.
2. Listings, prices and availability are not verified live Airbnb information.
3. The price target and baseline input are closely related; the price experiment is not independent market-price discovery.
4. Recommendation evaluation uses scenarios and heuristic diagnostics, not ground-truth user relevance labels.
5. The recommendation engine does not learn from a real user-click or booking history.
6. The cancellation benchmark is from Portugal hotel bookings and is not a Kathmandu Airbnb cancellation model.
7. National tourist arrivals are context only and do not directly measure Kathmandu accommodation occupancy or bookings.
8. Model quality on prepared sample data may not generalise to real operational data.

## 11.2 Future improvements

- Obtain lawful, representative and verified accommodation data.
- Remove target leakage and reconsider whether the baseline price is available at prediction time.
- Define a genuine price target and compare against simple baselines using property-level and time-aware evaluation.
- Use cross-validation and systematic hyperparameter search.
- Collect user feedback and relevance labels for recommendation evaluation.
- Explore collaborative filtering if sufficient user-item interactions become available.
- Add explainability, monitoring, drift checks and model versioning.
- Add secure deployment, access control and privacy safeguards.
- Validate tourism/occupancy data at the correct city and time scale.

## 11.3 Responsible data use

Do not claim that synthetic sample data represent real hosts or guests. Avoid exposing personal information. Clearly communicate uncertainty and the limits of model outputs. A recommendation score is not a guarantee of quality, and an estimated price is not a confirmed market quote.

---

# Part 12 — 100 viva questions with model answers

## A. Project overview (1–15)

**1. What is your project?**  
Airbnb Intelligence is an academic machine-learning and analytics application for exploring Kathmandu accommodation data, recommending properties and estimating a prepared nightly-price target.

**2. Why did you choose this project?**  
Accommodation selection involves multiple factors. I wanted to apply data analytics and machine learning to organise those factors and demonstrate preference-based recommendations and regression.

**3. What problem does it solve?**  
It reduces manual comparison by presenting accommodation insights, ranked property suggestions and a price-estimation experiment in one dashboard.

**4. What are the main modules?**  
Executive Analytics, Property Recommendations, Price Prediction, Data Analysis and Model Evaluation.

**5. What is the main output?**  
Interactive market-analysis charts, ranked property recommendations, price estimates and model-evaluation reports.

**6. Is it a booking website?**  
No. It is an analytics and ML demonstration dashboard. It does not process reservations or confirm live availability.

**7. Which city is the focus?**  
Kathmandu, Nepal.

**8. Which programming language did you use?**  
Python.

**9. Why Python?**  
It has a strong ecosystem for data manipulation, machine learning, visualisation and rapid application development.

**10. What is the role of Streamlit?**  
It provides the interactive dashboard, navigation, input widgets and presentation of results.

**11. What is the role of Scikit-learn?**  
It provides TF-IDF, cosine similarity, preprocessing, the Gradient Boosting model, data splitting and evaluation metrics.

**12. What is the role of Pandas?**  
It loads, cleans, joins, transforms and analyses tabular datasets.

**13. What is the role of NumPy?**  
It performs efficient numerical calculations and array operations.

**14. What is Joblib used for?**  
It saves and reloads trained model objects and related artefacts.

**15. What is the most important limitation?**  
Prepared property and price data include synthetic samples, so results cannot be treated as verified real-world market information.

## B. Data and preprocessing (16–30)

**16. What is data preprocessing?**  
It is the process of converting raw data into a consistent, usable form for analysis or modelling.

**17. What is missing data?**  
A value that is absent or unknown in a record.

**18. How can missing values be handled?**  
By removing affected rows/columns when justified, imputing with a suitable statistic or category, or explicitly representing missingness.

**19. What is a duplicate?**  
A repeated record. It should be removed only after deciding whether it is truly redundant.

**20. What is feature engineering?**  
Creating useful model inputs from existing data, such as extracting month and weekday from a stay date.

**21. What is EDA?**  
Exploratory Data Analysis: examining distributions, relationships, anomalies and data quality through statistics and visualisations.

**22. Why use median price?**  
It is less sensitive than the mean to unusually high or low prices.

**23. What is an outlier?**  
An observation unusually far from the rest of the data.

**24. What is data normalisation?**  
A transformation that places values on a common scale. It is different from standardisation, which commonly centres and scales using mean and standard deviation.

**25. What is categorical data?**  
Data representing labels or groups, such as neighbourhood or property type.

**26. What is numerical data?**  
Data represented by numbers, such as price, capacity or rating.

**27. What is a target variable?**  
The output a supervised model is trained to predict. Here it is `base_price_npr` for price regression.

**28. What is a feature?**  
An input variable used by a model.

**29. Why create separate feature datasets?**  
To keep the model inputs organised and separate from unrelated columns, making training and application more consistent.

**30. What is data validation?**  
Checking that required columns, types, ranges and target values meet expected conditions before training or inference.

## C. Machine learning fundamentals (31–45)

**31. What is machine learning?**  
A field where algorithms learn patterns from data to make predictions or decisions.

**32. What is supervised learning?**  
Learning from examples that contain both input features and known target labels.

**33. What is unsupervised learning?**  
Finding structure in data without supplied target labels.

**34. What is reinforcement learning?**  
Learning actions through interaction and rewards or penalties.

**35. Is your price model supervised or unsupervised?**  
Supervised, because training examples include a known numerical target.

**36. Is price prediction classification or regression?**  
Regression, because the target is a continuous numerical value.

**37. What is a training set?**  
The data used to fit the model.

**38. What is a test set?**  
Held-out data used to estimate performance on unseen examples.

**39. What is overfitting?**  
When a model fits training data too closely and generalises poorly.

**40. What is underfitting?**  
When a model is too simple to capture important patterns.

**41. What is generalisation?**  
The ability to perform well on unseen data.

**42. What is data leakage?**  
When information from outside the legitimate training inputs improperly influences model training or evaluation.

**43. Why is random splitting not always appropriate?**  
Related observations may be split across training and testing, making the test set less independent.

**44. What is a hyperparameter?**  
A setting chosen before or around training, such as tree depth or number of estimators.

**45. What is an ensemble model?**  
A model that combines multiple learners to produce a final prediction.

## D. Recommendation system (46–65)

**46. What is a recommendation system?**  
A system that ranks or suggests items likely to be relevant to a user.

**47. Which recommendation method does your project use?**  
A hybrid content-based approach combining text similarity and weighted structured-feature scoring.

**48. What is content-based filtering?**  
Recommending items using their attributes and similarity to the user's stated interests or a selected item.

**49. What is collaborative filtering?**  
Recommending from patterns in user-item interactions, such as ratings or purchases.

**50. Does your project use collaborative filtering?**  
No. It does not train from a user-item interaction matrix. It uses content and preference signals.

**51. What is TF-IDF?**  
A text-vectorisation method that weights terms according to their frequency in a document and rarity across the document collection.

**52. What does IDF do?**  
It reduces the weight of words that appear in many documents and are less informative.

**53. What is cosine similarity?**  
A measure of the angle-based similarity between two vectors.

**54. Why use cosine similarity?**  
It compares text vectors based on direction rather than raw document length.

**55. What is a bigram?**  
A sequence of two consecutive tokens.

**56. Why use stop words?**  
Common words often carry little distinguishing information, so removing them can reduce noise.

**57. What are the recommendation components?**  
Text similarity, neighbourhood match, budget match, property-type match, quality score and capacity match.

**58. Why use weighted scoring?**  
Different preferences contribute different amounts to the final ranking.

**59. Why is budget a hard filter?**  
A maximum budget is a constraint. A property above it should not be recommended merely because other signals are strong.

**60. How is quality calculated?**  
The implementation combines a normalised rating component (70%) and a log-normalised review-count component (30%).

**61. Why use log of review count?**  
It reduces the dominance of very large review counts and compresses the scale.

**62. What is top-N recommendation?**  
Returning the N highest-ranked eligible results.

**63. How do you avoid recommending the same property?**  
The similar-property method excludes the selected reference property from the result set.

**64. How are missing text values handled?**  
The advanced recommender fills missing text with empty strings before vectorisation.

**65. Are recommendation scores probabilities?**  
No. They are weighted relevance scores, not calibrated probabilities or guaranteed satisfaction rates.

## E. Price model and evaluation (66–85)

**66. Which algorithm is used for price prediction?**  
Gradient Boosting Regressor.

**67. Explain Gradient Boosting.**  
It builds trees sequentially, with each new tree improving the ensemble's errors, and combines their contributions.

**68. What is a decision tree?**  
A model that makes predictions through a sequence of feature-based decision rules.

**69. What is an ensemble?**  
A combination of multiple models or learners.

**70. What does n_estimators=150 mean?**  
The boosting model uses 150 boosting stages/trees.

**71. What does learning_rate=0.05 mean?**  
It scales each new tree's contribution to the ensemble.

**72. What does max_depth=3 mean?**  
It limits the depth of each individual regression tree.

**73. What is One-Hot Encoding?**  
It converts categories into binary indicator features.

**74. What is TF-IDF doing in the price pipeline?**  
It converts the amenities text into numerical features for the regressor.

**75. What is a ColumnTransformer?**  
It applies different preprocessing operations to different columns or column groups.

**76. What is a Pipeline?**  
It chains preprocessing and model steps into one reusable estimator.

**77. What is MAE?**  
The average absolute difference between actual and predicted values.

**78. What is RMSE?**  
The square root of average squared prediction error; it penalises large errors more strongly.

**79. What is R-squared?**  
A measure comparing residual error with the variation around the target mean.

**80. What are your documented metrics?**  
MAE NPR 148.33, RMSE NPR 218.93 and R² 0.9942.

**81. What is your baseline?**  
The existing `nightly_price_npr` value used directly as a prediction for comparison.

**82. What improvement did the model achieve?**  
The documented MAE decreased from NPR 158.85 to NPR 148.33, an improvement of approximately 6.62%.

**83. Why use GroupShuffleSplit?**  
To keep all observations belonging to a property in either training or testing, preventing property-ID overlap.

**84. What is the most important caveat about the price model?**  
It uses an existing nightly listing price as an input to estimate a closely related prepared target. It is a baseline-adjustment experiment, not independent market-price discovery.

**85. Does R² of 0.9942 mean the model is 99.42% accurate?**  
No. R² is not classification accuracy or a percentage guarantee. It measures fit relative to a mean-prediction reference on the evaluated dataset.

## F. Application, testing and advanced questions (86–100)

**86. What is Streamlit?**  
A Python framework for creating interactive data applications.

**87. Why cache data?**  
To avoid repeatedly loading or recomputing the same data and improve response time.

**88. Difference between cache_data and cache_resource?**  
Use cache_data for data results and cache_resource for shared resources such as loaded models.

**89. Why save the model?**  
So the application can load the fitted artefact without retraining every time it starts.

**90. What is a virtual environment?**  
An isolated Python environment for project dependencies.

**91. What is requirements.txt?**  
A file listing project dependencies and their pinned versions.

**92. What is Git?**  
A version-control system that tracks changes in source code.

**93. What is GitHub?**  
A platform for hosting Git repositories and collaborating on software.

**94. What is unit testing?**  
Testing a small function or component independently.

**95. What is a smoke test?**  
A basic test that checks whether the application or key workflow starts and operates at a fundamental level.

**96. Why is modular architecture useful?**  
It separates responsibilities, improves readability and makes testing and maintenance easier.

**97. What would you improve first?**  
I would obtain representative verified data and redesign the price target/features to remove dependence on an already-known baseline price.

**98. How would you improve recommendation quality?**  
Collect real user feedback or relevance labels, evaluate against those labels, and compare the current hybrid method with additional approaches.

**99. Can this model be used for actual business decisions today?**  
Not without representative verified data, appropriate validation and deployment safeguards. It is currently an academic demonstration.

**100. What did you learn from this project?**  
I learned the end-to-end ML workflow: data preparation, EDA, feature engineering, recommendation logic, regression, evaluation, dashboard development, testing, documentation and the importance of communicating limitations honestly.

---

# Part 13 — Advanced questions teachers may ask

## 13.1 Why not use a neural network?

The current dataset and task do not require a deep neural network. TF-IDF and cosine similarity are interpretable for text-based content matching, while Gradient Boosting is a strong choice for tabular regression. Model choice should depend on data, task, evaluation and complexity—not on using the most fashionable algorithm.

## 13.2 What happens if a new property is added?

The recommendation model's text vectorizer and feature matrix need to be updated or rebuilt so the new property's text can be represented in the same feature space. The dashboard data and model artefact must remain consistent.

## 13.3 What happens if a new category appears?

The price pipeline uses OneHotEncoder with `handle_unknown="ignore"`, so unseen categories do not cause the encoder to fail. They will not have a learned category-specific coefficient/representation from training, however.

## 13.4 Why use both text and structured attributes?

Text captures semantic information in amenities and descriptions. Structured attributes capture explicit constraints such as location, property type, budget and guest capacity. Combining them gives a more useful ranking than text similarity alone.

## 13.5 Why is a high R² not sufficient?

R² does not show the absolute size of errors, does not prove generalisation to real-world data and can be misleading when the input contains a feature closely related to the target. MAE, RMSE, baseline comparison, data provenance and split design all matter.

## 13.6 What is a fair evaluation of a recommender?

A fair evaluation needs relevance ground truth, such as user ratings, clicks, saves or bookings collected under an appropriate protocol. Depending on the task, metrics may include Precision@K, Recall@K, MAP, NDCG or controlled user studies. The current project's scenario diagnostics do not constitute those ground-truth metrics.

## 13.7 What is model drift?

Model drift is a change over time in input data patterns or the relationship between inputs and targets. Real accommodation prices, neighbourhood popularity and guest preferences can change. A deployed model needs monitoring and periodic re-evaluation.

## 13.8 What is explainability?

Explainability means providing understandable reasons for a model output. In this project, recommendation component scores can help show whether text, location, quality or capacity contributed to a ranking. For the price model, feature importance or suitable explanation methods could be added and validated.

## 13.9 What is the difference between model training and inference?

- Training fits the model using examples with known targets.
- Inference uses the fitted model to produce outputs for new inputs.

The dashboard primarily performs inference using saved artefacts.

## 13.10 Why should the test set remain untouched during tuning?

Repeatedly tuning against the test set leaks information from the test set into model selection. It makes the final score optimistic. Use training/validation or cross-validation for tuning, then evaluate once on a held-out test set.

---

# Part 14 — Practical viva demonstration script

Use this order if the teacher asks you to run the project:

1. **Start with the home/overview:** State the project purpose and scope.
2. **Executive Analytics:** Explain one KPI, one neighbourhood chart, one price chart and tourism-context limitation.
3. **Data Analysis:** Show the dataset and explain rows, columns, data types and one cleaning/feature-engineering step.
4. **Property Recommendations:** Enter a neighbourhood, budget, guest count or amenity preference. Explain hard filters versus ranking scores.
5. **Similar Property:** Select a property and explain TF-IDF, cosine similarity and the hybrid score.
6. **Price Prediction:** Explain the inputs, target, Gradient Boosting and the baseline caveat.
7. **Model Evaluation:** Explain MAE, RMSE, R² and the baseline improvement.
8. **Source code:** Open `src/advanced_recommender.py` and `src/train_price_model.py`. Identify the actual methods and pipeline.
9. **Testing:** Show a test or report only if it is available and has been run.
10. **Close:** State limitations and the next improvement.

**If the app fails during demonstration:** Stay calm. Explain the intended workflow, identify the likely dependency/data/model-file issue, and show the relevant source code or saved report if available. Never pretend an output appeared when it did not.

---

# Part 15 — Final-night revision plan

## Priority 1: Must know (first 45 minutes)

- Project introduction (60 seconds).
- Problem statement, objectives and five workspaces.
- End-to-end architecture.
- TF-IDF and cosine similarity.
- Hybrid recommendation components and weights.
- Gradient Boosting and its hyperparameters.
- MAE, RMSE, R² and baseline improvement.
- GroupShuffleSplit and data leakage.
- Synthetic-data limitation and price-model caveat.

## Priority 2: Source-code understanding (30 minutes)

Open these files and identify the key sections:
- `app.py`
- `src/advanced_recommender.py`
- `src/train_price_model.py`
- `src/prepare_ml_datasets.py`
- `src/model_evaluation.py`

For each file, explain its input, processing and output in your own words.

## Priority 3: Practice aloud (30 minutes)

Answer questions 1–30 without reading. Then answer questions 46–85. Practise the 60-second introduction three times.

## Priority 4: Before sleeping

Memorise these facts:
- 5 dashboard workspaces.
- 400 property records; 354 published (documented dataset scale).
- TF-IDF + cosine similarity + weighted hybrid ranking.
- Six recommender components.
- Gradient Boosting Regressor.
- 150 estimators, learning rate 0.05, max depth 3.
- Group split: 320 training properties, 80 test properties, zero overlap (documented run).
- MAE NPR 148.33.
- RMSE NPR 218.93.
- R² 0.9942.
- Baseline MAE NPR 158.85.
- MAE improvement 6.62%.
- Synthetic sample data and baseline-price caveat.

---

# Part 16 — How to answer when you do not know

Use this structure:

1. **Acknowledge:** “I am not fully certain about that specific detail.”
2. **Explain what you do know:** connect it to the relevant algorithm or code.
3. **Reason carefully:** explain how you would verify or test it.
4. **Do not invent:** never make up a dataset source, metric, feature or implemented module.

Example:

“I have not implemented collaborative filtering in this version. The current recommender is content-based and combines TF-IDF text similarity with structured preference scores. Collaborative filtering could be explored in future work if we collect sufficient user-property interaction data.”

This is a stronger answer than claiming an algorithm you did not implement.

---

# Part 17 — Final checklist

- [ ] I can explain the project in 60 seconds.
- [ ] I can explain all five dashboard workspaces.
- [ ] I understand the datasets and their limitations.
- [ ] I can explain cleaning, EDA and feature engineering.
- [ ] I can explain TF-IDF with a simple example.
- [ ] I can explain cosine similarity and why it is not a probability.
- [ ] I can explain all six recommendation components.
- [ ] I can distinguish hard filters from ranking weights.
- [ ] I can explain Gradient Boosting and its hyperparameters.
- [ ] I can explain One-Hot Encoding, TF-IDF, ColumnTransformer and Pipeline.
- [ ] I can explain GroupShuffleSplit and data leakage.
- [ ] I can calculate and interpret MAE, RMSE and R².
- [ ] I can explain the baseline and 6.62% MAE improvement.
- [ ] I can clearly explain the synthetic-data and baseline-price limitations.
- [ ] I can open the main source files and identify the key methods.
- [ ] I can demonstrate the app without making unsupported claims.

**Best final statement to teachers:**  
“This project demonstrates how data preparation, machine learning, recommendation algorithms, regression evaluation and interactive visualisation can be integrated into one application. I have also identified the limitations of the prepared data and current modelling approach. My next step would be to validate the system using representative real-world data and a more independent prediction target.”
