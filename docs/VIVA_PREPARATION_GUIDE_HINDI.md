# Airbnb Intelligence — संपूर्ण प्रोजेक्ट Viva तैयारी गाइड (हिंदी)

**उद्देश्य:** 200 अंकों की प्रोजेक्ट Viva के लिए शुरुआती स्तर से Advanced स्तर तक तैयारी।  
**प्रोजेक्ट:** Airbnb Intelligence  
**रिपॉजिटरी:** https://github.com/Netrahoni/Airbnb-Intelligence  
**टेक्नोलॉजी:** Python, Streamlit, Pandas, NumPy और Scikit-learn  
**डोमेन:** काठमांडू, नेपाल के accommodation data का विश्लेषण और Machine Learning

> **सबसे जरूरी बात — प्रोजेक्ट की वास्तविक सीमा:** यह एक academic analytics और machine-learning demonstration application है, booking website नहीं। Property और price datasets में synthetic/sample records शामिल हैं। इसलिए इन्हें verified live Airbnb listings, वास्तविक market prices, bookings या availability न बताएं। Cancellation dataset पुर्तगाल के hotels का external benchmark है, Kathmandu Airbnb data नहीं। Nepal tourist-arrival dataset पूरे देश का संदर्भ देता है, Kathmandu occupancy का सीधा माप नहीं।

---

# भाग 1 — 60 सेकंड में अपना प्रोजेक्ट कैसे समझाएँ

## 1.1 यह परिचय बोलने का अभ्यास करें

“सुप्रभात आदरणीय शिक्षकों। मेरे प्रोजेक्ट का नाम **Airbnb Intelligence** है। यह Kathmandu, Nepal के accommodation-related data पर आधारित एक academic Machine Learning और Data Analytics application है।

इस प्रोजेक्ट का मुख्य उद्देश्य accommodation data का विश्लेषण करना और यह दिखाना है कि Machine Learning की मदद से property recommendations तथा nightly price estimation कैसे किया जा सकता है।

मैंने इस प्रोजेक्ट को Python, Pandas, NumPy और Scikit-learn की मदद से विकसित किया है तथा इसके परिणामों को एक interactive Streamlit dashboard में प्रस्तुत किया है।

इस application में पाँच मुख्य workspaces हैं: Executive Analytics, Property Recommendations, Price Prediction, Data Analysis और Model Evaluation।

Recommendation module एक hybrid content-based approach का उपयोग करता है। इसमें text के लिए TF-IDF और cosine similarity तथा neighbourhood, budget, property type, quality और guest capacity जैसे structured factors के weighted scores शामिल हैं। Price Prediction module में preprocessing pipeline के साथ Gradient Boosting Regressor का उपयोग किया गया है। Model को MAE, RMSE और R-squared से evaluate किया गया है तथा baseline से तुलना की गई है।

इस प्रोजेक्ट में data preparation, exploratory data analysis, model evaluation, testing और documentation भी शामिल हैं। इसकी एक महत्वपूर्ण सीमा यह है कि prepared property और price data में synthetic sample records हैं। इसलिए यह project ML workflow और model behaviour का academic demonstration है, verified live market prediction नहीं।”

## 1.2 छोटे सवालों के सीधे जवाब

- **प्रोजेक्ट का नाम:** Airbnb Intelligence
- **प्रोजेक्ट का प्रकार:** Machine Learning और Data Analytics application
- **मुख्य क्षेत्र:** Kathmandu accommodation data
- **Programming language:** Python
- **Dashboard framework:** Streamlit
- **ML library:** Scikit-learn
- **Data handling:** Pandas और NumPy
- **Model save/load:** Joblib
- **मुख्य ML कार्य:** Recommendation/ranking और Regression
- **Recommendation technique:** Hybrid content-based recommendation
- **Price algorithm:** Gradient Boosting Regressor
- **Interface:** Interactive Streamlit dashboard

---

# भाग 2 — पूरे प्रोजेक्ट को समझें

## 2.1 Problem Statement (समस्या क्या है?)

किसी accommodation को चुनते समय location, budget, property type, room type, amenities, guest capacity, rating और description जैसे कई factors देखने पड़ते हैं। बहुत सारी properties की manually तुलना करना समय लेने वाला काम है। अलग-अलग datasets में मौजूद data को साफ करके समझना भी जरूरी होता है।

Airbnb Intelligence इन समस्याओं को address करता है। यह accommodation-related data को prepare और analyse करता है, user preferences के आधार पर properties को rank करता है, prepared price target का अनुमान लगाता है और सभी insights को एक dashboard में दिखाता है।

## 2.2 Project Objectives (उद्देश्य)

1. Accommodation-related datasets को prepare और validate करना।
2. Exploratory Data Analysis (EDA) करना।
3. Neighbourhood, property type और price patterns का विश्लेषण करना।
4. Preference-based property recommendation engine बनाना।
5. Prepared nightly-price target के लिए regression pipeline बनाना।
6. MAE, RMSE, R² और baseline comparison से model evaluate करना।
7. Results को Streamlit dashboard में दिखाना।
8. Modular programming, testing, reproducibility और documentation का अभ्यास करना।

## 2.3 Dashboard के पाँच Workspaces

| Workspace | इसमें क्या होता है? |
|---|---|
| Executive Analytics | KPIs, property inventory, price distribution, neighbourhood comparison, property types और tourism context |
| Property Recommendations | User preferences के आधार पर recommendations, similar-property search और filters |
| Price Prediction | User inputs और trained pipeline के आधार पर estimated target price |
| Data Analysis | Dataset inspection, descriptive statistics और visual analysis |
| Model Evaluation | Regression metrics, baseline comparison, prediction diagnostics और recommendation evaluation |

**Viva में कैसे दिखाएँ:** हर workspace खोलें, उसका उद्देश्य बताएं, input data बताएं, output समझाएं और उसकी एक limitation बताएं। केवल pages पर click न करें—हर page का काम समझाएँ।

## 2.4 End-to-End Workflow

1. **Data preparation:** Property, price, availability, benchmark और tourism-context datasets तैयार करना।
2. **Data inspection:** Columns, data types, missing values, duplicates और dataset shape जाँचना।
3. **Data cleaning:** गलत या missing values को संभालना और fields को standardise करना।
4. **Feature engineering:** Dates से month/day/weekday निकालना और recommendation text तैयार करना।
5. **EDA:** Statistics और charts से data patterns समझना।
6. **Model training:** Recommendation representation और price-regression pipeline तैयार करना।
7. **Evaluation:** Test data पर predictions की तुलना actual values और baseline से करना।
8. **Persistence:** Trained model artefacts को Joblib से save करना।
9. **Application:** Streamlit में data और models load करके user inputs के आधार पर results दिखाना।
10. **Testing/documentation:** Tests चलाना और setup, assumptions तथा limitations लिखना।

## 2.5 Architecture Diagram — Viva में ऐसे बनाएँ

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
```

---

# भाग 3 — Dataset और Data Preprocessing

## 3.1 प्रोजेक्ट के मुख्य datasets

| File | उपयोग |
|---|---|
| `properties_clean.csv` | Property details, location, price, rating, amenities, description और publication status |
| `price_observations_clean.csv` | Prepared accommodation price observations |
| `availability_clean.csv` | Calendar/availability-related records |
| `booking_cancellations_clean.csv` | External hotel-booking cancellation benchmark |
| `nepal_monthly_tourist_arrivals_clean.csv` | Nepal के tourism का national-level context |
| `recommendation_features.csv` | Recommendation engine के लिए model-ready property features |
| `price_prediction_features.csv` | Price regression के लिए model-ready data |

पहले documented dataset scale के अनुसार लगभग 400 property records (354 published), 1,000 price observations, 36,000 availability records और 360 monthly tourism-context records हैं। Cancellation benchmark में मूल रूप से 119,390 rows और 25 columns थे। यदि local files में counts बदल गए हों तो Viva से पहले वर्तमान CSV files से verify करें।

## 3.2 Dataset क्या होता है?

Dataset observations का एक व्यवस्थित collection होता है, जिसे सामान्यतः table के रूप में रखा जाता है।

- **Row:** एक record या observation।
- **Column:** एक variable या attribute।
- **Feature:** Model को दिया जाने वाला input variable।
- **Target/Label:** वह output जिसे supervised model predict करना सीखता है।
- **Record ID:** किसी entity की पहचान, जैसे property_id।

उदाहरण:

| property_id | neighbourhood | capacity | nightly_price_npr |
|---|---|---:|---:|
| KTM001 | Thamel | 2 | 3500 |
| KTM002 | Boudha | 4 | 5000 |

यहाँ प्रत्येक row एक property है। Neighbourhood, capacity और price columns/features हैं।

## 3.3 Data Types

- **Integer:** पूर्ण संख्या, जैसे guest capacity।
- **Float:** दशमलव संख्या, जैसे rating या price।
- **String/Object:** Text, जैसे neighbourhood या description।
- **Boolean:** True/False, जैसे published status।
- **Datetime:** Date और time की value।

Data type सही होना जरूरी है क्योंकि string पर numerical calculation नहीं कर सकते। Text model को text input चाहिए और date features निकालने के लिए valid datetime चाहिए।

## 3.4 Data Cleaning

Data cleaning का अर्थ raw data को consistent और usable बनाना है।

| Pandas function | काम |
|---|---|
| `isna()` / `isnull()` | Missing values पहचानना |
| `dropna()` | उचित स्थिति में missing rows हटाना |
| `fillna()` | Missing values भरना |
| `drop_duplicates()` | वास्तविक duplicate records हटाना |
| `astype()` | Data type बदलना |
| `pd.to_numeric(errors="coerce")` | Invalid numeric text को missing बनाना |
| `pd.to_datetime(errors="coerce")` | Date parse करना; invalid dates को missing बनाना |

**Viva में ध्यान रखें:** हर missing value को zero से replace नहीं करना चाहिए। Zero का अपना वास्तविक अर्थ हो सकता है। Replacement field के अर्थ के अनुसार तय किया जाता है।

## 3.5 Feature Engineering

Feature engineering raw columns से उपयोगी model inputs बनाने की प्रक्रिया है।

इस project में:
- Recommendation के लिए property ID, location, property/room type, capacity, bedrooms, bathrooms, amenities, price, rating, review count और description जैसे fields तैयार किए जाते हैं।
- Price model के लिए observation date और stay date parse करके date-based features निकाले जाते हैं:
  - `lead_time_days`: Observation date और stay date के बीच दिनों का अंतर।
  - `stay_month`: Stay का महीना।
  - `stay_day`: महीने का दिन।
  - `stay_day_of_week`: सप्ताह का दिन।
- Regression target `base_price_npr` रखा गया है।

**उदाहरण:** यदि observation 1 जून को है और stay 11 जून को है, तो lead time 10 दिन होगा।

## 3.6 EDA — Exploratory Data Analysis

EDA का अर्थ है model बनाने से पहले या modelling के दौरान data को statistics और visualisations से समझना।

इससे पता चलता है:
- Values का distribution कैसा है।
- Missing values और unusual observations कहाँ हैं।
- कौन-सी categories अधिक हैं।
- Variables के बीच क्या संबंध है।
- Data quality की समस्याएँ क्या हैं।

Project के उदाहरण:
- Histogram: Nightly price distribution।
- Bar chart: Neighbourhood के अनुसार property count।
- Bar chart: Neighbourhood के अनुसार median price।
- Bar chart: Property type composition।
- Line chart: Monthly tourist arrivals।
- Actual vs Predicted plot: Regression prediction का visual comparison।

**Mean:** सभी values का योग ÷ कुल values।  
**Median:** Sorted values का बीच का मान; extreme values से कम प्रभावित होता है।  
**Mode:** सबसे अधिक बार आने वाली value।  
**Standard deviation:** Mean के आसपास data का फैलाव।  
**Outlier:** बाकी observations से बहुत दूर value; इसे जाँचें, बिना कारण delete न करें।

---

# भाग 4 — Machine Learning: Beginner से Advanced

## 4.1 Artificial Intelligence (AI) क्या है?

Artificial Intelligence computer science का वह क्षेत्र है जिसमें ऐसी systems बनाई जाती हैं जो reasoning, language understanding, planning या perception जैसे बुद्धिमत्ता से जुड़े काम कर सकें।

## 4.2 Machine Learning (ML) क्या है?

Machine Learning, AI का एक भाग है। इसमें algorithms data से patterns सीखते हैं और उन patterns का उपयोग prediction या decision के लिए करते हैं। हर rule को manually program करना जरूरी नहीं होता।

**Project example:** Accommodation price के लिए हर परिस्थिति का fixed formula लिखने के बजाय regression model training examples से input features और target price के बीच संबंध सीखता है।

## 4.3 AI vs ML vs Deep Learning

- **AI:** Intelligent computer systems का व्यापक क्षेत्र।
- **ML:** Data से patterns सीखने वाले algorithms।
- **Deep Learning:** Multi-layer neural networks पर आधारित ML।

यह project Scikit-learn के traditional machine-learning methods का उपयोग करता है। यह Deep Learning project नहीं है।

## 4.4 Machine Learning के प्रकार

1. **Supervised Learning:** Training data में inputs और known target दोनों होते हैं।
   - Regression: Continuous numerical value predict करता है।
   - Classification: Category/class predict करता है।
2. **Unsupervised Learning:** Target label के बिना data में patterns या groups खोजता है।
3. **Reinforcement Learning:** Agent rewards और penalties के आधार पर actions सीखता है।

**इस project में क्या इस्तेमाल हुआ है?** Price Prediction supervised regression है। Recommendation module content similarity और weighted ranking पर आधारित है। यह historical user clicks या bookings से preferences train नहीं करता।

## 4.5 Regression vs Classification

| Regression | Classification |
|---|---|
| Continuous numerical value predict करता है | Discrete class/category predict करता है |
| उदाहरण: NPR में price | उदाहरण: Cancellation Yes/No |
| Metrics: MAE, RMSE, R² | Metrics: Accuracy, Precision, Recall, F1 |

Price Prediction regression है क्योंकि `base_price_npr` एक numerical target है।

## 4.6 Training, Validation और Testing

- **Training set:** Model को fit करने के लिए data।
- **Validation set:** Models compare करने या hyperparameters tune करने के लिए।
- **Test set:** Final evaluation के लिए अलग रखा गया data।

यदि model को उन्हीं examples पर evaluate किया जाए जिन्हें उसने training में देखा है, तो performance वास्तविकता से बेहतर दिख सकती है।

## 4.7 Overfitting और Underfitting

- **Overfitting:** Model training data को बहुत अधिक याद कर लेता है और नए data पर खराब perform करता है।
- **Underfitting:** Model बहुत simple होता है और महत्वपूर्ण patterns नहीं सीख पाता।
- **Generalisation:** Unseen data पर भी अच्छा perform करने की क्षमता।

Overfitting कम करने के तरीके: representative data, simpler model, regularisation, sensible hyperparameters, cross-validation और leakage-free splitting।

## 4.8 Data Leakage

Data leakage तब होता है जब ऐसी जानकारी training/evaluation में पहुँच जाती है जो prediction के समय उपलब्ध नहीं होनी चाहिए या test data से training प्रभावित हो जाती है। इससे metrics भ्रामक रूप से अच्छे हो सकते हैं।

**Project-specific उदाहरण:** एक ही property की कई price observations हो सकती हैं। यदि उसी property की कुछ rows training और कुछ testing में चली जाएँ, तो model को दोनों sets में उस property से जुड़ी जानकारी मिल सकती है।

आपकी price training script में `GroupShuffleSplit` का उपयोग `property_id` को group बनाकर किया गया है। इससे एक property की सभी observations या तो training में रहती हैं या testing में। Documented run में 320 training properties और 80 testing properties थीं तथा दोनों sets में property ID overlap शून्य था।

## 4.9 Feature और Target

Price model के inputs में city, neighbourhood, property type, room type, capacity, bedrooms, bathrooms, lead time, stay month/day/weekday, nightly price और amenities शामिल हैं।

Target: `base_price_npr`

Mathematical रूप:

[
\hat{y}=f(X)
]

यहाँ (X) input features हैं, (y) actual target है और (hat{y}) model prediction है।

---

# भाग 5 — Recommendation System

## 5.1 Recommendation System क्या है?

Recommendation system ऐसा system है जो user को उसकी रुचि या जरूरत के अनुसार relevant items suggest करता है। उदाहरण: movies, music, products और accommodation।

इस project में property content और user द्वारा दी गई preferences के आधार पर properties rank की जाती हैं।

## 5.2 Recommendation के मुख्य तरीके

- **Content-based filtering:** Item के content/attributes और user की पसंद के आधार पर recommendation।
- **Collaborative filtering:** कई users के interactions या preferences के patterns का उपयोग।
- **Hybrid recommendation:** दो या अधिक recommendation signals/approaches को मिलाना।

आपका project **hybrid content-based / weighted scoring recommender** है। यह text similarity को structured attribute matching के साथ जोड़ता है।

**महत्वपूर्ण:** Collaborative filtering implemented नहीं है। Project user-item interaction matrix से train नहीं करता।

## 5.3 TF-IDF क्या है?

TF-IDF का पूरा नाम **Term Frequency–Inverse Document Frequency** है। यह text को numerical vectors में बदलता है ताकि शब्दों की importance को represent किया जा सके।

- **TF (Term Frequency):** किसी document में term कितनी बार आया।
- **IDF (Inverse Document Frequency):** जो शब्द लगभग सभी documents में आते हैं, उनका importance कम करना।
- किसी एक description में महत्वपूर्ण लेकिन पूरे collection में कम मिलने वाले शब्द को अधिक weight मिल सकता है।

Project में amenities और descriptions को recommendation text के रूप में इस्तेमाल किया जाता है। Advanced recommender में:
- `stop_words="english"`: सामान्य English words हटाता है।
- `ngram_range=(1, 2)`: Single words और दो शब्दों के phrases दोनों देखता है।
- `max_features=5000`: Vocabulary size को अधिकतम 5000 features तक सीमित करता है।

उदाहरण: “free wifi near airport” से “wifi”, “airport”, “free wifi” जैसे tokens बन सकते हैं।

**TF-IDF क्यों?** यह text को numerical form में बदलने का सरल, समझने योग्य और efficient तरीका है। इसके लिए labelled training data जरूरी नहीं है।

## 5.4 N-gram क्या होता है?

N-gram लगातार आने वाले n tokens का समूह है:
- **Unigram:** एक token, जैसे “mountain”।
- **Bigram:** दो tokens, जैसे “mountain view”।
- **Trigram:** तीन tokens, जैसे “near city centre”।

Unigram और bigram दोनों इस्तेमाल करने से “free parking” जैसे phrases का अर्थ बेहतर capture हो सकता है।

## 5.5 Cosine Similarity क्या है?

Cosine similarity दो vectors के बीच angle के आधार पर similarity मापती है।

[
\text{Cosine Similarity}(A,B)=\frac{A\cdot B}{\|A\|\|B\|}
]

Non-negative TF-IDF vectors के लिए score सामान्यतः 0 से 1 के बीच होता है:
- 1 के पास: vectors बहुत similar दिशा में हैं।
- 0 के पास: weighted direction में बहुत कम similarity है।

यह probability नहीं है। 0.90 का अर्थ यह नहीं कि property के suitable होने की 90% संभावना है।

## 5.6 Similar Property Recommendation कैसे काम करता है?

`recommend_similar()` method का सामान्य workflow:

1. Property ID से selected property खोजता है।
2. TF-IDF vector की तुलना बाकी properties से cosine similarity द्वारा करता है।
3. Neighbourhood match score निकालता है।
4. Relative price compare करता है।
5. Property type और room type similarity score करता है।
6. Quality और guest-capacity similarity निकालता है।
7. Configured weights से component scores combine करता है।
8. Selected property को results से हटाता है।
9. Optional budget और minimum-rating filters लगाता है।
10. Score, rating, review count और property ID के आधार पर sort करके Top-N results देता है।

## 5.7 Personalised Recommendation कैसे काम करता है?

`recommend_personalized()` preferences की dictionary लेता है। उदाहरण:
- `neighbourhood`
- `property_type`
- `room_type`
- `amenities`
- `description`
- `max_budget`
- `guests`

इसके बाद हर eligible property को preference matching के आधार पर score किया जाता है। Maximum budget constraint hard filter की तरह भी काम करता है—budget से अधिक property केवल दूसरे scores अच्छे होने के कारण recommend नहीं होनी चाहिए।

यदि user कुछ preferences ही देता है, तो active preferences के अनुसार weights dynamically redistribute हो सकते हैं।

## 5.8 Recommendation के छह मुख्य components और weights

Advanced recommender के configured base weights:

| Component | Weight | अर्थ |
|---|---:|---|
| Text similarity | 0.20 | Amenities और description की text similarity |
| Neighbourhood match | 0.22 | User की पसंद के क्षेत्र से match |
| Budget match | 0.15 | Budget के साथ price compatibility |
| Property type match | 0.18 | Property type की समानता |
| Quality score | 0.10 | Rating और review-count आधारित quality |
| Capacity match | 0.15 | Guests की संख्या के लिए capacity suitability |
| **Total** | **1.00** | **100%** |

Weighted score का सामान्य रूप:

[
S=\sum_{i=1}^{n}w_i s_i
]

यहाँ (w_i) component का weight और (s_i) उसका score है। कुल weights 1.00 हैं।

**Quality score:** Implementation में normalised rating का 70% और log-normalised review count का 30% योगदान है।

**Review count पर log क्यों?** बहुत अधिक reviews वाली property को असमान रूप से dominate करने से रोकने और बड़े numbers का scale compress करने के लिए।

**Capacity score:** Personalised recommendation में capacity और required guests की तुलना की जाती है। पर्याप्त capacity मिलने पर score अधिक हो सकता है।

## 5.9 Hard Filter और Weighted Score में अंतर

- **Hard filter:** ऐसी property को पूरी तरह हटा देता है जो आवश्यक शर्त पूरी नहीं करती, जैसे maximum budget से ऊपर होना।
- **Weighted score:** Eligible properties को rank करने के लिए अलग-अलग factors का योगदान जोड़ता है।

उदाहरण: Budget 4000 NPR है तो 5000 NPR वाली property को budget hard filter हटाएगा, भले ही उसकी rating बहुत अच्छी हो।

## 5.10 Recommendation System की सीमाएँ

- Historical user clicks, saves, ratings या bookings से collaborative learning नहीं होती।
- Similarity score satisfaction की guarantee नहीं है।
- Results dataset की completeness और accuracy पर निर्भर करते हैं।
- Synthetic/sample records के कारण real-world relevance साबित नहीं होती।
- वास्तविक relevance जाँचने के लिए user feedback और ground-truth labels चाहिए।

---

# भाग 6 — Price Prediction और Gradient Boosting

## 6.1 Regression model क्या करता है?

Regression model numerical output predict करता है। यहाँ model prepared input features से `base_price_npr` का अनुमान लगाता है।

## 6.2 Gradient Boosting Regressor क्या है?

Gradient Boosting एक ensemble learning technique है। यह कई छोटे decision trees को क्रम से बनाता है। हर अगला tree पहले बने ensemble की errors को कम करने की कोशिश करता है। सभी trees का योगदान मिलकर final prediction बनाता है।

सरल भाषा में: पहला tree शुरुआती prediction करता है; अगला tree बची हुई गलती सुधारता है; फिर अगले trees सुधार जारी रखते हैं।

## 6.3 Decision Tree क्या है?

Decision tree data पर सवालों/conditions की श्रृंखला बनाकर prediction करता है। उदाहरण: capacity कितनी है? Neighbourhood कौन-सा है? Price किस range में है? इन splits के आधार पर output निकाला जाता है।

## 6.4 Project के Gradient Boosting Hyperparameters

| Parameter | Configured value | अर्थ |
|---|---:|---|
| `n_estimators` | 150 | Boosting stages/trees की संख्या |
| `learning_rate` | 0.05 | हर नए tree के contribution को scale करता है |
| `max_depth` | 3 | प्रत्येक tree की अधिकतम depth |
| `random_state` | 42 | Reproducible random behaviour में मदद |

**Learning rate कम करने पर** model अधिक धीरे-धीरे सीखता है; अक्सर अधिक estimators की जरूरत पड़ सकती है।  
**Max depth कम रखने पर** trees कम complex होते हैं और overfitting का जोखिम कम हो सकता है।

## 6.5 One-Hot Encoding क्या है?

Machine-learning algorithms सामान्यतः raw category names को सीधे numerical form में नहीं समझते। One-Hot Encoding category को binary indicator columns में बदलता है।

उदाहरण:

| Property type | Apartment | House | Hotel |
|---|---:|---:|---:|
| Apartment | 1 | 0 | 0 |
| House | 0 | 1 | 0 |
| Hotel | 0 | 0 | 1 |

Project में `OneHotEncoder(handle_unknown="ignore")` उपयोग किया गया है। इससे inference के समय unseen category आने पर encoder सीधे error नहीं देता। लेकिन उस नई category का learned category-specific pattern training में मौजूद नहीं होगा।

## 6.6 ColumnTransformer क्या है?

ColumnTransformer अलग-अलग columns या column groups पर अलग preprocessing लागू करता है। उदाहरण:
- Categorical columns पर One-Hot Encoding।
- Amenities text पर TF-IDF।
- Numerical columns को numerical रूप में pass करना।

## 6.7 Pipeline क्या है?

Pipeline preprocessing और model steps को एक reusable object में जोड़ती है।

इसका फायदा:
- Training और prediction में वही preprocessing लागू होती है।
- Code व्यवस्थित रहता है।
- अलग-अलग preprocessing steps भूलने का जोखिम कम होता है।
- Model को एक साथ save/load किया जा सकता है।

## 6.8 Project Price Pipeline

इस project की pipeline में:
1. Categorical features के लिए OneHotEncoder।
2. Amenities text के लिए TF-IDF।
3. Numerical features को passthrough।
4. अंतिम estimator के रूप में GradientBoostingRegressor।

## 6.9 बहुत महत्वपूर्ण — Price Model की Limitation

Price model में `nightly_price_npr` input feature के रूप में मौजूद है और target `base_price_npr` है। ये दोनों आपस में निकट रूप से संबंधित हैं। इसलिए model का task पहले से मौजूद nightly price को आधार बनाकर related target का अनुमान लगाना है।

इसे **baseline-adjustment experiment** कहना अधिक सही है। इसे independent real-world price discovery या पूरी तरह नए market price का prediction बताना गलत होगा।

---

# भाग 7 — Model Evaluation और Metrics

## 7.1 Model Evaluation क्यों जरूरी है?

Training के बाद यह जाँचना जरूरी है कि model unseen data पर कितना अच्छा काम करता है। केवल training score देखना पर्याप्त नहीं है।

## 7.2 MAE — Mean Absolute Error

MAE actual और predicted values के absolute differences का average है।

[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
]

सरल अर्थ: Prediction औसतन कितनी units से गलत है। Price के मामले में unit NPR है। MAE कम होना सामान्यतः बेहतर है।

## 7.3 RMSE — Root Mean Squared Error

[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
]

RMSE बड़ी errors को अधिक penalty देता है क्योंकि errors square की जाती हैं। यह भी NPR में व्यक्त होता है।

## 7.4 R-squared (R²)

R² model की predictions को target के mean-only baseline से compare करता है।

[
R^2=1-\frac{\sum (y_i-\hat{y}_i)^2}{\sum (y_i-\bar{y})^2}
]

- 1 के करीब: evaluated data पर बेहतर fit।
- 0: mean prediction के आसपास performance।
- Negative: mean prediction से भी खराब हो सकता है।

**R² = 0.9942 का अर्थ 99.42% accuracy नहीं है।** R² classification accuracy नहीं है और न ही real-world success की guarantee है।

## 7.5 Project के Documented Metrics

| Metric | Documented value |
|---|---:|
| MAE | NPR 148.33 |
| RMSE | NPR 218.93 |
| R² | 0.9942 |
| Baseline MAE | NPR 158.85 |
| MAE improvement | लगभग 6.62% |

MAE improvement की गणना:

[
\frac{158.85-148.33}{158.85}\times100\approx6.62\%
]

इसका अर्थ documented evaluation run में model का MAE baseline से लगभग 6.62% कम था। यह metric केवल उसी prepared evaluation dataset और split के संदर्भ में समझना चाहिए।

## 7.6 Baseline क्या है?

Baseline एक सरल reference prediction है जिससे complex model की तुलना की जाती है। इस project में existing `nightly_price_npr` को सीधे prediction के रूप में इस्तेमाल करके baseline MAE निकाला गया।

## 7.7 GroupShuffleSplit क्यों?

एक ही property की कई observations हो सकती हैं। सामान्य random split से एक property training और testing दोनों में आ सकती है। GroupShuffleSplit `property_id` के आधार पर पूरा group एक ही split में रखता है।

Documented split:
- Training: 808 rows, 320 properties।
- Testing: 192 rows, 80 properties।
- दोनों में shared property IDs: 0।

यह property-level leakage का जोखिम कम करता है।

## 7.8 Actual vs Predicted Plot

इस graph में:
- X-axis पर actual target values।
- Y-axis पर predicted values।
- Diagonal reference line के पास points हों तो predictions actual values के करीब हैं।
- Line से दूर points बड़ी errors दिखाते हैं।

Visual plot उपयोगी है, लेकिन metrics और data limitations का विकल्प नहीं है।

---

# भाग 8 — Python और Project Technologies

| Technology | Project में उपयोग |
|---|---|
| Python | मुख्य programming language |
| Pandas | CSV loading, cleaning, transformation और analysis |
| NumPy | Numerical operations और arrays |
| Scikit-learn | TF-IDF, cosine similarity, preprocessing, Gradient Boosting, splitting और metrics |
| Streamlit | Interactive dashboard |
| Joblib | Model artefacts save/load करना |
| Matplotlib / visualisation tools | Charts और graphs |
| Git | Source code का version control |
| GitHub | Repository hosting और collaboration |

## 8.1 Streamlit क्या है?

Streamlit Python framework है जिसकी मदद से Python code से interactive data applications और dashboards बनाए जा सकते हैं। इस project में यह पाँच workspaces वाला interface प्रदान करता है।

## 8.2 Joblib क्यों?

Training में fit हुए model और preprocessing objects को file में save करके बाद में reload किया जा सकता है। इससे application हर बार शुरू होने पर model को फिर से train नहीं करती।

## 8.3 Virtual Environment क्या है?

Virtual environment project-specific isolated Python environment है। इससे अलग projects की dependencies एक-दूसरे को प्रभावित नहीं करतीं।

## 8.4 requirements.txt क्या है?

यह file project की required Python packages और versions बताती है। इससे दूसरे computer पर dependencies install करना आसान होता है।

---

# भाग 9 — Repository और महत्वपूर्ण Files

| File / folder | भूमिका |
|---|---|
| `app.py` | Streamlit dashboard का मुख्य entry point |
| `src/advanced_recommender.py` | Advanced property recommendation logic |
| `src/train_price_model.py` | Price regression model training |
| `src/train_recommendation_model.py` | Recommendation model/features की तैयारी |
| `src/prepare_ml_datasets.py` | ML-ready datasets बनाना |
| `src/model_evaluation.py` | Model evaluation utilities |
| `data/processed/` | Cleaned और processed datasets |
| `reports/` | Reports और generated figures |
| `tests/` | Automated tests |
| `docs/` | Project documentation |

## 9.1 Application कैसे चलाएँ?

Windows PowerShell में repository के root folder में जाएँ:

```powershell
# अपने वास्तविक project folder में जाएँ
cd "C:\Users\Unknown Person\Desktop\AirBnb Intelligence"

# Virtual environment activate करें
.venv\Scripts\Activate.ps1

# यदि dependencies install नहीं हैं
pip install -r requirements.txt

# Dashboard चलाएँ
streamlit run app.py
```

यदि आपके computer पर project किसी दूसरे folder में है, तो `cd` के बाद वही वास्तविक path लिखें। `C:\path\to\...` केवल placeholder है, वास्तविक path नहीं।

---

# भाग 10 — Testing, Quality और Reproducibility

## 10.1 Testing क्या है?

Testing से जाँचा जाता है कि code अपेक्षा के अनुसार काम कर रहा है या नहीं।

- **Unit test:** किसी छोटे function/component को अलग से test करना।
- **Integration test:** कई components के साथ काम करने की जाँच।
- **Smoke test:** Application या मुख्य workflow शुरू हो रहा है या नहीं, इसकी basic जाँच।
- **Data validation:** Required columns, types, ranges और target values की जाँच।

## 10.2 Random State क्यों?

Random operations के लिए fixed `random_state` रखने से समान environment और data पर परिणाम दोबारा reproduce करना आसान होता है। यह हर तरह की hardware/library variation के बावजूद identical output की guarantee नहीं है।

## 10.3 Modular Architecture का लाभ

- Code पढ़ना और समझना आसान।
- हर module की अलग जिम्मेदारी।
- Testing आसान।
- Bugs को locate करना आसान।
- Future updates और maintenance सरल।

---

# भाग 11 — Dataset और Project की सीमाएँ

Viva में इन बातों को ईमानदारी से बताना बहुत जरूरी है:

1. Property और price datasets में synthetic/sample records हैं।
2. Dashboard verified live Airbnb inventory या real-time availability नहीं दिखाता।
3. Nepal tourist arrivals national context है; यह Kathmandu occupancy का direct measure नहीं है।
4. Cancellation dataset Portugal hotel-booking benchmark है, Kathmandu Airbnb cancellation model नहीं।
5. Current recommender collaborative filtering नहीं करता।
6. Recommendation scores probabilities नहीं हैं।
7. Price model existing nightly price को input लेकर related target estimate करता है; यह independent price discovery नहीं है।
8. High R² को “99.42% accuracy” नहीं कहना चाहिए।
9. वास्तविक deployment से पहले verified representative data, stronger validation, privacy/security checks और monitoring चाहिए।

**Future improvements:**
- Verified और regularly updated property data लेना।
- Price target को independent बनाना और known-price dependency हटाना।
- Real user feedback, clicks, saves या bookings से recommender evaluate करना।
- Precision@K, Recall@K और NDCG जैसे ranking metrics जोड़ना।
- Model drift और prediction errors monitor करना।
- Explainability और fairness checks जोड़ना।

---

# भाग 12 — 100 महत्वपूर्ण Viva Questions और Answers

## A. Project Basics (1–15)

**1. आपका project क्या है?**  
Airbnb Intelligence एक academic Machine Learning और Data Analytics application है जो Kathmandu accommodation-related data को analyse करता है, properties recommend करता है और prepared target price का estimate देता है।

**2. आपने यह project क्यों बनाया?**  
Accommodation data में location, price, property type और amenities जैसे कई factors होते हैं। इस project में इन data को analyse करके preference-based recommendations और price estimation का demonstration किया गया है।

**3. इसका मुख्य उद्देश्य क्या है?**  
Data preparation, analytics, property ranking, regression और dashboard को एक application में जोड़ना।

**4. इसमें कौन-सी language इस्तेमाल हुई?**  
Python।

**5. Dashboard किससे बनाया?**  
Streamlit से।

**6. कौन-सी ML library इस्तेमाल की?**  
Scikit-learn।

**7. मुख्य modules कौन-से हैं?**  
Executive Analytics, Property Recommendations, Price Prediction, Data Analysis और Model Evaluation।

**8. यह booking website है?**  
नहीं। यह analytics और ML demonstration application है, booking/payment platform नहीं।

**9. Recommendation और prediction में क्या अंतर है?**  
Recommendation items को relevance के अनुसार rank करती है। Prediction model input से target value का अनुमान लगाता है।

**10. Project में कौन-से मुख्य ML tasks हैं?**  
Hybrid content-based recommendation/ranking और supervised regression।

**11. Scikit-learn का क्या काम है?**  
TF-IDF, cosine similarity, preprocessing, Gradient Boosting, data splitting और evaluation metrics।

**12. Pandas का क्या काम है?**  
Tabular datasets को load, clean, transform और analyse करना।

**13. NumPy का क्या काम है?**  
Numerical calculations और array operations।

**14. Joblib का क्या काम है?**  
Trained model objects और artefacts को save/load करना।

**15. सबसे महत्वपूर्ण limitation क्या है?**  
Property और price data में synthetic samples हैं, इसलिए outputs verified real-world market information नहीं हैं।

## B. Data और Preprocessing (16–30)

**16. Data preprocessing क्या है?**  
Raw data को consistent और model-ready form में बदलने की प्रक्रिया।

**17. Missing data क्या है?**  
ऐसी value जो record में मौजूद नहीं है या unknown है।

**18. Missing values कैसे handle करते हैं?**  
उचित स्थिति में rows/columns हटाकर, suitable statistic/category से impute करके या missingness को अलग represent करके।

**19. Duplicate क्या है?**  
दोहराया गया record। इसे तभी हटाना चाहिए जब वह वास्तव में redundant हो।

**20. Feature engineering क्या है?**  
Existing data से उपयोगी model inputs बनाना, जैसे stay date से month और weekday निकालना।

**21. EDA क्या है?**  
Exploratory Data Analysis—statistics और charts से distribution, relationships, anomalies और data quality समझना।

**22. Median price क्यों उपयोगी है?**  
क्योंकि यह mean की तुलना में extreme prices से कम प्रभावित होता है।

**23. Outlier क्या है?**  
ऐसी observation जो बाकी data से काफी दूर हो।

**24. Normalisation क्या है?**  
Values को common scale पर लाने की transformation। Standardisation अलग प्रक्रिया है, जिसमें सामान्यतः mean और standard deviation का उपयोग होता है।

**25. Categorical data क्या है?**  
Labels या groups वाला data, जैसे neighbourhood और property type।

**26. Numerical data क्या है?**  
Numbers वाला data, जैसे price, capacity और rating।

**27. Target variable क्या है?**  
वह output जिसे supervised model predict करना सीखता है। यहाँ price model का target `base_price_npr` है।

**28. Feature क्या है?**  
Model को दिया गया input variable।

**29. अलग feature datasets क्यों बनाए?**  
Model inputs को व्यवस्थित रखने और unrelated columns से अलग करने के लिए।

**30. Data validation क्या है?**  
Training या inference से पहले required columns, types, ranges और target values की जाँच।

## C. Machine Learning Fundamentals (31–45)

**31. Machine Learning क्या है?**  
ऐसा क्षेत्र जहाँ algorithms data से patterns सीखकर predictions या decisions लेते हैं।

**32. Supervised learning क्या है?**  
ऐसी learning जिसमें examples के साथ input features और known target labels होते हैं।

**33. Unsupervised learning क्या है?**  
Target labels के बिना data में structure या patterns खोजना।

**34. Reinforcement learning क्या है?**  
Rewards और penalties के आधार पर actions सीखना।

**35. आपका price model supervised है या unsupervised?**  
Supervised, क्योंकि training examples में known numerical target मौजूद है।

**36. Price prediction classification है या regression?**  
Regression, क्योंकि output continuous numerical value है।

**37. Training set क्या है?**  
वह data जिस पर model fit होता है।

**38. Test set क्या है?**  
अलग रखा गया data जिससे unseen examples पर performance का अनुमान मिलता है।

**39. Overfitting क्या है?**  
जब model training data को बहुत अधिक fit करता है और unseen data पर खराब perform करता है।

**40. Underfitting क्या है?**  
जब model important patterns सीखने के लिए बहुत simple होता है।

**41. Generalisation क्या है?**  
Unseen data पर भी अच्छा perform करने की क्षमता।

**42. Data leakage क्या है?**  
जब अनुचित जानकारी training या evaluation को प्रभावित करती है और performance भ्रामक रूप से अच्छी दिखती है।

**43. Random split हमेशा सही क्यों नहीं होता?**  
Related observations training और testing दोनों में जा सकती हैं, जिससे test set स्वतंत्र नहीं रहता।

**44. Hyperparameter क्या है?**  
Training से पहले तय की गई setting, जैसे tree depth या estimators की संख्या।

**45. Ensemble model क्या है?**  
कई learners के outputs को combine करके final prediction देने वाला model।

## D. Recommendation System (46–65)

**46. Recommendation system क्या है?**  
ऐसा system जो user के लिए relevant items को rank या suggest करता है।

**47. Project कौन-सी recommendation method उपयोग करता है?**  
Text similarity और structured-feature weighted scoring को जोड़ने वाला hybrid content-based approach।

**48. Content-based filtering क्या है?**  
Item के attributes और user की पसंद के आधार पर recommendation करना।

**49. Collaborative filtering क्या है?**  
कई users के item interactions, ratings या purchases के patterns से recommendation करना।

**50. क्या project collaborative filtering करता है?**  
नहीं। यह user-item interaction matrix से train नहीं करता। यह content और preference signals उपयोग करता है।

**51. TF-IDF क्या है?**  
Text vectorisation method जो term frequency और document collection में term की rarity के आधार पर weight देता है।

**52. IDF क्या करता है?**  
बहुत सारे documents में आने वाले सामान्य शब्दों का weight कम करता है।

**53. Cosine similarity क्या है?**  
दो vectors की angle-based similarity मापने का तरीका।

**54. Cosine similarity क्यों इस्तेमाल की?**  
यह raw document length के बजाय vector direction की तुलना करती है।

**55. Bigram क्या है?**  
दो लगातार tokens का समूह।

**56. Stop words क्यों हटाते हैं?**  
बहुत सामान्य शब्द अक्सर कम distinguishing information देते हैं, इसलिए noise कम करने के लिए हटाए जाते हैं।

**57. Recommendation के components कौन-से हैं?**  
Text similarity, neighbourhood match, budget match, property-type match, quality score और capacity match।

**58. Weighted scoring क्यों?**  
क्योंकि हर preference final ranking में अलग contribution दे सकती है।

**59. Budget hard filter क्यों है?**  
Maximum budget एक constraint है। उससे महँगी property को केवल दूसरे scores अच्छे होने के कारण recommend नहीं करना चाहिए।

**60. Quality score कैसे बनता है?**  
Implementation में normalised rating का 70% और log-normalised review count का 30% योगदान है।

**61. Review count पर log क्यों लेते हैं?**  
बहुत बड़े review counts का प्रभाव कम करने और scale compress करने के लिए।

**62. Top-N recommendation क्या है?**  
सबसे अधिक score वाली N eligible properties लौटाना।

**63. Same property को फिर से recommend होने से कैसे रोकते हैं?**  
Similar-property method selected reference property को results से exclude करता है।

**64. Missing text values कैसे संभाली जाती हैं?**  
Advanced recommender vectorisation से पहले missing text को empty strings से भरता है।

**65. क्या recommendation scores probabilities हैं?**  
नहीं। ये weighted relevance scores हैं, calibrated probabilities या satisfaction की guarantee नहीं।

## E. Price Model और Evaluation (66–85)

**66. Price prediction में कौन-सा algorithm है?**  
Gradient Boosting Regressor।

**67. Gradient Boosting समझाएँ।**  
यह trees को sequentially बनाता है। हर अगला tree ensemble की बची हुई errors सुधारने की कोशिश करता है।

**68. Decision tree क्या है?**  
Feature-based conditions की श्रृंखला से prediction करने वाला model।

**69. Ensemble क्या है?**  
कई models या learners का combination।

**70. `n_estimators=150` का अर्थ?**  
Model में 150 boosting stages/trees इस्तेमाल होते हैं।

**71. `learning_rate=0.05` का अर्थ?**  
हर नए tree का contribution 0.05 के factor से scale होता है।

**72. `max_depth=3` का अर्थ?**  
हर individual regression tree की maximum depth 3 है।

**73. One-Hot Encoding क्या है?**  
Categorical values को binary indicator features में बदलना।

**74. Price pipeline में TF-IDF क्या करता है?**  
Amenities text को numerical features में बदलता है।

**75. ColumnTransformer क्या है?**  
अलग columns/groups पर अलग preprocessing operations लागू करता है।

**76. Pipeline क्या है?**  
Preprocessing और model steps को एक reusable estimator में जोड़ती है।

**77. MAE क्या है?**  
Actual और predicted values के absolute differences का average।

**78. RMSE क्या है?**  
Squared prediction errors के average का square root; बड़ी errors को अधिक penalty देता है।

**79. R-squared क्या है?**  
Residual error को target के mean के आसपास variation से compare करने वाला measure।

**80. Project के documented metrics क्या हैं?**  
MAE NPR 148.33, RMSE NPR 218.93 और R² 0.9942।

**81. Baseline क्या है?**  
Existing `nightly_price_npr` को सीधे prediction मानकर comparison करना।

**82. Model ने baseline से कितना improvement किया?**  
Documented MAE NPR 158.85 से NPR 148.33 हुआ—लगभग 6.62% improvement।

**83. GroupShuffleSplit क्यों?**  
एक property की सभी observations को training या testing में से केवल एक तरफ रखने के लिए, ताकि property-level overlap न हो।

**84. Price model की सबसे महत्वपूर्ण caveat क्या है?**  
यह existing nightly price को input बनाकर उससे संबंधित prepared target estimate करता है। यह independent market-price discovery नहीं है।

**85. क्या R² 0.9942 का मतलब 99.42% accuracy है?**  
नहीं। R² classification accuracy नहीं है। यह evaluated dataset पर mean-prediction reference की तुलना में fit का measure है।

## F. Application, Testing और Advanced (86–100)

**86. Streamlit क्या है?**  
Python framework जिससे interactive data applications बनाई जाती हैं।

**87. Data cache क्यों करते हैं?**  
बार-बार वही data load या computation करने से बचने और app तेज करने के लिए।

**88. `cache_data` और `cache_resource` में अंतर?**  
`cache_data` data results के लिए और `cache_resource` loaded models जैसे shared resources के लिए उपयोग होता है।

**89. Model save क्यों करते हैं?**  
ताकि application हर बार start होने पर training दोबारा न करनी पड़े।

**90. Virtual environment क्या है?**  
Project dependencies के लिए isolated Python environment।

**91. requirements.txt क्या है?**  
Project की dependencies और versions की सूची।

**92. Git क्या है?**  
Source code में changes track करने वाला version-control system।

**93. GitHub क्या है?**  
Git repositories host और collaborate करने का platform।

**94. Unit testing क्या है?**  
किसी छोटे function या component को अलग से test करना।

**95. Smoke test क्या है?**  
Application या मुख्य workflow की basic startup/operation जाँच।

**96. Modular architecture क्यों उपयोगी है?**  
यह responsibilities अलग करती है, code readable बनाती है और testing/maintenance आसान करती है।

**97. सबसे पहले क्या improve करेंगे?**  
Verified representative data लूँगा और price target/features को redesign करूँगा ताकि known baseline price पर निर्भरता कम हो।

**98. Recommendation quality कैसे improve करेंगे?**  
Real user feedback या relevance labels collect करके current hybrid method की अन्य approaches से तुलना करूँगा।

**99. क्या model आज real business decisions के लिए इस्तेमाल हो सकता है?**  
Verified representative data, उचित validation और deployment safeguards के बिना नहीं। अभी यह academic demonstration है।

**100. इस project से आपने क्या सीखा?**  
Data preparation, EDA, feature engineering, recommendation logic, regression, evaluation, dashboard development, testing, documentation और limitations को ईमानदारी से बताने का end-to-end ML workflow सीखा।

---

# भाग 13 — Advanced Viva Questions

## 13.1 Neural Network क्यों नहीं इस्तेमाल किया?

हर problem के लिए Deep Learning जरूरी नहीं है। Text matching के लिए TF-IDF और cosine similarity interpretable तथा efficient हैं। Tabular regression के लिए Gradient Boosting एक उपयुक्त विकल्प है। Algorithm का चुनाव data, task, evaluation और complexity देखकर करना चाहिए।

## 13.2 नई property add होने पर क्या होगा?

नई property के text को उसी feature space में represent करने के लिए text vectorizer और property feature matrix को update/rebuild करना होगा। Dashboard data और model artefacts consistent रहने चाहिए।

## 13.3 नई category आने पर क्या होगा?

Price pipeline में `OneHotEncoder(handle_unknown="ignore")` है, इसलिए unseen category आने पर encoder fail नहीं करेगा। लेकिन उस नई category का learned category-specific pattern training में उपलब्ध नहीं होगा।

## 13.4 Text और structured attributes दोनों क्यों?

Text amenities और descriptions की जानकारी देता है। Structured fields location, property type, budget और capacity जैसी स्पष्ट constraints बताते हैं। दोनों मिलकर text-only similarity से अधिक उपयोगी ranking दे सकते हैं।

## 13.5 High R² पर्याप्त क्यों नहीं है?

R² absolute error नहीं बताता, real-world generalisation साबित नहीं करता और target से निकटता से जुड़े input के कारण misleading हो सकता है। MAE, RMSE, baseline, data source और split design भी देखना जरूरी है।

## 13.6 Recommender का fair evaluation कैसे होगा?

इसके लिए ground-truth relevance चाहिए—जैसे user ratings, clicks, saves या bookings। Metrics में Precision@K, Recall@K, MAP और NDCG शामिल हो सकते हैं। Current scenario diagnostics इन ground-truth metrics के बराबर नहीं हैं।

## 13.7 Model drift क्या है?

समय के साथ input data या input-target relationship बदलने को model drift कहते हैं। Accommodation prices और user preferences बदल सकती हैं, इसलिए deployed model को monitor और re-evaluate करना चाहिए।

## 13.8 Explainability क्या है?

Model output के पीछे के कारणों को समझने योग्य बनाना explainability है। इस project में recommendation component scores से पता चल सकता है कि text, location, quality या capacity ने ranking में कितना योगदान दिया। Price model में feature importance या explanation methods जोड़े जा सकते हैं।

## 13.9 Training और Inference में अंतर?

- Training: Known targets वाले examples से model fit करना।
- Inference: Fitted model से नए inputs के outputs निकालना।

Dashboard मुख्यतः saved artefacts से inference करता है।

## 13.10 Tuning के दौरान test set को untouched क्यों रखना चाहिए?

Test set पर बार-बार tuning करने से model selection test information से प्रभावित हो जाता है और final score optimistic हो सकता है। Tuning training/validation या cross-validation से करें और final test evaluation अलग रखें।

---

# भाग 14 — Practical Viva Demonstration Script

यदि teacher app चलाने को कहें, तो यह क्रम अपनाएँ:

1. **Overview/Home:** Project का purpose और scope बताएं।
2. **Executive Analytics:** एक KPI, एक neighbourhood chart, एक price chart और tourism-context की limitation समझाएँ।
3. **Data Analysis:** Dataset दिखाकर rows, columns, data types और एक cleaning/feature-engineering step समझाएँ।
4. **Property Recommendations:** Neighbourhood, budget, guest count या amenity preference डालें। Hard filter और ranking score का अंतर बताएं।
5. **Similar Property:** Property select करके TF-IDF, cosine similarity और hybrid score समझाएँ।
6. **Price Prediction:** Inputs, target, Gradient Boosting और baseline caveat समझाएँ।
7. **Model Evaluation:** MAE, RMSE, R² और baseline improvement समझाएँ।
8. **Source Code:** `src/advanced_recommender.py` और `src/train_price_model.py` खोलकर वास्तविक methods/pipeline दिखाएँ।
9. **Testing:** केवल वही test/report दिखाएँ जो उपलब्ध है और वास्तव में run किया गया है।
10. **Conclusion:** Limitations और future improvements बताएं।

**यदि demonstration के समय app fail हो जाए:** घबराएँ नहीं। Workflow समझाएँ, संभावित dependency/data/model-file issue पहचानें और relevant source code या saved report दिखाएँ। ऐसा output आने का दावा न करें जो वास्तव में नहीं आया।

---

# भाग 15 — आखिरी रात की तैयारी का प्लान

## Priority 1: सबसे जरूरी (45 मिनट)

- 60-second project introduction।
- Problem statement, objectives और पाँच workspaces।
- End-to-end architecture।
- TF-IDF और cosine similarity।
- Hybrid recommendation components और weights।
- Gradient Boosting और hyperparameters।
- MAE, RMSE, R² और baseline improvement।
- GroupShuffleSplit और data leakage।
- Synthetic-data limitation और price-model caveat।

## Priority 2: Source Code (30 मिनट)

इन files को खोलकर हर file का input, processing और output अपने शब्दों में समझें:

- `app.py`
- `src/advanced_recommender.py`
- `src/train_price_model.py`
- `src/prepare_ml_datasets.py`
- `src/model_evaluation.py`

## Priority 3: बोलकर अभ्यास (30 मिनट)

Questions 1–30 बिना पढ़े जवाब दें। फिर Questions 46–85 का अभ्यास करें। 60-second introduction कम से कम तीन बार बोलें।

## Priority 4: याद रखने वाले तथ्य

- Dashboard के 5 workspaces।
- 400 property records; 354 published (documented scale)।
- TF-IDF + cosine similarity + weighted hybrid ranking।
- Recommendation के 6 components।
- Gradient Boosting Regressor।
- 150 estimators, learning rate 0.05, max depth 3।
- Group split: 320 training properties, 80 testing properties, zero overlap (documented run)।
- MAE NPR 148.33।
- RMSE NPR 218.93।
- R² 0.9942।
- Baseline MAE NPR 158.85।
- MAE improvement 6.62%।
- Synthetic sample data और baseline-price limitation।

---

# भाग 16 — जब किसी सवाल का जवाब न पता हो

यह तरीका अपनाएँ:

1. **ईमानदारी से स्वीकार करें:** “मुझे इस specific detail की पूरी पुष्टि नहीं है।”
2. **जो जानते हैं वह बताएं:** संबंधित algorithm या code से जोड़कर समझाएँ।
3. **जाँचने का तरीका बताएं:** बताएं कि आप इसे कैसे verify या test करेंगे।
4. **बनावटी दावा न करें:** Dataset source, metric या implemented module invent न करें।

उदाहरण:

“इस version में मैंने collaborative filtering implement नहीं किया है। Current recommender content-based है और TF-IDF text similarity को structured preference scores के साथ combine करता है। यदि future में पर्याप्त user-property interaction data मिलता है, तो collaborative filtering को explore किया जा सकता है।”

गलत algorithm का दावा करने से यह जवाब अधिक मजबूत है।

---

# भाग 17 — Final Viva Checklist

- [ ] मैं 60 seconds में project समझा सकता हूँ।
- [ ] मैं पाँचों dashboard workspaces समझा सकता हूँ।
- [ ] मुझे datasets और उनकी limitations पता हैं।
- [ ] मैं cleaning, EDA और feature engineering समझा सकता हूँ।
- [ ] मैं TF-IDF को सरल उदाहरण से समझा सकता हूँ।
- [ ] मैं cosine similarity और यह probability क्यों नहीं है, बता सकता हूँ।
- [ ] मैं recommendation के छह components समझा सकता हूँ।
- [ ] मैं hard filters और ranking weights का अंतर बता सकता हूँ।
- [ ] मैं Gradient Boosting और hyperparameters समझा सकता हूँ।
- [ ] मैं One-Hot Encoding, TF-IDF, ColumnTransformer और Pipeline समझा सकता हूँ।
- [ ] मैं GroupShuffleSplit और data leakage समझा सकता हूँ।
- [ ] मैं MAE, RMSE और R² की व्याख्या कर सकता हूँ।
- [ ] मैं baseline और 6.62% MAE improvement समझा सकता हूँ।
- [ ] मैं synthetic-data और baseline-price limitations स्पष्ट बता सकता हूँ।
- [ ] मैं मुख्य source files खोलकर methods पहचान सकता हूँ।
- [ ] मैं app का demonstration बिना unsupported claims के कर सकता हूँ।

## Teachers के सामने अंतिम statement

“यह project दिखाता है कि Data Preparation, Machine Learning, Recommendation Algorithms, Regression Evaluation और Interactive Visualisation को एक application में कैसे integrate किया जा सकता है। मैंने prepared data और current modelling approach की limitations भी पहचानी हैं। मेरा अगला कदम representative real-world data के साथ system को validate करना और अधिक independent prediction target तैयार करना होगा।”

---

**अध्ययन सुझाव:** पहले भाग 1, भाग 5, भाग 6 और भाग 7 अच्छी तरह समझें। Viva में केवल definitions याद करने के बजाय हर concept को अपने project के किसी वास्तविक module, file, input या output से जोड़कर समझाएँ।
