from pathlib import Path
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "recommendation_features.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

print("\n========== IMPROVED PROPERTY RECOMMENDATION MODEL ==========")

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

print("\nOriginal properties:", len(df))

# 2. Keep only published properties
df["published"] = df["published"].astype(str).str.lower().eq("true")
df["verified"] = df["verified"].astype(str).str.lower().eq("true")

df = df[df["published"]].copy()
df = df.reset_index(drop=True)

print("Published properties:", len(df))
print("Unpublished properties excluded:", 400 - len(df))

# 3. Prepare text features
text_columns = [
    "city",
    "neighbourhood",
    "property_type",
    "room_type",
    "amenities",
    "description"
]

for column in text_columns:
    df[column] = df[column].fillna("").astype(str)

# 4. Prepare numerical characteristics
df["capacity_feature"] = df["capacity"].fillna(0).astype(int).apply(
    lambda x: f"capacity_{x}"
)

df["bedroom_feature"] = df["bedrooms"].fillna(0).astype(int).apply(
    lambda x: f"bedrooms_{x}"
)

df["bathroom_feature"] = df["bathrooms"].fillna(0).astype(int).apply(
    lambda x: f"bathrooms_{x}"
)

def price_category(price):
    if price <= 2500:
        return "budget_price"
    elif price <= 5000:
        return "midrange_price"
    elif price <= 9000:
        return "premium_price"
    return "luxury_price"

df["price_feature"] = df["nightly_price_npr"].apply(price_category)

# 5. Combine features
feature_columns = text_columns + [
    "capacity_feature",
    "bedroom_feature",
    "bathroom_feature",
    "price_feature"
]

df["combined_features"] = df[feature_columns].agg(" ".join, axis=1)

# 6. Convert text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

feature_matrix = vectorizer.fit_transform(df["combined_features"])

# 7. Calculate similarity
similarity_matrix = cosine_similarity(feature_matrix)

print("\nTF-IDF matrix:", feature_matrix.shape)
print("Similarity matrix:", similarity_matrix.shape)

# 8. Save model
model = {
    "vectorizer": vectorizer,
    "feature_matrix": feature_matrix,
    "similarity_matrix": similarity_matrix,
    "properties": df
}

model_path = MODEL_DIR / "property_recommender.joblib"
joblib.dump(model, model_path)

print("\nUpdated model saved:", model_path)

# 9. Test recommendation
property_id = input(
    "\nEnter a property ID (example: KTM001): "
).strip().upper()

matches = df.index[df["property_id"].astype(str) == property_id].tolist()

if not matches:
    print("\nProperty not found among published properties.")
else:
    index = matches[0]
    selected = df.iloc[index]

    scores = [
        (i, score)
        for i, score in enumerate(similarity_matrix[index])
        if i != index
    ]

    scores.sort(key=lambda item: item[1], reverse=True)
    top_results = scores[:5]

    print("\n========== SELECTED PROPERTY ==========")
    print("Name:", selected["property_name"])
    print("Neighbourhood:", selected["neighbourhood"])
    print("Price: NPR", selected["nightly_price_npr"])

    print("\n========== TOP 5 RECOMMENDATIONS ==========")

    for rank, (i, score) in enumerate(top_results, start=1):
        item = df.iloc[i]

        print(f"\n{rank}. {item['property_name']}")
        print("ID:", item["property_id"])
        print("Neighbourhood:", item["neighbourhood"])
        print("Property type:", item["property_type"])
        print("Room type:", item["room_type"])
        print("Price: NPR", item["nightly_price_npr"])
        print("Rating:", item["rating"])
        print("Verified:", item["verified"])
        print("Similarity:", f"{score * 100:.2f}%")

print("\nImproved recommendation model completed.")
