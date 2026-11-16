"""
Kathmandu Airbnb ML Analysis
Model 01: Content-based accommodation recommender.

Important provenance:
The supplied property dataset labels source_name as "Synthetic Sample Data".
Outputs are demonstrations on synthetic property records, not live Airbnb listings.
"""
from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DATA = Path(__file__).parent / "properties_clean.csv"
DEFAULT_MODEL = Path(__file__).parent / "accommodation_recommender.joblib"

class AccommodationRecommender:
    def __init__(self):
        self.properties = None
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.text_matrix = None

    @staticmethod
    def _truthy(series):
        return series.astype(str).str.lower().isin(["true", "1", "yes"])

    def fit(self, csv_path):
        df = pd.read_csv(csv_path)
        required = [
            "property_id", "property_name", "neighbourhood", "property_type",
            "room_type", "capacity", "bedrooms", "amenities",
            "nightly_price_npr", "rating", "review_count", "description",
            "published"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        df["published"] = self._truthy(df["published"])
        for col in ["capacity", "bedrooms", "nightly_price_npr", "rating", "review_count"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["amenities"] = df["amenities"].fillna("").astype(str)
        df["description"] = df["description"].fillna("").astype(str)
        df["neighbourhood"] = df["neighbourhood"].fillna("").astype(str)
        df["property_type"] = df["property_type"].fillna("").astype(str)
        df["room_type"] = df["room_type"].fillna("").astype(str)

        # Only published properties are eligible for recommendations.
        df = df[df["published"]].copy().reset_index(drop=True)
        if df.empty:
            raise ValueError("No published properties are available.")

        # Text representation combines amenities and description.
        df["_text"] = (
            df["amenities"].str.replace(";", " ", regex=False) + " " +
            df["description"]
        )
        self.text_matrix = self.vectorizer.fit_transform(df["_text"])
        self.properties = df
        return self

    @staticmethod
    def _normalise(values):
        values = pd.Series(values, dtype="float64").fillna(0).clip(lower=0)
        maximum = values.max()
        return values / maximum if maximum > 0 else values * 0

    def recommend(
        self, neighbourhood=None, property_type=None, room_type=None,
        guests=None, bedrooms=None, max_budget=None, amenities=None, top_n=10
    ):
        if self.properties is None:
            raise RuntimeError("Call fit() or load a saved model first.")
        df = self.properties
        mask = pd.Series(True, index=df.index)

        # Hard constraints: capacity, bedroom minimum, and maximum nightly budget.
        if guests is not None:
            mask &= df["capacity"].fillna(0) >= int(guests)
        if bedrooms is not None:
            mask &= df["bedrooms"].fillna(0) >= int(bedrooms)
        if max_budget is not None:
            mask &= df["nightly_price_npr"].fillna(float("inf")) <= float(max_budget)

        candidates = df.loc[mask].copy()
        if candidates.empty:
            return candidates.assign(relevance_score=pd.Series(dtype=float))

        # Optional preference scores. Missing preference dimensions are excluded
        # from the weighted average rather than treated as a mismatch.
        scores = pd.DataFrame(index=candidates.index)
        weights = {}

        if neighbourhood:
            scores["location"] = candidates["neighbourhood"].str.casefold().eq(
                str(neighbourhood).casefold()
            ).astype(float)
            weights["location"] = 0.20

        if property_type:
            scores["property_type"] = candidates["property_type"].str.casefold().eq(
                str(property_type).casefold()
            ).astype(float)
            weights["property_type"] = 0.15

        if room_type:
            scores["room_type"] = candidates["room_type"].str.casefold().eq(
                str(room_type).casefold()
            ).astype(float)
            weights["room_type"] = 0.10

        if amenities:
            query = " ".join(amenities) if not isinstance(amenities, str) else amenities
            q_vec = self.vectorizer.transform([query])
            sim = cosine_similarity(q_vec, self.text_matrix[candidates.index]).ravel()
            scores["amenities_description"] = sim
            weights["amenities_description"] = 0.40

        if max_budget is not None:
            # Budget is already a hard cap; this score prefers options nearer
            # the user's budget without exceeding it.
            denom = max(float(max_budget), 1.0)
            scores["budget_fit"] = 1 - (
                (float(max_budget) - candidates["nightly_price_npr"]).clip(lower=0) / denom
            )
            weights["budget_fit"] = 0.15

        if scores.shape[1]:
            total_weight = sum(weights.values())
            relevance = sum(scores[col] * weights[col] for col in scores.columns) / total_weight
        else:
            relevance = pd.Series(0.0, index=candidates.index)

        candidates["relevance_score"] = relevance.clip(0, 1)
        candidates = candidates.sort_values(
            ["relevance_score", "rating", "review_count"],
            ascending=[False, False, False]
        )
        output_cols = [
            "property_id", "property_name", "neighbourhood", "property_type",
            "room_type", "capacity", "bedrooms", "nightly_price_npr",
            "rating", "review_count", "amenities", "relevance_score"
        ]
        return candidates[output_cols].head(int(top_n)).reset_index(drop=True)

    def save(self, path=DEFAULT_MODEL):
        joblib.dump(self, path)

    @staticmethod
    def load(path=DEFAULT_MODEL):
        return joblib.load(path)

def main():
    parser = argparse.ArgumentParser(description="Recommend Kathmandu accommodation.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--neighbourhood")
    parser.add_argument("--property-type")
    parser.add_argument("--room-type")
    parser.add_argument("--guests", type=int)
    parser.add_argument("--bedrooms", type=int)
    parser.add_argument("--max-budget", type=float)
    parser.add_argument("--amenities", nargs="*")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    model = AccommodationRecommender().fit(args.data)
    if args.save_model:
        model.save(DEFAULT_MODEL)
        print(f"Saved model: {DEFAULT_MODEL}")
    results = model.recommend(
        neighbourhood=args.neighbourhood,
        property_type=args.property_type,
        room_type=args.room_type,
        guests=args.guests,
        bedrooms=args.bedrooms,
        max_budget=args.max_budget,
        amenities=args.amenities,
        top_n=args.top_n,
    )
    if results.empty:
        print("No properties match the hard constraints. Try increasing the budget or reducing guest/bedroom requirements.")
    else:
        print(results.to_string(index=False))

if __name__ == "__main__":
    main()
