
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class AdvancedPropertyRecommender:
    """
    Hybrid property recommendation engine.

    Features:
    - Amenities and description similarity
    - Neighbourhood matching
    - Strict maximum budget filtering
    - Property and room type matching
    - Rating and review quality
    - Guest capacity matching
    """

    WEIGHTS = {
    "text_similarity": 0.20,
    "neighbourhood_match": 0.22,
    "budget_match": 0.15,
    "property_type_match": 0.18,
    "quality_score": 0.10,
    "capacity_match": 0.15,
}

    def __init__(self, properties):

        self.properties = properties.copy()

        # --------------------------------------------------
        # 1. FILTER PUBLISHED PROPERTIES
        # --------------------------------------------------

        if "published" in self.properties.columns:

            published = (
                self.properties["published"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )

            self.properties = self.properties[published].copy()

        self.properties = self.properties.reset_index(drop=True)

        if self.properties.empty:
            raise ValueError("No published properties available.")

        # --------------------------------------------------
        # 2. VALIDATE REQUIRED COLUMNS
        # --------------------------------------------------

        required_columns = [
            "property_id",
            "property_name",
            "neighbourhood",
            "property_type",
            "room_type",
            "capacity",
            "nightly_price_npr",
            "rating",
            "review_count",
            "amenities",
            "description",
        ]

        missing = [
            column for column in required_columns
            if column not in self.properties.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # --------------------------------------------------
        # 3. CLEAN TEXT DATA
        # --------------------------------------------------

        text_columns = [
            "property_id",
            "property_name",
            "neighbourhood",
            "property_type",
            "room_type",
            "amenities",
            "description",
        ]

        for column in text_columns:

            self.properties[column] = (
                self.properties[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        # --------------------------------------------------
        # 4. CLEAN NUMERIC DATA
        # --------------------------------------------------

        numeric_columns = [
            "capacity",
            "nightly_price_npr",
            "rating",
            "review_count",
        ]

        for column in numeric_columns:

            self.properties[column] = pd.to_numeric(
                self.properties[column],
                errors="coerce",
            )

        # Invalid prices cannot be recommended.
        self.properties["nightly_price_npr"] = (
            self.properties["nightly_price_npr"]
            .fillna(0)
        )

        self.properties["capacity"] = (
            self.properties["capacity"].fillna(0)
        )

        self.properties["rating"] = (
            self.properties["rating"].fillna(0)
        )

        self.properties["review_count"] = (
            self.properties["review_count"].fillna(0)
        )

        # --------------------------------------------------
        # 5. BUILD TEXT FEATURES
        # --------------------------------------------------

        self.properties["recommendation_text"] = (
            self.properties["amenities"]
            + " "
            + self.properties["description"]
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )

        try:

            self.feature_matrix = self.vectorizer.fit_transform(
                self.properties["recommendation_text"]
            )

        except ValueError:

            # Handle datasets containing no usable text.
            self.vectorizer = None

            self.feature_matrix = None

        self.max_reviews = max(
            float(self.properties["review_count"].max()),
            1.0,
        )

    # ======================================================
    # HELPER FUNCTIONS
    # ======================================================

    @staticmethod
    def _normalise(value):

        return str(value).strip().lower()

    @staticmethod
    def _positive_number(value, name):

        try:
            value = float(value)

        except (TypeError, ValueError):

            raise ValueError(
                f"{name} must be a valid number."
            )

        if not np.isfinite(value) or value <= 0:

            raise ValueError(
                f"{name} must be greater than zero."
            )

        return value

    def _text_similarity(self, query):

        if not query or not str(query).strip():

            return np.full(
                len(self.properties),
                0.5,
            )

        if self.vectorizer is None:

            return np.full(
                len(self.properties),
                0.5,
            )

        query_vector = self.vectorizer.transform(
            [str(query)]
        )

        return cosine_similarity(
            query_vector,
            self.feature_matrix,
        ).flatten()

    def _quality_score(self):

        ratings = (
            self.properties["rating"].to_numpy(dtype=float)
            / 5.0
        )

        ratings = np.clip(ratings, 0, 1)

        reviews = self.properties[
            "review_count"
        ].to_numpy(dtype=float)

        review_scores = (
            np.log1p(np.maximum(reviews, 0))
            / np.log1p(self.max_reviews)
        )

        return (
            0.70 * ratings
            + 0.30 * review_scores
        )

    def _budget_score(self, max_budget=None):

        prices = self.properties[
            "nightly_price_npr"
        ].to_numpy(dtype=float)

        if max_budget is None:

            return np.full(
                len(self.properties),
                0.5,
            )

        # A maximum budget is an affordability ceiling, not a target spend.
        # Every eligible listing receives full budget compliance; listings
        # above the ceiling are filtered out again by the hard constraint.
        return np.where(
            prices <= max_budget,
            1.0,
            0.0,
        )

    def _calculate_final_score(self, components):

        return sum(
            components[name] * weight
            for name, weight in self.WEIGHTS.items()
        )

    # ======================================================
    # RESULT PROCESSING
    # ======================================================

    def _make_results(
    self,
    scores,
    components,
    excluded_index=None,
    number=5,
    max_budget=None,
    min_rating=0,
    min_capacity=None,
):
        if number <= 0:

            raise ValueError(
                "Number of recommendations must be positive."
            )

        if min_rating < 0 or min_rating > 5:

            raise ValueError(
                "Minimum rating must be between 0 and 5."
            )

        results = self.properties.copy()

        # Attach recommendation components.
        results["text_similarity"] = components[
            "text_similarity"
        ]

        results["neighbourhood_match"] = components[
            "neighbourhood_match"
        ]

        results["budget_match"] = components[
            "budget_match"
        ]

        results["property_type_match"] = components[
            "property_type_match"
        ]

        results["quality_score"] = components[
            "quality_score"
        ]

        results["capacity_match"] = components[
            "capacity_match"
        ]

        results["recommendation_score"] = (
            np.asarray(scores) * 100
        )

        # --------------------------------------------------
        # 1. REMOVE REFERENCE PROPERTY
        # --------------------------------------------------

        if excluded_index is not None:

            results = results.drop(
                index=excluded_index,
                errors="ignore",
            )

        # --------------------------------------------------
        # 2. APPLY HARD BUDGET CONSTRAINT
        # --------------------------------------------------

        if max_budget is not None:

            results = results[
                results["nightly_price_npr"] <= max_budget
            ]

        # --------------------------------------------------
        # 3. APPLY MINIMUM RATING CONSTRAINT
        # --------------------------------------------------

        results = results[
            results["rating"] >= min_rating
        ]
        # --------------------------------------------------
        # 4. APPLY STRICT GUEST CAPACITY CONSTRAINT
        # --------------------------------------------------

        if min_capacity is not None:

             results = results[
                 results["capacity"] >= min_capacity
                 ]

        # --------------------------------------------------
        # 4. REMOVE INVALID PRICES
        # --------------------------------------------------

        results = results[
            results["nightly_price_npr"] > 0
        ]

        # --------------------------------------------------
        # 5. SORT AND SELECT TOP RECOMMENDATIONS
        # --------------------------------------------------

        results = results.sort_values(
            by=[
                "recommendation_score",
                "rating",
                "review_count",
            ],
            ascending=[False, False, False],
        )

        results = results.head(number).copy()

        # --------------------------------------------------
        # 6. FORMAT OUTPUT COLUMNS
        # --------------------------------------------------

        results["Recommendation Score (%)"] = (
            results["recommendation_score"].round(2)
        )

        results["Text Similarity (%)"] = (
            results["text_similarity"] * 100
        ).round(2)

        results["Neighbourhood Match (%)"] = (
            results["neighbourhood_match"] * 100
        ).round(2)

        results["Budget Match (%)"] = (
            results["budget_match"] * 100
        ).round(2)

        results["Quality Score (%)"] = (
            results["quality_score"] * 100
        ).round(2)

        results["Capacity Match (%)"] = (
            results["capacity_match"] * 100
        ).round(2)

        return results.reset_index(drop=True)

    # ======================================================
    # SIMILAR PROPERTY RECOMMENDATIONS
    # ======================================================

    def recommend_similar(
        self,
        property_id,
        number=5,
        max_budget=None,
        min_rating=0,
    ):

        matches = self.properties.index[
            self.properties["property_id"].str.lower()
            == self._normalise(property_id)
        ].tolist()

        if not matches:

            raise ValueError(
                f"Property ID '{property_id}' was not found."
            )

        selected_index = matches[0]

        selected = self.properties.iloc[selected_index]

        # Text similarity.
        text_scores = cosine_similarity(
            self.feature_matrix[selected_index],
            self.feature_matrix,
        ).flatten() if self.feature_matrix is not None else np.full(
            len(self.properties), 0.5
        )

        # Neighbourhood similarity.
        neighbourhood_scores = (
            self.properties["neighbourhood"]
            .str.lower()
            .eq(self._normalise(selected["neighbourhood"]))
            .astype(float)
            .to_numpy()
        )

        # Relative price similarity.
        reference_price = max(
            float(selected["nightly_price_npr"]),
            1.0,
        )

        candidate_prices = np.maximum(
            self.properties["nightly_price_npr"].to_numpy(
                dtype=float
            ),
            1.0,
        )

        budget_scores = np.exp(
            -np.abs(
                np.log(candidate_prices / reference_price)
            )
        )

        # Property and room type matching.
        property_matches = (
            self.properties["property_type"]
            .str.lower()
            .eq(self._normalise(selected["property_type"]))
            .astype(float)
            .to_numpy()
        )

        room_matches = (
            self.properties["room_type"]
            .str.lower()
            .eq(self._normalise(selected["room_type"]))
            .astype(float)
            .to_numpy()
        )

        type_scores = (
            property_matches + room_matches
        ) / 2

        # Quality score.
        quality_scores = self._quality_score()

        # Capacity similarity.
        reference_capacity = max(
            float(selected["capacity"]),
            1.0,
        )

        candidate_capacity = np.maximum(
            self.properties["capacity"].to_numpy(dtype=float),
            0,
        )

        capacity_scores = np.maximum(
            0,
            1 - (
                np.abs(
                    candidate_capacity - reference_capacity
                )
                / np.maximum(
                    candidate_capacity,
                    reference_capacity,
                )
            ),
        )

        components = {
            "text_similarity": text_scores,
            "neighbourhood_match": neighbourhood_scores,
            "budget_match": budget_scores,
            "property_type_match": type_scores,
            "quality_score": quality_scores,
            "capacity_match": capacity_scores,
        }

        final_scores = self._calculate_final_score(
            components
        )

        if max_budget is not None:

            max_budget = self._positive_number(
                max_budget,
                "Maximum budget",
            )

        return self._make_results(
            final_scores,
            components,
            excluded_index=selected_index,
            number=number,
            max_budget=max_budget,
            min_rating=min_rating,
        )

    # ======================================================
    # PERSONALIZED RECOMMENDATIONS
    # ======================================================

    def recommend_personalized(
        self,
        preferences,
        number=5,
        min_rating=0,
    ):

        if not isinstance(preferences, dict):

            raise TypeError(
                "Preferences must be provided as a dictionary."
            )

        # --------------------------------------------------
        # 1. BUILD GUEST SEARCH QUERY
        # --------------------------------------------------

        query_parts = []

        for field in [
            "neighbourhood",
            "property_type",
            "room_type",
            "amenities",
            "description",
        ]:

            value = preferences.get(field)

            if value:

                if isinstance(value, (list, tuple, set)):

                    value = " ".join(
                        map(str, value)
                    )

                query_parts.append(str(value))

        query = " ".join(query_parts)

        text_scores = self._text_similarity(query)

        # --------------------------------------------------
        # 2. NEIGHBOURHOOD MATCHING
        # --------------------------------------------------

        neighbourhood = preferences.get("neighbourhood")

        if neighbourhood:

            neighbourhood_scores = (
                self.properties["neighbourhood"]
                .str.lower()
                .eq(self._normalise(neighbourhood))
                .astype(float)
                .to_numpy()
            )

        else:

            neighbourhood_scores = np.full(
                len(self.properties),
                0.5,
            )

        # --------------------------------------------------
        # 3. MAXIMUM BUDGET
        # --------------------------------------------------

        max_budget = preferences.get("max_price")

        if max_budget is not None:

            max_budget = self._positive_number(
                max_budget,
                "Maximum budget",
            )

        budget_scores = self._budget_score(
            max_budget
        )

        # --------------------------------------------------
        # 4. PROPERTY AND ROOM TYPE MATCHING
        # --------------------------------------------------

        type_scores = []

        property_type = preferences.get("property_type")

        room_type = preferences.get("room_type")

        if property_type:

            type_scores.append(
                self.properties["property_type"]
                .str.lower()
                .eq(self._normalise(property_type))
                .astype(float)
                .to_numpy()
            )

        if room_type:

            type_scores.append(
                self.properties["room_type"]
                .str.lower()
                .eq(self._normalise(room_type))
                .astype(float)
                .to_numpy()
            )

        if type_scores:

            property_type_scores = np.mean(
                type_scores,
                axis=0,
            )

        else:

            property_type_scores = np.full(
                len(self.properties),
                0.5,
            )

        # --------------------------------------------------
        # 5. QUALITY SCORE
        # --------------------------------------------------

        quality_scores = self._quality_score()

        # --------------------------------------------------
        # 6. GUEST CAPACITY MATCHING
        # --------------------------------------------------

        guests = preferences.get("guests")

        if guests is not None:

            guests = self._positive_number(
                guests,
                "Number of guests",
            )

            capacities = self.properties[
                "capacity"
            ].to_numpy(dtype=float)

            capacity_scores = np.clip(
                capacities / guests,
                0,
                1,
            )

        else:

            capacity_scores = np.full(
                len(self.properties),
                0.5,
            )

        # --------------------------------------------------
        # 7. COMBINE HYBRID RECOMMENDATION SCORES
        # --------------------------------------------------

        components = {
            "text_similarity": text_scores,
            "neighbourhood_match": neighbourhood_scores,
            "budget_match": budget_scores,
            "property_type_match": property_type_scores,
            "quality_score": quality_scores,
            "capacity_match": capacity_scores,
        }

        final_scores = self._calculate_final_score(
            components
        )

        # --------------------------------------------------
        # 8. APPLY HARD CONSTRAINTS BEFORE TOP-N SELECTION
        # --------------------------------------------------

        return self._make_results(
    final_scores,
    components,
    number=number,
    max_budget=max_budget,
    min_rating=min_rating,
    min_capacity=guests,
)
