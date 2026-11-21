"""Fast isolated tests for dashboard data loading, recommendations, and evaluation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import file_cache_key, optional_file_cache_keys, read_csv_if_exists, read_optional_csv, read_report
from src.advanced_recommender import AdvancedPropertyRecommender
from src.model_evaluation import regression_metrics, ensure_group_disjoint, price_band_diagnostics


def sample_properties():
    return pd.DataFrame([
        {"property_id":"A","property_name":"A Stay","neighbourhood":"Thamel","property_type":"Hotel","room_type":"Private room","capacity":2,"nightly_price_npr":2000,"rating":4.5,"review_count":20,"amenities":"wifi kitchen","description":"quiet central stay","published":True},
        {"property_id":"B","property_name":"B Stay","neighbourhood":"Thamel","property_type":"Hotel","room_type":"Entire place","capacity":4,"nightly_price_npr":3500,"rating":4.8,"review_count":50,"amenities":"wifi breakfast","description":"bright city hotel","published":True},
        {"property_id":"C","property_name":"C Stay","neighbourhood":"Patan","property_type":"Apartment","room_type":"Entire place","capacity":1,"nightly_price_npr":1500,"rating":3.8,"review_count":5,"amenities":"garden","description":"quiet garden room","published":True},
        {"property_id":"D","property_name":"Draft","neighbourhood":"Patan","property_type":"Apartment","room_type":"Private room","capacity":2,"nightly_price_npr":1000,"rating":5.0,"review_count":1,"amenities":"wifi","description":"draft listing","published":False},
    ])


class DashboardDataLoadingTests(unittest.TestCase):
    def test_missing_csv_returns_empty_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = read_csv_if_exists(Path(directory) / "missing.csv")
        self.assertTrue(frame.empty)

    def test_csv_loads_expected_rows_and_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "properties.csv"
            pd.DataFrame({"property_id":["P1","P2"],"price":[100,200]}).to_csv(path,index=False)
            frame = read_csv_if_exists(path)
        self.assertEqual(len(frame), 2)
        self.assertEqual(list(frame.columns), ["property_id","price"])

    def test_optional_csv_respects_filename_and_directory_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed, raw = root/"processed", root/"raw"
            processed.mkdir(); raw.mkdir()
            pd.DataFrame({"source":["raw"]}).to_csv(raw/"fallback.csv",index=False)
            pd.DataFrame({"source":["preferred"]}).to_csv(processed/"preferred.csv",index=False)
            frame = read_optional_csv(["preferred.csv","fallback.csv"],[processed,raw])
        self.assertEqual(frame.loc[0,"source"],"preferred")

    def test_optional_csv_returns_empty_when_no_candidate_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = read_optional_csv(["one.csv","two.csv"],[directory])
        self.assertTrue(frame.empty)

    def test_optional_csv_continues_after_unreadable_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"bad.csv").write_text("a,b\n1,2\n3,4,5\n", encoding="utf-8")
            pd.DataFrame({"ok":[1]}).to_csv(root/"good.csv",index=False)
            warnings=[]
            frame=read_optional_csv(["bad.csv","good.csv"],[root],lambda name,error:warnings.append(name))
        self.assertEqual(frame.loc[0,"ok"],1)
        self.assertEqual(warnings,["bad.csv"])

    def test_report_missing_and_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"report.csv"
            self.assertTrue(read_report(path).empty)
            pd.DataFrame({"MAE":[12.5]}).to_csv(path,index=False)
            self.assertAlmostEqual(read_report(path).loc[0,"MAE"],12.5)

    def test_cache_signatures_track_file_presence_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"data.csv"
            self.assertIsNone(file_cache_key(path))
            path.write_text("a\\n1\\n", encoding="utf-8")
            first=file_cache_key(path)
            path.write_text("a\\n123\\n", encoding="utf-8")
            second=file_cache_key(path)
            self.assertIsNotNone(first)
            self.assertNotEqual(first,second)
            signatures=optional_file_cache_keys(["data.csv"],[directory])
            self.assertEqual(len(signatures),1)
            self.assertEqual(signatures[0][1],second)


class RecommendationEdgeCaseTests(unittest.TestCase):
    def setUp(self):
        self.engine=AdvancedPropertyRecommender(sample_properties())

    def test_unpublished_properties_are_excluded(self):
        self.assertNotIn("D",set(self.engine.properties.property_id))

    def test_missing_required_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Missing required columns"):
            AdvancedPropertyRecommender(sample_properties().drop(columns=["capacity"]))

    def test_blank_text_features_do_not_break_recommendations(self):
        frame=sample_properties()
        frame["amenities"]=""
        frame["description"]=""
        engine=AdvancedPropertyRecommender(frame)
        result=engine.recommend_personalized({},number=3)
        self.assertFalse(result.empty)
        self.assertTrue(np.isfinite(result["Recommendation Score (%)"]).all())

    def test_budget_and_capacity_constraints_are_strict(self):
        result=self.engine.recommend_personalized({"max_price":2500,"guests":2},number=10)
        self.assertTrue((result.nightly_price_npr<=2500).all())
        self.assertTrue((result.capacity>=2).all())
        self.assertNotIn("C",set(result.property_id))

    def test_impossible_budget_returns_empty_results(self):
        result=self.engine.recommend_personalized({"max_price":500,"guests":2},number=10)
        self.assertTrue(result.empty)

    def test_maximum_budget_rewards_affordability_not_spending_closer_to_limit(self):
        scores = self.engine._budget_score(2500)
        by_id = dict(zip(self.engine.properties["property_id"], scores))
        self.assertEqual(by_id["A"], 1.0)
        self.assertEqual(by_id["C"], 1.0)
        self.assertEqual(by_id["B"], 0.0)

    def test_budget_score_is_neutral_when_no_budget_is_provided(self):
        scores = self.engine._budget_score()
        self.assertTrue(np.all(scores == 0.5))

    def test_neighbourhood_preference_changes_top_recommendation(self):
        self.engine._text_similarity = lambda query: np.full(
            len(self.engine.properties), 0.5
        )
        self.engine._quality_score = lambda: np.full(
            len(self.engine.properties), 0.5
        )

        thamel = self.engine.recommend_personalized(
            {"neighbourhood": "Thamel"}, number=3
        )
        patan = self.engine.recommend_personalized(
            {"neighbourhood": "Patan"}, number=3
        )

        self.assertEqual(thamel.iloc[0]["property_id"], "B")
        self.assertEqual(patan.iloc[0]["property_id"], "C")

    def test_property_type_preference_changes_top_recommendation(self):
        self.engine._text_similarity = lambda query: np.full(
            len(self.engine.properties), 0.5
        )
        self.engine._quality_score = lambda: np.full(
            len(self.engine.properties), 0.5
        )

        hotel = self.engine.recommend_personalized(
            {"property_type": "Hotel"}, number=3
        )
        apartment = self.engine.recommend_personalized(
            {"property_type": "Apartment"}, number=3
        )

        self.assertEqual(hotel.iloc[0]["property_id"], "B")
        self.assertEqual(apartment.iloc[0]["property_id"], "C")

    def test_room_type_preference_changes_top_recommendation(self):
        self.engine._text_similarity = lambda query: np.full(
            len(self.engine.properties), 0.5
        )
        self.engine._quality_score = lambda: np.full(
            len(self.engine.properties), 0.5
        )

        private_room = self.engine.recommend_personalized(
            {"room_type": "Private room"}, number=3
        )
        entire_place = self.engine.recommend_personalized(
            {"room_type": "Entire place"}, number=3
        )

        self.assertEqual(private_room.iloc[0]["property_id"], "A")
        self.assertEqual(entire_place.iloc[0]["property_id"], "B")

    def test_multi_value_neighbourhood_preference_matches_any_selection(self):
        components = self.engine.recommend_personalized(
            {"neighbourhood": [" thamel ", "PATAN"]}, number=3
        )

        self.assertTrue((components["neighbourhood_match"] == 1.0).all())

    def test_multi_value_property_and_room_preferences_match_any_selection(self):
        result = self.engine.recommend_personalized(
            {
                "property_type": [" hotel ", "APARTMENT"],
                "room_type": ["PRIVATE ROOM", " entire place "],
            },
            number=3,
        )

        self.assertTrue((result["property_type_match"] == 1.0).all())

    def test_explicit_neighbourhood_preference_outweighs_unrelated_text_and_quality(self):
        self.engine._text_similarity = lambda query: np.array([0.0, 0.0, 1.0])
        self.engine._quality_score = lambda: np.array([0.0, 0.0, 1.0])

        result = self.engine.recommend_personalized(
            {"neighbourhood": "Thamel"}, number=3
        )

        self.assertEqual(result.iloc[0]["property_id"], "B")

    def test_personalized_weights_normalize_only_explicit_ranking_signals(self):
        weights = self.engine._personalized_weights(
            {"neighbourhood": "Thamel"}
        )

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["neighbourhood_match"], 0.22 / 0.32)
        self.assertAlmostEqual(weights["quality_score"], 0.10 / 0.32)
        self.assertNotIn("text_similarity", weights)
        self.assertNotIn("budget_match", weights)
        self.assertNotIn("property_type_match", weights)
        self.assertNotIn("capacity_match", weights)

    def test_categorical_preferences_are_not_repeated_in_text_query(self):
        captured = []
        self.engine._text_similarity = lambda query: (
            captured.append(query) or np.full(len(self.engine.properties), 0.5)
        )

        self.engine.recommend_personalized(
            {
                "neighbourhood": "Thamel",
                "property_type": "Hotel",
                "room_type": "Private room",
            },
            number=3,
        )

        self.assertEqual(captured, [""])

    def test_descriptive_and_guest_preferences_activate_their_own_weights(self):
        weights = self.engine._personalized_weights(
            {"amenities": ["wifi", "kitchen"], "guests": 2}
        )

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["text_similarity"], 0.20 / 0.45)
        self.assertAlmostEqual(weights["capacity_match"], 0.15 / 0.45)
        self.assertAlmostEqual(weights["quality_score"], 0.10 / 0.45)
        self.assertNotIn("neighbourhood_match", weights)
        self.assertNotIn("property_type_match", weights)
        self.assertNotIn("budget_match", weights)

    def test_equal_scores_use_property_id_as_deterministic_tiebreaker(self):
        frame = sample_properties().iloc[[2, 1, 0]].copy()
        frame["rating"] = 4.0
        frame["review_count"] = 10
        engine = AdvancedPropertyRecommender(frame)
        engine._text_similarity = lambda query: np.full(
            len(engine.properties), 0.5
        )
        engine._quality_score = lambda: np.full(
            len(engine.properties), 0.5
        )

        result = engine.recommend_personalized({}, number=3)

        self.assertEqual(
            result["property_id"].tolist(),
            ["A", "B", "C"],
        )

    def test_recommendation_scores_are_finite_and_bounded(self):
        result = self.engine.recommend_personalized({}, number=10)
        scores = result["Recommendation Score (%)"].to_numpy(dtype=float)

        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 100)).all())

    def test_similar_recommendation_excludes_reference(self):
        result=self.engine.recommend_similar("a",number=10)
        self.assertNotIn("A",set(result.property_id.str.upper()))

    def test_invalid_minimum_rating_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.recommend_personalized({},number=5,min_rating=5.1)

    def test_non_finite_minimum_rating_is_rejected(self):
        for value in (float("nan"), float("inf"), "not-a-rating"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.engine.recommend_personalized({}, min_rating=value)

    def test_recommendation_count_must_be_a_positive_integer(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.engine.recommend_personalized({}, number=value)


class ModelEvaluationTests(unittest.TestCase):
    def test_price_band_diagnostics_reconcile_to_overall_mae(self):
        predictions = pd.DataFrame({
            "actual_price_npr": [1000, 2500, 4500, 7000, 0],
            "predicted_price_npr": [1100, 2000, 5000, 6000, 100],
        })
        bands = price_band_diagnostics(predictions)
        self.assertEqual(int(bands["sample_count"].sum()), len(predictions))
        weighted_mae = np.average(
            bands["mean_absolute_error"], weights=bands["sample_count"]
        )
        overall_mae = regression_metrics(
            predictions["actual_price_npr"], predictions["predicted_price_npr"]
        )["MAE"]
        self.assertAlmostEqual(weighted_mae, overall_mae)

    def test_price_band_diagnostics_excludes_zero_actual_from_percentage_error(self):
        predictions = pd.DataFrame({
            "actual_price_npr": [0, 1000],
            "predicted_price_npr": [100, 1100],
        })
        bands = price_band_diagnostics(predictions)
        self.assertAlmostEqual(bands["mean_percentage_error"].iloc[0], 10.0)

    def test_price_band_diagnostics_requires_prediction_columns(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            price_band_diagnostics(pd.DataFrame({"actual_price_npr": [1000]}))

    def test_regression_metrics_known_values(self):
        metrics=regression_metrics([3, -0.5, 2, 7],[2.5, 0, 2, 8])
        self.assertAlmostEqual(metrics["MAE"],0.5)
        self.assertAlmostEqual(metrics["RMSE"],np.sqrt(0.375))
        self.assertAlmostEqual(metrics["R2"],0.948608137)

    def test_regression_metrics_reject_empty_inputs(self):
        with self.assertRaisesRegex(ValueError,"at least one"):
            regression_metrics([],[])

    def test_regression_metrics_reject_single_observation(self):
        with self.assertRaisesRegex(ValueError,"at least two"):
            regression_metrics([1000], [1100])

    def test_regression_metrics_reject_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError,"same length"):
            regression_metrics([1,2],[1])

    def test_regression_metrics_reject_non_finite_values(self):
        with self.assertRaisesRegex(ValueError,"finite"):
            regression_metrics([1,np.nan],[1,2])

    def test_group_split_is_disjoint(self):
        counts=ensure_group_disjoint(["a","a","b","c"],[0,1],[2,3])
        self.assertEqual(counts,(1,2))

    def test_group_split_detects_leakage(self):
        with self.assertRaisesRegex(ValueError,"leakage"):
            ensure_group_disjoint(["a","b","a","c"],[0,1],[2,3])



class PriceTrainingDataValidationTests(unittest.TestCase):

    def test_price_training_data_accepts_valid_grouped_rows(self):
        from src.model_evaluation import validate_price_training_data

        frame = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5],
            "lead_time_days": [1, 2, 3, 4, 5],
            "base_price_npr": [1000, 1200, 1300, 1500, 1800],
            "property_id": ["A", "B", "C", "D", "E"],
        })
        count = validate_price_training_data(frame, ["feature", "lead_time_days"])
        self.assertEqual(count, 5)

    def test_price_training_data_rejects_non_positive_targets(self):
        from src.model_evaluation import validate_price_training_data

        frame = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5],
            "base_price_npr": [1000, 0, 1300, 1500, 1800],
            "property_id": ["A", "B", "C", "D", "E"],
        })
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            validate_price_training_data(frame, ["feature"])

    def test_price_training_data_rejects_negative_lead_time(self):
        from src.model_evaluation import validate_price_training_data

        frame = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5],
            "lead_time_days": [1, -1, 3, 4, 5],
            "base_price_npr": [1000, 1200, 1300, 1500, 1800],
            "property_id": ["A", "B", "C", "D", "E"],
        })
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_price_training_data(frame, ["feature", "lead_time_days"])

    def test_price_training_data_requires_enough_distinct_properties(self):
        from src.model_evaluation import validate_price_training_data

        frame = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5],
            "base_price_npr": [1000, 1200, 1300, 1500, 1800],
            "property_id": ["A", "A", "B", "B", "C"],
        })
        with self.assertRaisesRegex(ValueError, "at least 5 distinct"):
            validate_price_training_data(frame, ["feature"])

    def test_price_training_data_rejects_missing_features(self):
        from src.model_evaluation import validate_price_training_data

        frame = pd.DataFrame({
            "base_price_npr": [1000, 1200, 1300, 1500, 1800],
            "property_id": ["A", "B", "C", "D", "E"],
        })
        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate_price_training_data(frame, ["unknown_feature"])


class DataQualityProfileTests(unittest.TestCase):
    def test_profile_counts_missing_invalid_and_out_of_range_values(self):
        from src.data_quality import profile_dataframe

        frame = pd.DataFrame({
            "property_id": ["A", "A", "C"],
            "rating": [4.5, 6.0, np.nan],
            "nightly_price_npr": [1200, "unknown", -5],
        })
        report = profile_dataframe(
            frame,
            "properties",
            key_columns=("property_id",),
            numeric_bounds={"rating": (0, 5), "nightly_price_npr": (0, None)},
        ).set_index("column")

        self.assertEqual(report.loc["property_id", "duplicate_key_rows"], 1)
        self.assertEqual(report.loc["rating", "missing_count"], 1)
        self.assertEqual(report.loc["rating", "out_of_range_count"], 1)
        self.assertEqual(report.loc["nightly_price_npr", "invalid_numeric_count"], 1)
        self.assertEqual(report.loc["nightly_price_npr", "out_of_range_count"], 1)

    def test_profile_rejects_missing_key_columns(self):
        from src.data_quality import profile_dataframe

        with self.assertRaisesRegex(ValueError, "missing key columns"):
            profile_dataframe(pd.DataFrame({"name": ["Stay"]}), "properties", ("property_id",))

    def test_empty_report_has_stable_schema(self):
        from src.data_quality import build_data_quality_report

        report = build_data_quality_report([])
        self.assertIn("missing_percent", report.columns)
        self.assertTrue(report.empty)

    def test_report_combines_multiple_datasets(self):
        from src.data_quality import build_data_quality_report

        report = build_data_quality_report([
            ("properties", pd.DataFrame({"id": [1]}), {}),
            ("availability", pd.DataFrame({"days": [30]}), {}),
        ])
        self.assertEqual(set(report["dataset"]), {"properties", "availability"})


if __name__ == "__main__":
    unittest.main()
