"""Fast isolated tests for dashboard data loading, recommendations, and evaluation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import file_cache_key, optional_file_cache_keys, read_csv_if_exists, read_optional_csv, read_report
from src.advanced_recommender import AdvancedPropertyRecommender
from src.model_evaluation import regression_metrics, ensure_group_disjoint


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

    def test_similar_recommendation_excludes_reference(self):
        result=self.engine.recommend_similar("a",number=10)
        self.assertNotIn("A",set(result.property_id.str.upper()))

    def test_invalid_minimum_rating_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.recommend_personalized({},number=5,min_rating=5.1)


class ModelEvaluationTests(unittest.TestCase):
    def test_regression_metrics_known_values(self):
        metrics=regression_metrics([3, -0.5, 2, 7],[2.5, 0, 2, 8])
        self.assertAlmostEqual(metrics["MAE"],0.5)
        self.assertAlmostEqual(metrics["RMSE"],np.sqrt(0.375))
        self.assertAlmostEqual(metrics["R2"],0.948608137)

    def test_regression_metrics_reject_empty_inputs(self):
        with self.assertRaisesRegex(ValueError,"at least one"):
            regression_metrics([],[])

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


if __name__ == "__main__":
    unittest.main()
