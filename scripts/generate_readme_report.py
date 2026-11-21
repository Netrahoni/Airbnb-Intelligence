"""Generate reproducible visual report figures from available project datasets and reports.

Run from repository root:
    python scripts/generate_readme_report.py

Figures are written to reports/figures/. Missing input files or columns are skipped.
This script does not fabricate data or train models.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.grid": True, "grid.alpha": 0.25})


def read_csv(path):
    if not path.exists():
        print(f"[SKIP] Missing: {path.relative_to(ROOT)}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[SKIP] Cannot read {path.name}: {exc}")
        return None


def save(fig, filename):
    fig.tight_layout()
    path = FIGURES / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path.relative_to(ROOT)}")


def first_col(frame, names):
    return next((name for name in names if name in frame.columns), None)


def numeric(frame, col):
    return pd.to_numeric(frame[col], errors="coerce")


def plot_properties():
    frame = read_csv(DATA / "properties_clean.csv")
    if frame is None or frame.empty:
        return
    price = first_col(frame, ["nightly_price_npr", "price_npr", "base_price_npr", "price"])
    neighbourhood = first_col(frame, ["neighbourhood", "neighborhood", "area"])
    ptype = first_col(frame, ["property_type", "accommodation_type", "listing_type"])

    if price:
        values = numeric(frame, price).dropna()
        values = values[values > 0]
        if not values.empty:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.hist(values, bins=30, edgecolor="white")
            ax.set(title="Sample Nightly Price Distribution", xlabel="Price (NPR)", ylabel="Records")
            save(fig, "price_distribution.png")

    if neighbourhood:
        counts = frame[neighbourhood].fillna("Unknown").value_counts().head(15).sort_values()
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            counts.plot.barh(ax=ax)
            ax.set(title="Top Neighbourhoods by Record Count", xlabel="Records", ylabel="")
            save(fig, "neighbourhood_listing_counts.png")

        if price:
            temp = frame.assign(_price=numeric(frame, price)).dropna(subset=["_price"])
            medians = temp.groupby(neighbourhood)["_price"].median().sort_values(ascending=False).head(15).sort_values()
            if not medians.empty:
                fig, ax = plt.subplots(figsize=(9, 5.5))
                medians.plot.barh(ax=ax)
                ax.set(title="Median Sample Nightly Price by Neighbourhood", xlabel="Median price (NPR)", ylabel="")
                save(fig, "neighbourhood_median_prices.png")

    if ptype:
        counts = frame[ptype].fillna("Unknown").value_counts().head(12).sort_values()
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(8.5, 5))
            counts.plot.barh(ax=ax)
            ax.set(title="Property Type Composition", xlabel="Records", ylabel="")
            save(fig, "property_type_composition.png")


def plot_tourism():
    frame = read_csv(DATA / "nepal_monthly_tourist_arrivals_clean.csv")
    if frame is None or frame.empty:
        return
    date_col = first_col(frame, ["month_date", "date", "month"])
    value_col = first_col(frame, ["tourist_arrivals", "arrivals", "visitor_count", "total_arrivals"])
    if not date_col or not value_col:
        print("[SKIP] Tourism columns not recognized.")
        return
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[value_col] = numeric(frame, value_col)
    frame = frame.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(frame[date_col], frame[value_col], marker="o", linewidth=1.7)
    ax.set(title="Nepal Monthly Tourist Arrivals (National Context)", xlabel="Month", ylabel="Arrivals")
    ax.tick_params(axis="x", rotation=35)
    save(fig, "tourist_arrivals_trend.png")


def plot_price_reports():
    pred = read_csv(REPORTS / "price_predictions.csv")
    if pred is not None and not pred.empty:
        actual_col = first_col(pred, ["actual_price_npr"])
        predicted_col = first_col(pred, ["predicted_price_npr"])
        if actual_col and predicted_col:
            actual, estimated = numeric(pred, actual_col), numeric(pred, predicted_col)
            valid = pd.DataFrame({"actual": actual, "predicted": estimated}).dropna()
            if not valid.empty:
                fig, ax = plt.subplots(figsize=(6.5, 6))
                ax.scatter(valid.actual, valid.predicted, alpha=0.65)
                low = min(valid.actual.min(), valid.predicted.min())
                high = max(valid.actual.max(), valid.predicted.max())
                ax.plot([low, high], [low, high], linestyle="--")
                ax.set(title="Actual vs Predicted Price", xlabel="Actual price (NPR)", ylabel="Predicted price (NPR)")
                save(fig, "actual_vs_predicted.png")

                errors = valid.predicted - valid.actual
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.hist(errors, bins=30, edgecolor="white")
                ax.axvline(0, linestyle="--")
                ax.set(title="Prediction Error Distribution", xlabel="Predicted − Actual (NPR)", ylabel="Test records")
                save(fig, "prediction_error_distribution.png")

                if "absolute_error_npr" in pred.columns:
                    abs_err = numeric(pred, "absolute_error_npr").dropna()
                    if not abs_err.empty:
                        fig, ax = plt.subplots(figsize=(8, 4.5))
                        ax.boxplot(abs_err, vert=True)
                        ax.set(title="Absolute Prediction Error", ylabel="Absolute error (NPR)")
                        save(fig, "absolute_prediction_error.png")

    metrics = read_csv(REPORTS / "price_model_metrics.csv")
    if metrics is not None and not metrics.empty:
        name_col = first_col(metrics, ["metric", "name", "metric_name"])
        value_col = first_col(metrics, ["value", "metric_value", "score"])
        if name_col and value_col:
            m = metrics.copy()
            m[value_col] = numeric(m, value_col)
            selected = m[m[name_col].astype(str).str.upper().isin(["MAE", "RMSE", "R2", "BASELINE MAE", "BASELINE RMSE", "BASELINE R2"])]
            if not selected.empty:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                selected.set_index(name_col)[value_col].plot.bar(ax=ax)
                ax.set(title="Saved Price Model Metrics", ylabel="Metric value")
                save(fig, "price_model_metrics.png")

    comparison = read_csv(REPORTS / "price_model_comparison.csv")
    if comparison is not None and not comparison.empty:
        model_col = first_col(comparison, ["model", "model_name"])
        if model_col:
            for metric in ["MAE", "RMSE", "R2", "R²"]:
                if metric in comparison.columns:
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    comparison.set_index(model_col)[metric].plot.bar(ax=ax)
                    ax.set(title=f"Model Comparison — {metric}", ylabel=metric)
                    save(fig, f"model_comparison_{metric.replace('²','2')}.png")

    cv = read_csv(REPORTS / "price_model_cross_validation.csv")
    if cv is not None and not cv.empty:
        fold_col = first_col(cv, ["fold", "cv_fold"])
        if fold_col:
            for metric in ["GB_MAE", "GB_RMSE", "GB_R2"]:
                if metric in cv.columns:
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    cv.set_index(fold_col)[metric].plot.bar(ax=ax)
                    ax.set(title=f"Cross-Validation — {metric}", xlabel="Fold", ylabel=metric)
                    save(fig, f"cross_validation_{metric.lower()}.png")


def plot_recommendation_reports():
    weight = read_csv(REPORTS / "recommendation_weight_comparison.csv")
    if weight is not None and not weight.empty:
        config = first_col(weight, ["configuration"])
        if config:
            metrics = [
                ("average_score", "Average Recommendation Score"),
                ("neighbourhood_match", "Neighbourhood Match (%)"),
                ("property_type_match", "Property Type Match (%)"),
                ("budget_compliance", "Budget Compliance (%)"),
                ("capacity_sufficiency", "Capacity Sufficiency (%)"),
            ]
            for col, title in metrics:
                if col in weight.columns:
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    weight.set_index(config)[col].plot.bar(ax=ax)
                    ax.set(title=f"Recommendation Weight Experiment — {title}", ylabel=title)
                    save(fig, f"recommendation_weights_{col}.png")

    score = read_csv(REPORTS / "recommendation_score_distribution.csv")
    if score is not None and not score.empty and "score" in score.columns:
        stat_col = first_col(score, ["statistic", "label"])
        if stat_col:
            fig, ax = plt.subplots(figsize=(7.5, 4))
            score.set_index(stat_col)["score"].plot.bar(ax=ax)
            ax.set(title="Recommendation Score Summary", ylabel="Score")
            save(fig, "recommendation_score_summary.png")

    personalized = read_csv(REPORTS / "recommendation_personalized_quality.csv")
    if personalized is not None and not personalized.empty:
        score_col = first_col(personalized, ["average_recommendation_score"])
        if score_col:
            values = numeric(personalized, score_col).dropna()
            if not values.empty:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.hist(values, bins=20, edgecolor="white")
                ax.set(title="Personalized Recommendation Score Distribution", xlabel="Average score", ylabel="Scenarios")
                save(fig, "personalized_recommendation_scores.png")

    similarity = read_csv(REPORTS / "recommendation_similarity_quality.csv")
    if similarity is not None and not similarity.empty:
        col = first_col(similarity, ["average_similarity_score"])
        if col:
            values = numeric(similarity, col).dropna()
            if not values.empty:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.hist(values, bins=20, edgecolor="white")
                ax.set(title="Similar-Property Score Distribution", xlabel="Similarity score", ylabel="Scenarios")
                save(fig, "similar_property_scores.png")


def plot_cancellation():
    frame = read_csv(DATA / "booking_cancellations_clean.csv")
    if frame is None or frame.empty:
        return
    target = first_col(frame, ["is_canceled", "is_cancelled", "canceled", "cancelled"])
    if not target:
        print("[SKIP] Cancellation target column not found.")
        return
    counts = frame[target].value_counts(dropna=False).sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts.plot.bar(ax=ax)
    ax.set(title="Hotel Booking Benchmark — Cancellation Target Distribution", xlabel="Target value", ylabel="Rows")
    save(fig, "cancellation_target_distribution.png")


def main():
    print("Generating README visual report figures from available artifacts...")
    plot_properties()
    plot_tourism()
    plot_price_reports()
    plot_recommendation_reports()
    plot_cancellation()
    generated = sorted(p.name for p in FIGURES.glob("*.png"))
    print(f"\nFinished. Figures generated: {len(generated)}")
    for filename in generated:
        print(" -", filename)
    print("Only generated figures should be committed and embedded in the README.")
    print("Review each chart's data provenance before presenting it as a result.")


if __name__ == "__main__":
    main()
