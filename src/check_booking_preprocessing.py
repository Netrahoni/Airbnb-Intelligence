"""Step 2 check: run the preprocessing on the real data and report what happens."""
from pathlib import Path

import numpy as np
import pandas as pd
from booking_preprocessing import (
    ROOT, TARGET, load_booking_data, remove_invalid_rows, build_features,
    make_preprocessor, stratified_split, time_based_split,
)

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

lines = []
def say(text=""):
    print(text); lines.append(str(text))

raw = load_booking_data()
say(f"Raw rows: {len(raw):,}")
df, counts = remove_invalid_rows(raw)
say(f"Invalid rows removed: {counts}")
say(f"Rows used for modelling: {len(df):,}")
say(f"Cancellation rate after removal: {df[TARGET].mean()*100:.2f}%")

X, y = build_features(df)
say(f"\nFeature table X shape: {X.shape}")
say("Feature columns: " + ", ".join(X.columns))
banned = {"booking_id","booking_date","arrival_date","total_nights",
          "adr_is_negative","guest_count_is_zero","adr_iqr_outlier_flag",TARGET}
say(f"Banned/leaky columns present in X: {sorted(banned & set(X.columns)) or 'none'}")

X_train, X_test, y_train, y_test = stratified_split(X, y)
say(f"\nStratified split -> train {len(X_train):,} rows, test {len(X_test):,} rows")
say(f"Cancellation rate: train {y_train.mean()*100:.2f}%  test {y_test.mean()*100:.2f}%")

pre = make_preprocessor()
Xtr = pre.fit_transform(X_train)          # fit on TRAIN only
Xte = pre.transform(X_test)               # only transform the test set
names = pre.get_feature_names_out()
say(f"\nTransformed train shape: {Xtr.shape}; test shape: {Xte.shape}")
say(f"Any NaN/inf in train: {not np.isfinite(Xtr).all()}; in test: {not np.isfinite(Xte).all()}")
say(f"Output features: {len(names)} "
    f"({sum(n.startswith('log_num') for n in names)} log-scaled numeric, "
    f"{sum(n.startswith('num__') for n in names)} scaled numeric, "
    f"{sum(n.startswith('cat') for n in names)} one-hot)")
tr = pd.DataFrame(Xtr, columns=names)
say("\nTrain means/stds of numeric outputs (should be ~0 and ~1):")
say(tr.filter(regex='^(log_num|num)__').agg(['mean','std']).T.round(3).to_string())

# Risk measured: identical feature rows appearing in both train and test
key = lambda d: d.astype(str).agg("|".join, axis=1)
tr_keys = set(key(X_train)); te_keys = key(X_test)
share = te_keys.isin(tr_keys).mean()
say(f"\nTest rows whose exact feature values also appear in train: {share*100:.1f}%")

# Unseen-category safety test
probe = X_test.head(3).copy(); probe["market_segment"] = "SomethingNew"
say(f"Unseen category handled without error: {pre.transform(probe).shape}")

# Time-based split sizes
a, b = time_based_split(df)
say(f"\nTime-based split -> train {len(a):,} (arrivals up to {a['arrival_date'].max()[:10]}), "
    f"test {len(b):,} (from {b['arrival_date'].min()[:10]})")
say(f"Cancellation rate: train {a[TARGET].mean()*100:.2f}%  test {b[TARGET].mean()*100:.2f}%")

# Without deposit_type
Xn, _ = build_features(df, use_deposit_type=False)
pn = make_preprocessor(use_deposit_type=False).fit(Xn.loc[X_train.index])
say(f"\nWithout deposit_type: {Xn.shape[1]} feature columns -> {len(pn.get_feature_names_out())} output features")

(REPORT_DIR / "booking_preprocessing_check.txt").write_text("\n".join(lines), encoding="utf-8")
