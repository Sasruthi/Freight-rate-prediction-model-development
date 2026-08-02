"""
Generate final predictions using the saved production model.

Usage:
    python src/predict.py

Produces (at repo root, matching the assessment's required filenames):
    validation_predictions.csv
    data/december_chart_inputs.csv   (predicted_rate column filled in place)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import engineer_features, get_feature_columns  # noqa: E402

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"


def one_hot(df: pd.DataFrame, feature_cols: list[str], ohe_columns: list[str]) -> pd.DataFrame:
    out = df[feature_cols].copy()
    cat_cols = [c for c in out.columns if str(out[c].dtype) == "category"]
    out = pd.get_dummies(out, columns=cat_cols, dummy_na=False)
    return out.reindex(columns=ohe_columns, fill_value=0)


def load_artifacts():
    model = joblib.load(MODELS / "best_model.joblib")
    pre = joblib.load(MODELS / "preprocessor.joblib")
    with open(MODELS / "feature_columns.json") as f:
        cols = json.load(f)
    with open(MODELS / "model_kind.json") as f:
        kind_info = json.load(f)
    ohe_columns = None
    ohe_path = MODELS / "ohe_columns.joblib"
    if ohe_path.exists():
        ohe_columns = joblib.load(ohe_path)
    numeric_medians = joblib.load(MODELS / "numeric_medians.joblib")
    return model, pre, cols["feature_cols"], cols["cat_cols"], kind_info, ohe_columns, numeric_medians


def predict_frame(df: pd.DataFrame, model, pre, feature_cols, cat_cols, kind_info, ohe_columns, numeric_medians):
    feat = engineer_features(df, pre)
    # some engineered columns may not exist for this frame (e.g. haversine
    # for december); keep the column set identical to training either way
    for c in feature_cols:
        if c not in feat.columns:
            feat[c] = np.nan
    X = feat[feature_cols]

    if kind_info["kind"] == "ohe":
        # Linear/Ridge/RF (one-hot path) can't take NaN: impute any
        # optional numeric column that's fully or partially absent in this
        # frame (e.g. lat/lon on december_chart_inputs.csv) with the
        # training-set median. Tree models below handle NaN natively and
        # don't need this.
        feat_ohe_input = feat.copy()
        for c, med in numeric_medians.items():
            if c in feat_ohe_input.columns:
                feat_ohe_input[c] = feat_ohe_input[c].fillna(med)
        X_ohe = one_hot(feat_ohe_input, feature_cols, ohe_columns)
        log_pred = model.predict(X_ohe)
    elif kind_info["kind"] == "catboost":
        from catboost import Pool
        cat_idx = [X.columns.get_loc(c) for c in cat_cols]
        pool = Pool(X, cat_features=cat_idx)
        log_pred = model.predict(pool)
    else:  # native (xgboost / lightgbm)
        log_pred = model.predict(X)

    return np.expm1(log_pred)


def main():
    model, pre, feature_cols, cat_cols, kind_info, ohe_columns, numeric_medians = load_artifacts()
    print(f"Loaded production model: {kind_info['model_name']}")

    # ---------------- validation.csv -> validation_predictions.csv ----------------
    val = pd.read_csv(DATA / "validation.csv")
    val_pred = predict_frame(val, model, pre, feature_cols, cat_cols, kind_info, ohe_columns, numeric_medians)
    val_pred = np.clip(val_pred, 1.0, None)  # scorer requires strictly positive rates

    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    pred_map = dict(zip(val["load_id"], val_pred))
    template["predicted_rate"] = template["load_id"].map(pred_map).round(2)
    assert template["predicted_rate"].notna().all(), "missing predictions for some load_id"
    assert (template["predicted_rate"] > 0).all()

    out_path = ROOT / "validation_predictions.csv"
    template.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(template)} rows)")

    # ---------------- december_chart_inputs.csv (fill in place) ----------------
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")
    dec_pred = predict_frame(dec, model, pre, feature_cols, cat_cols, kind_info, ohe_columns, numeric_medians)
    dec["predicted_rate"] = np.clip(dec_pred, 1.0, None).round(2)
    dec.to_csv(DATA / "december_chart_inputs.csv", index=False)
    print(f"Filled {DATA / 'december_chart_inputs.csv'} ({len(dec)} rows)")
    print(dec[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
