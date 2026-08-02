"""
Train and compare several regression models for freight rate prediction.

Usage:
    python src/train.py

Produces:
    models/best_model.joblib          -- the winning model, refit on 100% of train_test.csv
    models/preprocessor.joblib        -- fitted imputation stats (refit on 100%)
    models/feature_columns.json       -- feature list the model expects
    models/model_kind.json            -- which model family won ("lightgbm", "catboost", ...)
    reports/model_comparison.csv       -- holdout metrics for every candidate model
    reports/figures/model_comparison.png
    reports/figures/holdout_actual_vs_pred.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from features import Preprocessor, engineer_features, get_feature_columns  # noqa: E402

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
for d in (MODELS, REPORTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def evaluate(y_true, y_pred, name):
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def one_hot(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """One-hot encode categoricals for the linear / RF baselines."""
    out = df[feature_cols].copy()
    cat_cols = [c for c in out.columns if str(out[c].dtype) == "category"]
    return pd.get_dummies(out, columns=cat_cols, dummy_na=False)


def align_columns(train_ohe: pd.DataFrame, other_ohe: pd.DataFrame) -> pd.DataFrame:
    return other_ohe.reindex(columns=train_ohe.columns, fill_value=0)


def main():
    print("Loading data...")
    df = pd.read_csv(DATA / "train_test.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ---------------------------------------------------------------
    # Time-based split. The real deployment gap is large: train_test.csv
    # covers Jan-Oct 2025, but validation.csv/december_chart_inputs.csv
    # require predictions for Nov-Dec 2025 (1-3 months beyond the training
    # window). A random row split would let the model "see" market
    # conditions from dates immediately neighbouring the ones it's scored
    # on, which overstates real-world accuracy. Instead we hold out the
    # most recent ~15% of days as a proxy for that forecast gap.
    # ---------------------------------------------------------------
    cutoff = df["date"].quantile(0.85)
    train_df = df[df["date"] <= cutoff].copy()
    holdout_df = df[df["date"] > cutoff].copy()
    print(f"Train: {len(train_df)} rows ({train_df.date.min().date()} - {train_df.date.max().date()})")
    print(f"Holdout: {len(holdout_df)} rows ({holdout_df.date.min().date()} - {holdout_df.date.max().date()})")

    pre = Preprocessor().fit(train_df)
    train_feat = engineer_features(train_df, pre)
    holdout_feat = engineer_features(holdout_df, pre)

    feature_cols = get_feature_columns(train_feat)
    cat_cols = [c for c in feature_cols if str(train_feat[c].dtype) == "category"]
    print(f"Using {len(feature_cols)} features ({len(cat_cols)} categorical): {feature_cols}")

    y_train = np.log1p(train_feat["posted_rate"].values)
    y_holdout_true = holdout_feat["posted_rate"].values

    X_train = train_feat[feature_cols]
    X_holdout = holdout_feat[feature_cols]

    results = []
    fitted_models = {}

    # ---------------- Linear Regression (baseline) ----------------
    print("\nTraining Linear Regression...")
    X_train_ohe = one_hot(train_feat, feature_cols)
    X_holdout_ohe = align_columns(X_train_ohe, one_hot(holdout_feat, feature_cols))
    lr = LinearRegression()
    lr.fit(X_train_ohe, y_train)
    pred = np.expm1(lr.predict(X_holdout_ohe))
    results.append(evaluate(y_holdout_true, pred, "LinearRegression"))
    fitted_models["LinearRegression"] = (lr, "ohe")

    # ---------------- Ridge Regression ----------------
    print("Training Ridge Regression...")
    ridge = Ridge(alpha=5.0, random_state=RANDOM_STATE)
    ridge.fit(X_train_ohe, y_train)
    pred = np.expm1(ridge.predict(X_holdout_ohe))
    results.append(evaluate(y_holdout_true, pred, "Ridge"))
    fitted_models["Ridge"] = (ridge, "ohe")

    # ---------------- Random Forest ----------------
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=14, min_samples_leaf=3,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf.fit(X_train_ohe, y_train)
    pred = np.expm1(rf.predict(X_holdout_ohe))
    results.append(evaluate(y_holdout_true, pred, "RandomForest"))
    fitted_models["RandomForest"] = (rf, "ohe")

    # Carve an early-stopping validation slice out of the *training* window
    # only (never touch holdout_df here) so the boosted models pick sane
    # tree counts/depths instead of overfitting on default iteration counts.
    es_cutoff = train_df["date"].quantile(0.88)
    fit_df = train_feat[train_feat["date"] <= es_cutoff]
    es_df = train_feat[train_feat["date"] > es_cutoff]
    X_fit, y_fit = fit_df[feature_cols], np.log1p(fit_df["posted_rate"].values)
    X_es, y_es = es_df[feature_cols], np.log1p(es_df["posted_rate"].values)

    # ---------------- XGBoost ----------------
    try:
        import xgboost as xgb
        print("Training XGBoost (with early stopping)...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=2000, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, reg_lambda=5.0, reg_alpha=0.5,
            random_state=RANDOM_STATE, tree_method="hist",
            enable_categorical=True, n_jobs=-1,
            early_stopping_rounds=50, eval_metric="rmse",
        )
        xgb_model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        print(f"  best_iteration={xgb_model.best_iteration}")
        pred = np.expm1(xgb_model.predict(X_holdout))
        results.append(evaluate(y_holdout_true, pred, "XGBoost"))
        fitted_models["XGBoost"] = (xgb_model, "native")
    except Exception as e:
        print(f"XGBoost failed: {e}")

    # ---------------- LightGBM ----------------
    try:
        import lightgbm as lgb
        print("Training LightGBM (with early stopping)...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=2000, max_depth=5, num_leaves=20,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=40, reg_lambda=5.0, reg_alpha=0.5,
            random_state=RANDOM_STATE, verbosity=-1,
        )
        lgb_model.fit(
            X_fit, y_fit, categorical_feature=cat_cols,
            eval_set=[(X_es, y_es)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        print(f"  best_iteration={lgb_model.best_iteration_}")
        pred = np.expm1(lgb_model.predict(X_holdout))
        results.append(evaluate(y_holdout_true, pred, "LightGBM"))
        fitted_models["LightGBM"] = (lgb_model, "native")
    except Exception as e:
        print(f"LightGBM failed: {e}")

    # ---------------- CatBoost ----------------
    try:
        from catboost import CatBoostRegressor, Pool
        print("Training CatBoost (with early stopping)...")
        cat_idx = [X_fit.columns.get_loc(c) for c in cat_cols]
        fit_pool = Pool(X_fit, y_fit, cat_features=cat_idx)
        es_pool = Pool(X_es, y_es, cat_features=cat_idx)
        holdout_pool = Pool(X_holdout, cat_features=cat_idx)
        cb_model = CatBoostRegressor(
            iterations=2000, depth=6, learning_rate=0.03,
            l2_leaf_reg=6.0, loss_function="RMSE",
            random_state=RANDOM_STATE, verbose=False,
            early_stopping_rounds=50,
        )
        cb_model.fit(fit_pool, eval_set=es_pool, use_best_model=True)
        print(f"  best_iteration={cb_model.get_best_iteration()}")
        pred = np.expm1(cb_model.predict(holdout_pool))
        results.append(evaluate(y_holdout_true, pred, "CatBoost"))
        fitted_models["CatBoost"] = (cb_model, "catboost")
    except Exception as e:
        print(f"CatBoost failed: {e}")

    # ---------------- Compare & select ----------------
    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\n=== Holdout performance (sorted by RMSE) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(REPORTS / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    print(f"\nBest model: {best_name}")

    # ---------------- Plots ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(results_df["model"], results_df["RMSE"], color="#064A56")
    ax.set_xlabel("Holdout RMSE ($, lower is better)")
    ax.set_title("Model comparison on time-based holdout")
    ax.invert_yaxis()
    for i, v in enumerate(results_df["RMSE"]):
        ax.text(v, i, f"  ${v:,.0f}", va="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)
    plt.close(fig)

    best_model_obj, best_kind = fitted_models[best_name]
    if best_kind == "ohe":
        best_pred = np.expm1(best_model_obj.predict(X_holdout_ohe))
    elif best_kind == "catboost":
        best_pred = np.expm1(best_model_obj.predict(holdout_pool))
    else:
        best_pred = np.expm1(best_model_obj.predict(X_holdout))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_holdout_true, best_pred, s=6, alpha=0.25, color="#064A56")
    lims = [0, max(y_holdout_true.max(), best_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual posted_rate ($)")
    ax.set_ylabel("Predicted rate ($)")
    ax.set_title(f"{best_name}: actual vs predicted (holdout)")
    fig.tight_layout()
    fig.savefig(FIGURES / "holdout_actual_vs_pred.png", dpi=150)
    plt.close(fig)

    # ---------------- Refit best model on 100% of train_test.csv ----------------
    print(f"\nRefitting {best_name} on the full dataset for production use...")
    full_pre = Preprocessor().fit(df)
    full_feat = engineer_features(df, full_pre)
    feature_cols_full = get_feature_columns(full_feat)
    cat_cols_full = [c for c in feature_cols_full if str(full_feat[c].dtype) == "category"]
    y_full = np.log1p(full_feat["posted_rate"].values)
    X_full = full_feat[feature_cols_full]

    if best_kind == "native" and best_name == "XGBoost":
        import xgboost as xgb
        params = xgb_model.get_params()
        params["early_stopping_rounds"] = None
        params["n_estimators"] = int(xgb_model.best_iteration + 1)
        final_model = xgb.XGBRegressor(**params)
        final_model.fit(X_full, y_full)
    elif best_name == "LightGBM":
        import lightgbm as lgb
        params = lgb_model.get_params()
        params["n_estimators"] = int(lgb_model.best_iteration_)
        final_model = lgb.LGBMRegressor(**params)
        final_model.fit(X_full, y_full, categorical_feature=cat_cols_full)
    elif best_name == "CatBoost":
        from catboost import CatBoostRegressor, Pool
        cat_idx_full = [X_full.columns.get_loc(c) for c in cat_cols_full]
        full_pool = Pool(X_full, y_full, cat_features=cat_idx_full)
        params = cb_model.get_params()
        params["iterations"] = int(cb_model.get_best_iteration() or params["iterations"])
        params.pop("early_stopping_rounds", None)
        final_model = CatBoostRegressor(**params)
        final_model.fit(full_pool)
    elif best_name == "RandomForest":
        final_model = RandomForestRegressor(**rf.get_params())
        X_full_ohe = one_hot(full_feat, feature_cols_full)
        final_model.fit(X_full_ohe, y_full)
        joblib.dump(list(X_full_ohe.columns), MODELS / "ohe_columns.joblib")
    else:  # Linear / Ridge
        final_model = type(fitted_models[best_name][0])(**fitted_models[best_name][0].get_params())
        X_full_ohe = one_hot(full_feat, feature_cols_full)
        final_model.fit(X_full_ohe, y_full)
        joblib.dump(list(X_full_ohe.columns), MODELS / "ohe_columns.joblib")

    # medians for any optional numeric feature (only used by the one-hot /
    # linear path, since tree models handle NaN natively) so that frames
    # missing a whole column (e.g. december_chart_inputs.csv has no lat/lon)
    # can still be scored without dropping rows
    numeric_medians = {
        c: float(X_full[c].median())
        for c in feature_cols_full
        if c not in cat_cols_full
    }
    joblib.dump(numeric_medians, MODELS / "numeric_medians.joblib")

    joblib.dump(final_model, MODELS / "best_model.joblib")
    joblib.dump(full_pre, MODELS / "preprocessor.joblib")
    with open(MODELS / "feature_columns.json", "w") as f:
        json.dump({"feature_cols": feature_cols_full, "cat_cols": cat_cols_full}, f, indent=2)
    with open(MODELS / "model_kind.json", "w") as f:
        json.dump({"model_name": best_name, "kind": best_kind}, f, indent=2)

    print("\nSaved: models/best_model.joblib, models/preprocessor.joblib, "
          "models/feature_columns.json, models/model_kind.json")
    print("Done.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
