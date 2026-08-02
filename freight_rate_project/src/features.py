"""
Shared preprocessing / feature-engineering logic for the freight rate model.

This module is imported by both train.py and predict.py so that the exact
same transformations are applied at training time and at inference time.

Design notes (see reports/report.md for full rationale):
- validation.csv has lat/lon + market_index + quote_signal; december_chart_inputs.csv
  does NOT. So every feature that is not guaranteed to exist in ALL three
  datasets (train, validation, december) is treated as OPTIONAL: it is used
  when present and safely skipped (with a missing-indicator flag) when absent.
- Categorical columns (pickup, delivery, equipment) are kept as pandas
  'category' dtype so tree-based models (LightGBM / CatBoost / XGBoost) can
  split on them natively, without exploding dimensionality via one-hot
  encoding on ~64 x 64 city combinations.
- A continuous "days_since_anchor" trend feature is anchored to a fixed
  calendar date (2025-01-01) so that it extrapolates naturally into
  November/December, which lie outside the training window (Jan-Oct 2025).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANCHOR_DATE = pd.Timestamp("2025-01-01")

CATEGORICAL_COLS = ["pickup", "delivery", "equipment"]

# Columns that are present in train_test.csv and validation.csv but NOT in
# december_chart_inputs.csv. Used opportunistically, never required.
OPTIONAL_NUMERIC_COLS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
                          "market_index", "quote_signal"]

BASE_NUMERIC_COLS = ["distance", "weight"]


def _haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8  # earth radius, miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["days_since_anchor"] = (df["date"] - ANCHOR_DATE).dt.days.astype(float)
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    doy = df["date"].dt.dayofyear.astype(float)
    # cyclical encodings capture annual / weekly seasonality without
    # creating an artificial jump between Dec 31 and Jan 1
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    dow = df["day_of_week"].astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_distance"] = np.log1p(df["distance"])
    if {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"}.issubset(df.columns):
        df["haversine_distance"] = _haversine_miles(
            df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
        )
        # circuity ratio: how much longer the road route is than the straight
        # line. Flags obviously bad coordinate/distance combinations too.
        df["circuity_ratio"] = df["distance"] / df["haversine_distance"].replace(0, np.nan)
    return df


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["weight", "market_index", "quote_signal"]:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


class Preprocessor:
    """Fit imputation statistics on the training data, then apply them
    consistently to validation / december inference data (which have fewer
    columns and different missingness patterns)."""

    def __init__(self):
        self.weight_median_by_equipment_: dict | None = None
        self.weight_global_median_: float | None = None
        self.market_index_median_: float | None = None
        self.quote_signal_median_: float | None = None

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.weight_median_by_equipment_ = df.groupby("equipment")["weight"].median().to_dict()
        self.weight_global_median_ = df["weight"].median()
        if "market_index" in df.columns:
            self.market_index_median_ = df["market_index"].median()
        if "quote_signal" in df.columns:
            self.quote_signal_median_ = df["quote_signal"].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # weight: impute by equipment-specific median (a Reefer/Flatbed load
        # is systematically heavier/lighter than a Dry Van load)
        if "weight" in df.columns:
            fallback = df["equipment"].map(self.weight_median_by_equipment_)
            df["weight"] = df["weight"].fillna(fallback).fillna(self.weight_global_median_)

        # market_index / quote_signal: weak linear correlation with the
        # target (|r| < 0.04 in EDA) so a simple median impute is safe and
        # doesn't materially bias the model; also covers the case where the
        # column is entirely absent (december_chart_inputs.csv)
        if "market_index" in df.columns:
            df["market_index"] = df["market_index"].fillna(self.market_index_median_)
        else:
            df["market_index"] = self.market_index_median_
            df["market_index_missing"] = 1

        if "quote_signal" in df.columns:
            df["quote_signal"] = df["quote_signal"].fillna(self.quote_signal_median_)
        else:
            df["quote_signal"] = self.quote_signal_median_
            df["quote_signal_missing"] = 1

        return df


def engineer_features(df: pd.DataFrame, preprocessor: Preprocessor) -> pd.DataFrame:
    """Full pipeline: time features -> distance features -> missing flags ->
    imputation. Safe to call on train, validation, or december frames."""
    df = add_time_features(df)
    df = add_distance_features(df)
    df = add_missing_indicators(df)
    df = preprocessor.transform(df)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns actually available in this frame (handles the fact
    that december_chart_inputs.csv lacks lat/lon)."""
    candidate_cols = (
        CATEGORICAL_COLS
        + BASE_NUMERIC_COLS
        + ["log_distance", "days_since_anchor", "month", "day_of_week",
           "is_weekend", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
           "market_index", "quote_signal",
           "weight_missing", "market_index_missing", "quote_signal_missing"]
        + ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
           "haversine_distance", "circuity_ratio"]
    )
    return [c for c in candidate_cols if c in df.columns]
