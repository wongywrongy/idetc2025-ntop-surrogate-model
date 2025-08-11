#!/usr/bin/env python3
"""
train_model.py
---------------
Train a multi-output surrogate model for the nTop ASME Hackathon dataset and save it for later use.

Usage (example):
  python train_model.py \
    --csv "nTop ASME Hackathon Data.csv" \
    --model-out "surrogate_model.pkl" \
    --metrics-out "metrics.json" \
    --featimp-out "feature_importances.csv"

Inputs (features):
  - X Cell Size
  - YZ Cell Size
  - Velocity Inlet

Targets (labels):
  - PressureDrop
  - AvgVelocity
  - Surface Area
  - Mass
"""

import argparse
import json
import time
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


EXPECTED_FEATURES = ["X Cell Size", "YZ Cell Size", "Velocity Inlet"]
EXPECTED_TARGETS  = ["PressureDrop", "AvgVelocity", "Surface Area", "Mass"]


def check_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {label} columns in CSV: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def aggregate_feature_importances(model: MultiOutputRegressor, feature_names: List[str]) -> pd.DataFrame:
    """
    Aggregate feature importances across targets for a MultiOutput RandomForest.
    Returns a DataFrame with per-target and mean importances.
    """
    per_target = {}
    for target_name, est in zip(EXPECTED_TARGETS, model.estimators_):
        if hasattr(est, "feature_importances_"):
            per_target[target_name] = est.feature_importances_
        else:
            per_target[target_name] = np.zeros(len(feature_names))

    table = pd.DataFrame(per_target, index=feature_names)
    table["MeanImportance"] = table.mean(axis=1)
    table = table.sort_values("MeanImportance", ascending=False)
    return table


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute RMSE, MAE, and R^2 per target and aggregate RMSE.
    """
    metrics = {}
    for i, target in enumerate(EXPECTED_TARGETS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        metrics[target] = {
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae":  float(mean_absolute_error(yt, yp)),
            "r2":   float(r2_score(yt, yp)),
        }

    agg_rmse = float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())))
    metrics["_aggregate"] = {"rmse": agg_rmse}
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a surrogate model for nTop Hackathon data.")
    parser.add_argument("--csv", required=True, help="Path to input CSV (e.g., 'nTop ASME Hackathon Data.csv')")
    parser.add_argument("--model-out", default="surrogate_model.pkl", help="Path to save trained model (.pkl)")
    parser.add_argument("--metrics-out", default="metrics.json", help="Path to save evaluation metrics (.json)")
    parser.add_argument("--featimp-out", default="feature_importances.csv", help="Path to save feature importances (.csv)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--n-estimators", type=int, default=600, help="Number of trees in RandomForest (default 600)")
    args = parser.parse_args()

    # 1) Load data
    df = pd.read_csv(args.csv)
    print(f"Loaded CSV with shape {df.shape}")
    check_columns(df, EXPECTED_FEATURES, "feature")
    check_columns(df, EXPECTED_TARGETS, "target")

    X = df[EXPECTED_FEATURES].values
    y = df[EXPECTED_TARGETS].values

    # 2) Train/val split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # 3) Model
    base = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1
    )
    model = MultiOutputRegressor(base)

    # 4) Train
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0
    print(f"Training completed in {train_time_s:.3f} s")

    # 5) Evaluate
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    print("\n=== Evaluation (Test Set) ===")
    for tgt in EXPECTED_TARGETS:
        m = metrics[tgt]
        print(f"{tgt:14s} | RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f} | R^2: {m['r2']:.4f}")
    print(f"Aggregate RMSE (all targets): {metrics['_aggregate']['rmse']:.4f}")

    # 6) Inference speed (single prediction)
    sample = np.array([[15.0, 15.0, 3000.0]])
    _ = model.predict(sample)  # warm-up
    t0 = time.perf_counter()
    _ = model.predict(sample)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Inference time (1 sample): {infer_ms:.3f} ms")

    # 7) Save artifacts
    joblib.dump(model, args.model_out)
    print(f"Saved model to: {args.model_out}")

    with open(args.metrics_out, "w") as f:
        json.dump({
            "train_time_s": train_time_s,
            "inference_ms_one": infer_ms,
            **metrics
        }, f, indent=2)
    print(f"Saved metrics to: {args.metrics_out}")

    # Feature importances (RF only)
    try:
        featimp = aggregate_feature_importances(model, EXPECTED_FEATURES)
        featimp.to_csv(args.featimp_out)
        print(f"Saved feature importances to: {args.featimp_out}")
    except Exception as e:
        print(f"Could not compute feature importances: {e}")


if __name__ == "__main__":
    main()
