#!/usr/bin/env python3
"""
predict.py
----------
Load a trained surrogate model and predict the outputs for given inputs.

Usage:
  python predict.py <X_Cell_Size> <YZ_Cell_Size> <Velocity_Inlet>

Example:
  python predict.py 15 15 3000
"""

import sys
import numpy as np
import joblib

EXPECTED_FEATURES = ["X Cell Size", "YZ Cell Size", "Velocity Inlet"]
EXPECTED_TARGETS  = ["PressureDrop", "AvgVelocity", "Surface Area", "Mass"]

def main():
    if len(sys.argv) != 4:
        print("Usage: python predict.py <X_Cell_Size> <YZ_Cell_Size> <Velocity_Inlet>")
        sys.exit(1)

    try:
        x_size = float(sys.argv[1])
        yz_size = float(sys.argv[2])
        velocity = float(sys.argv[3])
    except ValueError:
        print("All inputs must be numeric.")
        sys.exit(1)

    # Load the trained model
    model_path = "surrogate_model.pkl"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    # Prepare input and predict
    inputs = np.array([[x_size, yz_size, velocity]])
    predictions = model.predict(inputs)[0]

    # Print results
    print("\n=== Surrogate Model Prediction ===")
    for target, value in zip(EXPECTED_TARGETS, predictions):
        print(f"{target:14s}: {value:.4f}")

if __name__ == "__main__":
    main()
