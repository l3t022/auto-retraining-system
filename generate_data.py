#!/usr/bin/env python3
"""
Sample Data Generator for Auto-Retrain System

Generates synthetic training data for regression or classification tasks.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification


def generate_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42, output_path="data/train.csv"):
    """Generate regression dataset."""
    print(f"Generating regression data: {n_samples} samples, {n_features} features...")
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
        n_informative=n_features // 2,
    )
    
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Saved: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return df


def generate_classification(n_samples=1000, n_features=10, noise=0.1, random_state=42, output_path="data/train.csv"):
    """Generate classification dataset."""
    print(f"Generating classification data: {n_samples} samples, {n_features} features...")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
        n_informative=n_features // 2,
        n_clusters_per_class=2,
    )
    
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Saved: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return df


def generate_drift_data(original_path, noise_factor=1.5, output_path=None):
    """Generate drifted version of existing data for testing."""
    print(f"Generating drifted data from: {original_path}")
    
    df = pd.read_csv(original_path)
    target_col = "target"
    
    # Get features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Add noise to features (simulate drift)
    noise_scale = noise_factor * X.std()
    X_drifted = X + np.random.randn(*X.shape) * noise_scale
    
    # Option: add noise to target too
    y_drifted = y + np.random.randn(len(y)) * noise_factor * y.std() * 0.1
    
    df_drifted = X_drifted.copy()
    df_drifted[target_col] = y_drifted
    
    output_path = output_path or original_path.replace(".csv", "_drifted.csv")
    df_drifted.to_csv(output_path, index=False)
    
    print(f"Saved: {output_path}")
    print(f"Shape: {df_drifted.shape}")
    print(f"Target range: [{y_drifted.min():.2f}, {y_drifted.max():.2f}]")
    
    return df_drifted


def main():
    parser = argparse.ArgumentParser(description="Generate sample data for Auto-Retrain System")
    parser.add_argument("--type", choices=["regression", "classification", "drift"], default="regression")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--features", type=int, default=10, help="Number of features")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level")
    parser.add_argument("--output", default="data/train.csv", help="Output path")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--from-file", help="Generate drift from existing file")
    parser.add_argument("--noise-factor", type=float, default=1.5, help="Drift noise multiplier")
    
    args = parser.parse_args()
    
    if args.type == "regression":
        generate_regression(
            n_samples=args.samples,
            n_features=args.features,
            noise=args.noise,
            random_state=args.random_state,
            output_path=args.output,
        )
    elif args.type == "classification":
        generate_classification(
            n_samples=args.samples,
            n_features=args.features,
            noise=args.noise,
            random_state=args.random_state,
            output_path=args.output,
        )
    elif args.type == "drift":
        generate_drift_data(
            original_path=args.from_file or args.output,
            noise_factor=args.noise_factor,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()