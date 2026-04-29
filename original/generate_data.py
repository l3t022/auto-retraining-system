"""Generate sample data for testing the Auto-Retrain System."""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_regression_data(n_samples: int = 1000, noise: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic regression data."""
    np.random.seed(seed)
    
    # Generate features
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    X3 = np.random.randn(n_samples)
    X4 = np.random.randint(0, 5, n_samples)  # categorical
    X5 = np.random.exponential(1, n_samples)
    
    # Generate target (with some non-linear relationships)
    y = (
        2.0 * X1 
        + 0.5 * X2 
        - 1.5 * X3 
        + 0.3 * X1 * X2 
        + 0.1 * X5
        + np.random.randn(n_samples) * noise
    )
    
    df = pd.DataFrame({
        'feature1': X1,
        'feature2': X2,
        'feature3': X3,
        'feature4': X4,
        'feature5': X5,
        'target': y
    })
    
    return df


def generate_classification_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic binary classification data."""
    np.random.seed(seed)
    
    # Generate features
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    X3 = np.random.randn(n_samples)
    X4 = np.random.randint(0, 5, n_samples)
    X5 = np.random.exponential(1, n_samples)
    
    # Generate target (probabilities)
    probs = 1 / (1 + np.exp(-(
        0.5 * X1 
        + 0.3 * X2 
        - 0.4 * X3 
        + 0.1 * X5
    )))
    y = (np.random.rand(n_samples) < probs).astype(int)
    
    df = pd.DataFrame({
        'feature1': x1,
        'feature2': X2,
        'feature3': X3,
        'feature4': X4,
        'feature5': X5,
        'target': y
    })
    
    return df


def create_initial_dataset(output_dir: str = "data"):
    """Create initial dataset for the first training."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate regression dataset
    df = generate_regression_data(n_samples=1000, noise=0.1, seed=42)
    df.to_csv(f"{output_dir}/train.csv", index=False)
    print(f"Created {output_dir}/train.csv with {len(df)} samples")
    
    # Save test dataset
    df_test = generate_regression_data(n_samples=200, noise=0.1, seed=123)
    df_test.to_csv(f"{output_dir}/test.csv", index=False)
    print(f"Created {output_dir}/test.csv with {len(df_test)} samples")
    
    return df


def create_updated_dataset(output_dir: str = "data", version: str = "v2"):
    """Create updated dataset with some changes."""
    if version == "v2":
        # Slightly different data distribution
        np.random.seed(123)  # Different seed
        n_samples = 1100  # More samples
        
        X1 = np.random.randn(n_samples) * 1.2  # Different scale
        X2 = np.random.randn(n_samples)
        X3 = np.random.randn(n_samples)
        X4 = np.random.randint(0, 5, n_samples)
        X5 = np.random.exponential(1.2, n_samples)  # Different distribution
        
        # Slightly different relationship
        y = (
            2.5 * X1  # Different coefficient
            + 0.6 * X2
            - 1.8 * X3
            + 0.4 * X1 * X2
            + 0.15 * X5
            + np.random.randn(n_samples) * 0.12  # More noise
        )
        
        df = pd.DataFrame({
            'feature1': X1,
            'feature2': X2,
            'feature3': X3,
            'feature4': X4,
            'feature5': X5,
            'target': y
        })
        
        df.to_csv(f"{output_dir}/train.csv", index=False)
        print(f"Updated {output_dir}/train.csv with {len(df)} samples (version {version})")
        
    return df


if __name__ == "__main__":
    # Create data directory and sample files
    create_initial_dataset("data")
    print("\nSample data created successfully!")
    print("\nTo test the system:")
    print("1. python main.py --mode once --data data/train.csv --target target")
    print("2. Then modify data/train.csv with new data")
    print("3. Run again to trigger auto-retraining")