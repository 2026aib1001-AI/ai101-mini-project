"""
Used Car Price Prediction Pipeline
AI101 Mini-Project — IIT Ropar

This script runs the end-to-end data loading, feature engineering,
feature selection, model training, evaluation, and plot generation.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Project directory paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "CAR DETAILS FROM CAR DEKHO.csv"
OUTPUT_DIR = ROOT_DIR / "output"


def load_and_clean_data(data_path: Path) -> pd.DataFrame:
    """Load the dataset and remove duplicate rows."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print("=" * 60)
    print("   USED CAR PRICE PREDICTION - PIPELINE EXECUTION")
    print("=" * 60)
    print("\n[1/5] Loading and cleaning dataset...")

    df = pd.read_csv(data_path)
    initial_rows = len(df)
    df = df.drop_duplicates().copy()
    print(f"      Initial records: {initial_rows}")
    print(f"      Duplicates removed: {initial_rows - len(df)}")
    print(f"      Clean unique records: {len(df)}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive car age and extract car brand."""
    print("\n[2/5] Feature Engineering...")
    df = df.copy()
    # 1. Car Age relative to reference year (2020)
    df["age"] = 2020 - df["year"]
    # 2. Extract brand from vehicle name
    df["brand"] = df["name"].apply(lambda x: str(x).split()[0])

    print(f"      Derived 'age' (2020 - year, range: {df['age'].min()} to {df['age'].max()} years)")
    print(f"      Extracted 'brand' ({df['brand'].nunique()} unique brands detected)")
    return df


def select_features(df: pd.DataFrame):
    """
    Select features and isolate predictor matrix X and target y.
    Drops high-cardinality 'name' and redundant 'year'.
    """
    print("\n[3/5] Feature Selection...")
    target_col = "selling_price"
    numeric_features = ["age", "km_driven"]
    categorical_features = ["fuel", "seller_type", "transmission", "owner", "brand"]
    selected_features = numeric_features + categorical_features

    X = df[selected_features].copy()
    y = df[target_col].copy()

    print(f"      Target variable: '{target_col}'")
    print(f"      Total selected input features: {len(selected_features)}")
    print(f"        - Numeric ({len(numeric_features)}): {numeric_features}")
    print(f"        - Categorical ({len(categorical_features)}): {categorical_features}")
    print(f"      Feature matrix shape (X): {X.shape}")
    print(f"      Target vector shape (y): {y.shape}")

    return X, y, numeric_features, categorical_features


def main():
    # 1. Load and clean data
    df = load_and_clean_data(DATA_PATH)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Feature selection
    X, y, numeric_features, categorical_features = select_features(df)

    print("\n[Status] Data loading, cleaning, and feature selection completed.")
    print("         Next step: Model training & evaluation pipeline integration.")


if __name__ == "__main__":
    main()
