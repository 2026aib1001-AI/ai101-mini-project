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
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

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


def split_dataset(X: pd.DataFrame, y: pd.Series):
    """Split features and target into 80% train and 20% test sets."""
    print("\n[4/5] Partitioning Data (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"      Training set: {X_train.shape[0]} samples")
    print(f"      Testing set : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def build_and_train_model(X_train: pd.DataFrame, y_train: pd.Series, numeric_features: list, categorical_features: list):
    """
    Build scikit-learn ColumnTransformer and Pipeline, then train Linear Regression.
    """
    print("\n[5/5] Building Preprocessing Pipeline & Training Model...")

    # 1. Numerical pipeline: Median Imputer + Standard Scaling
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 2. Categorical pipeline: Mode Imputer + One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 3. Combine into ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # 4. End-to-end model pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # 5. Fit model
    model.fit(X_train, y_train)

    # 6. Inspect transformed dimensionality
    encoded_feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    print(f"      Model training completed successfully!")
    print(f"      Total encoded features after One-Hot Encoding: {len(encoded_feature_names)}")

    # 7. Model Intercept and learned coefficients
    intercept = model.named_steps["regressor"].intercept_
    coefficients = model.named_steps["regressor"].coef_
    coef_df = pd.DataFrame({
        "Feature": encoded_feature_names,
        "Coefficient": coefficients
    }).sort_values(by="Coefficient", ascending=False).reset_index(drop=True)

    print(f"      Base Intercept: ₹{intercept:,.2f}")
    print(f"      Top Positive Driver: {coef_df.iloc[0]['Feature']} (+₹{coef_df.iloc[0]['Coefficient']:,.2f})")
    print(f"      Top Negative Driver: {coef_df.iloc[-1]['Feature']} (₹{coef_df.iloc[-1]['Coefficient']:,.2f})")

    return model, encoded_feature_names


def main():
    # 1. Load and clean data
    df = load_and_clean_data(DATA_PATH)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Feature selection
    X, y, numeric_features, categorical_features = select_features(df)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # 5. Build pipeline and train Linear Regression model
    model, encoded_feature_names = build_and_train_model(
        X_train, y_train, numeric_features, categorical_features
    )

    print("\n[Status] Data preprocessing, feature scaling/encoding, and model training complete.")
    print("         Final step remaining: Test set evaluation and saving plot outputs.")


if __name__ == "__main__":
    main()
