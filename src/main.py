"""
Used Car Price Prediction Pipeline
AI101 Mini-Project — IIT Ropar

This script runs the end-to-end data loading, feature engineering,
feature selection, model training, evaluation, and plot generation.
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless backend for command-line / server execution
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project directory paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "CAR DETAILS FROM CAR DEKHO.csv"
OUTPUT_DIR = ROOT_DIR / "output"


def load_and_clean_data(data_path: Path) -> pd.DataFrame:
    """Load the dataset and remove duplicate rows."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print("=" * 65)
    print("      USED CAR PRICE PREDICTION - PIPELINE EXECUTION")
    print("=" * 65)
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

    return model, encoded_feature_names, coef_df


def evaluate_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Generate predictions and calculate regression metrics for both Training and Testing sets.
    """
    print("\n" + "=" * 65)
    print("             MODEL EVALUATION METRICS (TRAIN VS. TEST)             ")
    print("=" * 65)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training Metrics
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)

    # Testing Metrics
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)

    # Formatted display table
    print(f"{'Metric':<28} | {'Training Set':<16} | {'Testing Set':<16}")
    print("-" * 65)
    print(f"{'R² Score (Variance Exp.)':<28} | {r2_train:<16.4f} | {r2_test:<16.4f}")
    print(f"{'Mean Absolute Error (MAE)':<28} | ₹{mae_train:<15,.2f} | ₹{mae_test:<15,.2f}")
    print(f"{'Mean Squared Error (MSE)':<28} | {mse_train:<16,.2f} | {mse_test:<16,.2f}")
    print(f"{'Root Mean Squared Error':<28} | ₹{rmse_train:<15,.2f} | ₹{rmse_test:<15,.2f}")
    print("=" * 65)

    metrics = {
        "train": {"r2": r2_train, "mae": mae_train, "mse": mse_train, "rmse": rmse_train},
        "test": {"r2": r2_test, "mae": mae_test, "mse": mse_test, "rmse": rmse_test}
    }

    return y_test_pred, metrics


def save_plots(y_test: pd.Series, y_pred: np.ndarray, coef_df: pd.DataFrame, output_dir: Path):
    """
    Generate and save diagnostic evaluation plots in the output directory.
    Decomposes coefficients into Brand, Numeric, and Categorical plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving visualization plots to: {output_dir}")

    # 1. Actual vs. Predicted Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test / 100000, y_pred / 100000, alpha=0.6, color="royalblue", edgecolor="black")
    low = min(float(np.min(y_test)), float(np.min(y_pred))) / 100000
    high = max(float(np.max(y_test)), float(np.max(y_pred))) / 100000
    plt.plot([low, high], [low, high], color="red", linestyle="--", linewidth=2, label="Perfect Prediction (y = x)")
    plt.title("Actual vs. Predicted Selling Price (Test Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Selling Price (Lakhs ₹)", fontsize=11)
    plt.ylabel("Predicted Selling Price (Lakhs ₹)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    actual_pred_path = output_dir / "actual_vs_predicted.png"
    plt.savefig(actual_pred_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {actual_pred_path.name}")

    # 2. Residual Distribution Histogram
    residuals = (y_test.to_numpy() - y_pred) / 100000
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=35, color="coral", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Error")
    plt.title("Residual Error Distribution (Actual - Predicted)", fontsize=13, fontweight="bold")
    plt.xlabel("Prediction Error (Lakhs ₹)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    residual_path = output_dir / "residual_distribution.png"
    plt.savefig(residual_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {residual_path.name}")

    # 3. Plot A: Brand-wise Coefficients (All 27 Brands)
    brand_df = coef_df[coef_df["Feature"].str.startswith("cat__brand_")].copy()
    brand_df["Brand_Name"] = brand_df["Feature"].str.replace("cat__brand_", "")
    brand_df = brand_df.sort_values(by="Coefficient", ascending=True)

    plt.figure(figsize=(10, 8))
    colors = ["forestgreen" if c >= 0 else "crimson" for c in brand_df["Coefficient"]]
    plt.barh(brand_df["Brand_Name"], brand_df["Coefficient"] / 100000, color=colors, edgecolor="black", alpha=0.85)
    plt.axvline(0, color="black", linewidth=1.2)
    plt.title("Brand Equity Coefficients (Relative Price Impact in Lakhs ₹)", fontsize=13, fontweight="bold")
    plt.xlabel("Price Impact (Lakhs ₹)", fontsize=11)
    plt.ylabel("Car Brand", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    brand_path = output_dir / "brand_coefficients.png"
    plt.savefig(brand_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {brand_path.name}")

    # 4. Plot B: Numeric Features (Age vs. Km Driven Depreciation)
    numeric_df = coef_df[coef_df["Feature"].str.startswith("num__")].copy()
    name_map = {
        "num__age": "Vehicle Age\n(per +1 Std Dev ~ 4.2 yrs)",
        "num__km_driven": "Kilometers Driven\n(per +1 Std Dev ~ 44k km)"
    }
    numeric_df["Clean_Name"] = numeric_df["Feature"].map(name_map)
    numeric_df = numeric_df.sort_values(by="Coefficient", ascending=True)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(numeric_df["Clean_Name"], numeric_df["Coefficient"] / 100000, color=["firebrick", "indianred"], edgecolor="black", width=0.45)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Continuous Feature Coefficients: Rate of Depreciation", fontsize=13, fontweight="bold")
    plt.ylabel("Price Impact (Lakhs ₹ per Std Dev)", fontsize=11)
    plt.ylim(-2.0, 0.2)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        # Center value text cleanly inside the bar to prevent collision with axis labels
        plt.text(bar.get_x() + bar.get_width() / 2, height / 2, f"₹{height:.2f}L", ha="center", va="center", fontweight="bold", color="white", fontsize=11)

    plt.tight_layout()
    numeric_path = output_dir / "numeric_coefficients.png"
    plt.savefig(numeric_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {numeric_path.name}")

    # 5. Plot C: Categorical Specifications (Transmission, Fuel, Owner, Seller)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    specs = [
        ("Transmission", "cat__transmission_", axes[0, 0], "mediumpurple"),
        ("Fuel Type", "cat__fuel_", axes[0, 1], "darkcyan"),
        ("Owner History", "cat__owner_", axes[1, 0], "goldenrod"),
        ("Seller Type", "cat__seller_type_", axes[1, 1], "steelblue"),
    ]

    for title, prefix, ax, color in specs:
        sub_df = coef_df[coef_df["Feature"].str.startswith(prefix)].copy()
        sub_df["Label"] = sub_df["Feature"].str.replace(prefix, "")
        sub_df = sub_df.sort_values(by="Coefficient", ascending=True)
        ax.barh(sub_df["Label"], sub_df["Coefficient"] / 100000, color=color, edgecolor="black", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"{title} Coefficients", fontsize=11, fontweight="bold")
        ax.set_xlabel("Impact (Lakhs ₹)")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("Vehicle Specification Categorical Coefficients (Impact in Lakhs ₹)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cat_path = output_dir / "categorical_coefficients.png"
    plt.savefig(cat_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {cat_path.name}")

    # Clean up obsolete single feature_coefficients.png if present
    old_coef_file = output_dir / "feature_coefficients.png"
    if old_coef_file.exists():
        old_coef_file.unlink()


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
    model, encoded_feature_names, coef_df = build_and_train_model(
        X_train, y_train, numeric_features, categorical_features
    )

    # 6. Evaluate model on test set (with both train & test metrics)
    y_pred, metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 7. Save output plots
    save_plots(y_test, y_pred, coef_df, OUTPUT_DIR)

    print("\n" + "=" * 65)
    print("Pipeline execution and visualization completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
