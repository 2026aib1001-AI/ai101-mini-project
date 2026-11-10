# 🚗 Used Car Price Prediction

> **AI101: Data Science and AI Essentials with Python — Mini-Project**  
> **Indian Institute of Technology Ropar (IIT Ropar)**

---

## 📌 Project Overview
This project develops an end-to-end Machine Learning pipeline to predict used car market selling prices using the **Car Dekho Used Cars Dataset**. 

Using vehicle attributes such as age, mileage, transmission type, fuel type, owner history, and brand equity, we engineer informative predictors, prevent data leakage via Scikit-Learn `Pipeline` and `ColumnTransformer` constructs, train a multiple linear regression model, and evaluate prediction performance with standard regression metrics.

---

## 📂 Repository Structure

```text
ai101-mini-project/
├── data/
│   └── CAR DETAILS FROM CAR DEKHO.csv      # Raw Car Dekho used cars dataset
├── output/
│   ├── actual_vs_predicted.png             # Actual vs Predicted evaluation plot
│   ├── residual_distribution.png           # Residual error distribution plot
│   ├── brand_coefficients.png              # Brand equity valuation coefficients
│   ├── numeric_coefficients.png            # Age & mileage depreciation coefficients
│   └── categorical_coefficients.png        # Transmission, Fuel, Owner, Seller subplots
├── src/
│   └── main.py                             # Standalone command-line execution script
├── Used_Car_Price_Prediction.ipynb         # Interactive Jupyter Notebook walkthrough
├── requirements.txt                        # Required Python package dependencies
├── .gitignore                              # Git exclusion rules
└── README.md                               # Project documentation & execution guide
```

---

## ⚡ Quickstart & Execution Guide

### 1. Clone the Repository
```bash
git clone https://github.com/2026aib1001-AI/ai101-mini-project.git
cd ai101-mini-project
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate      # On macOS/Linux
# .venv\Scripts\activate       # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline from the Command Line
```bash
python src/main.py
```
> The script will clean the data, train the model, display all metrics in the terminal, and save/update the diagnostic charts in the `output/` folder.

### 5. (Optional) Run the Interactive Jupyter Notebook
```bash
jupyter notebook Used_Car_Price_Prediction.ipynb
```

---

## 🔬 Methodology & Pipeline Architecture

```text
Raw Data (4,340 rows)
   │
   ▼
[1] Data Cleaning ──► Deduplication (763 duplicate rows removed ➔ 3,577 unique rows)
   │
   ▼
[2] Feature Engineering:
       • age = 2020 - year
       • brand = first word of vehicle title
   │
   ▼
[3] Feature Selection:
       • Target (y) : selling_price
       • Numerics   : ['age', 'km_driven']
       • Categorical: ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
       • Dropped    : 'name' (1,491 distinct titles ➔ severe overfitting) & 'year' (redundant)
   │
   ▼
[4] Train-Test Split ──► 80% Training (2,861 rows) / 20% Testing (716 rows) [random_state=42]
   │
   ▼
[5] ColumnTransformer:
       • Numeric Pipeline: Median Imputer + StandardScaler
       • Categorical Pipeline: Mode Imputer + OneHotEncoder (43 total dimensions)
   │
   ▼
[6] Linear Regression Model ──► Minimizes residual sum of squares
   │
   ▼
[7] Evaluation & Plot Generation ──► R², MAE, MSE, RMSE, and visual diagnostic exports
```

---

## 📊 Model Evaluation & Results

The model was trained on $2,861$ instances and evaluated on $716$ unseen test cars ($80/20$ split):

| Metric | Training Set | Testing Set | Generalization Gap | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **$R^2$ Score (Variance Explained)** | **0.6608** | **0.5405** | $-0.1203$ | Explains ~54.1% of variance on completely unseen data. |
| **Mean Absolute Error (MAE)** | **₹1,62,856.05** | **₹1,79,641.77** | $+₹16,785.72$ | Average prediction error is closely matched between train and test. |
| **Mean Squared Error (MSE)** | **8.2620 × 10¹⁰** | **1.4801 × 10¹¹** | $+6.5392 \times 10^{10}$ | Heavily penalizes luxury car outliers in the test fold. |
| **Root Mean Squared Error (RMSE)** | **₹2,87,436.88** | **₹3,84,722.70** | $+₹97,285.82$ | Standard deviation of residual prediction errors. |
| **Model Base Intercept** | — | **₹1,042,435.48** | — | Baseline reference price at mean/standardized feature values. |

> **Key Observation:** The close agreement between Training MAE (₹1.62L) and Testing MAE (₹1.79L) confirms that the model avoids severe overfitting and generalizes well to unseen used cars.

---

## 📈 Visual Diagnostic Plots

All plots are automatically generated and saved by `src/main.py` in the `output/` directory:

### 1. Actual vs. Predicted Selling Price
Compares true test set prices against model predictions. Points along the dashed red diagonal line ($y = x$) represent perfect predictions:

![Actual vs Predicted](output/actual_vs_predicted.png)

### 2. Residual Error Distribution
Displays the distribution of prediction residuals ($\text{Actual} - \text{Predicted}$). Centering around zero confirms an unbiased linear estimator:

![Residual Distribution](output/residual_distribution.png)

### 3. Brand Equity Coefficients (All 27 Brands)
Demonstrates the massive impact of brand perception. Luxury marques (Land Rover, Volvo, Audi, Jaguar, BMW) command multi-lakh premiums, whereas budget-oriented brands experience relative resale discounts:

![Brand Coefficients](output/brand_coefficients.png)

### 4. Continuous Feature Depreciation (Age vs. Mileage)
Directly compares the rate of depreciation caused by vehicle aging versus odometer mileage per standard deviation:
- **Vehicle Age**: Drops price by **~₹1,50,690** per 1 standard deviation (~4.2 years).
- **Kilometers Driven**: Drops price by **~₹47,754** per 1 standard deviation (~44,000 km).

![Numeric Coefficients](output/numeric_coefficients.png)

### 5. Vehicle Specification Attributes (Transmission, Fuel, Owner, Seller)
Subplots detailing how specific mechanical and ownership features influence valuation:
- **Transmission**: Automatic cars command a substantial premium over manual models.
- **Fuel Type**: Diesel vehicles command higher market values than Petrol or CNG alternatives.
- **Ownership**: First-owner cars maintain superior resale value compared to multi-owner vehicles.

![Categorical Coefficients](output/categorical_coefficients.png)

---

## 💡 Key Business & Data Insights
1. **Age vs. Mileage**: Car age depreciates vehicle value more than three times as rapidly as odometer mileage.
2. **Transmission Premium**: Automatic gearboxes significantly elevate secondary market prices.
3. **Brand Stratification**: Brand equity is the single most dominant driver in the dataset, with premium European brands commanding up to ₹23.3L over baseline.

---

## 🛠️ Tech Stack & Requirements
- **Language**: Python 3.9+
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn` (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `LinearRegression`)
- **Data Visualization**: `matplotlib`
