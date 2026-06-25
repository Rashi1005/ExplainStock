print("=" * 55)
print("ExplainStock Environment Verification")
print("=" * 55)

import sys
print(f"\n✅ Python version: {sys.version}")

# Core data libraries
import numpy as np
import pandas as pd
print(f"✅ NumPy:       {np.__version__}")
print(f"✅ Pandas:      {pd.__version__}")

# Visualization
import matplotlib
import seaborn as sns
print(f"✅ Matplotlib:  {matplotlib.__version__}")
print(f"✅ Seaborn:     {sns.__version__}")

# ML libraries
import sklearn
import xgboost as xgb
import lightgbm as lgb
print(f"✅ Scikit-learn:{sklearn.__version__}")
print(f"✅ XGBoost:     {xgb.__version__}")
print(f"✅ LightGBM:    {lgb.__version__}")

# Explainability
import shap
print(f"✅ SHAP:        {shap.__version__}")

# Functional smoke test — train a tiny XGBoost model
print("\n--- Running XGBoost smoke test ---")
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBRegressor(n_estimators=10, verbosity=0)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"✅ XGBoost trained. Sample prediction: {preds[0]:.2f}")

# SHAP smoke test
print("\n--- Running SHAP smoke test ---")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:5])
print(f"✅ SHAP values computed. Shape: {shap_values.values.shape}")

# M5 dataset check
print("\n--- Checking M5 dataset files ---")
import os
raw_path = "data/raw"
expected_files = [
    "sales_train_validation.csv",
    "sell_prices.csv",
    "calendar.csv"
]
for fname in expected_files:
    fpath = os.path.join(raw_path, fname)
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"✅ Found: {fname} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Missing: {fname} — check data/raw/ folder")

print("\n" + "=" * 55)
print("Setup complete! You're ready for Month 1.")
print("=" * 55)