# ExplainStock — Explainable Inventory Decision System

> Inspired by Amazon SCOT's stated problem of explaining automated inventory decisions to third-party sellers.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)](https://xgboost.ai)
[![SHAP](https://img.shields.io/badge/SHAP-0.51-green)](https://shap.readthedocs.io)
[![Dataset](https://img.shields.io/badge/Dataset-M5%20Forecasting-red)](https://kaggle.com/competitions/m5-forecasting-accuracy)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Problem Statement

Amazon's automated inventory system (SCOT) makes thousands of daily reorder decisions for third-party sellers — but provides no explanation for why a specific quantity was recommended. Sellers cannot distinguish between demand-driven and algorithm-driven decisions, leading to mistrust and suboptimal inventory management.

ExplainStock addresses this gap by combining demand forecasting with three explainability layers that generate human-readable inventory reports.

---

## Key Research Findings

| Finding | Metric | Result |
|---|---|---|
| Forecast Accuracy | Test RMSE | 2.49 units |
| Forecast Accuracy | Test MAE | 1.35 units |
| Generalization | Overfit Ratio (Test/Train) | 0.89 — strong |
| Explainability | Top demand driver | roll_mean_7 (SHAP = 0.688) |
| Explainability | SNAP day demand lift | +25.5% in FOODS category |
| Explainability | Price elasticity | −0.445 (moderate inelasticity) |
| Faithfulness | SHAP vs Permutation Importance | 8/10 top features match |
| Feature Selection | 10 vs 26 features | 97.9% performance retained |

---

## System Architecture

```
M5 Dataset (2.7M rows, Walmart US)
        ↓
Feature Engineering — 27 features
  • Lag features      : lag_7, lag_14, lag_28
  • Rolling means     : 7 / 14 / 28 day windows
  • Calendar signals  : SNAP days, events, day-of-week
  • Price signals     : elasticity, change, relative pricing
        ↓
XGBoost Demand Forecasting
  • Test RMSE: 2.49  |  MAE: 1.35  |  Overfit ratio: 0.89
  • CA_1 store, FOODS category, 1,437 products, 5 years history
        ↓
SHAP Explainability Layer
  • Global feature importance (bar + beeswarm)
  • Per-decision waterfall explanations
  • SNAP day deep dive — +25.5% demand lift quantified
  • Faithfulness score vs permutation importance (Top-10: 0.80)
        ↓
Counterfactual Scenario Engine
  • 6 what-if scenarios per product
  • "If demand trend +50% → reorder increases 48%"
  • "If price drops 20% → demand increases 2.6%"
        ↓
LLM Narrative Layer (LLaMA 3.3 via Groq API)
  • Plain-English seller reports
  • Automated HTML dashboard
  • Actionable reorder recommendations with safety buffer
```

---

## Sample Output

```
FOODS_3_288 | Store: CA_1 | Date: 2016-04-24 | Price: $1.50

Reorder Recommendation: 25 units

Summary
We recommend reordering 25 units of product FOODS_3_288.
Average daily sales: 14.29 units (last 7 days), 16.79 units (last 28 days).

Why This Recommendation
Top drivers: 7-day rolling mean (SHAP = 4.58), 28-day rolling mean (SHAP = 3.02).
Today is not a SNAP day — no +25.5% demand lift expected.

What Could Change It
- Demand trend +50% → reorder jumps to 30 units
- Demand trend −40% → reorder drops to 6 units
- Price −20% → demand increases ~2.6%

Your Action Items
1. Reorder 25 units to meet predicted demand.
2. Monitor sales — 7-day avg (14.29) is below 28-day avg (16.79).
3. Prepare extra stock for upcoming SNAP days (+25.5% lift expected).
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Forecasting | XGBoost 3.2 |
| Explainability | SHAP 0.51 |
| Counterfactuals | Custom DiCE-style engine |
| LLM Narrative | LLaMA 3.3 70B (Groq API — free) |
| Dataset | M5 Forecasting Competition (Walmart US) |
| Visualization | Matplotlib, Seaborn |
| File Format | Parquet (PyArrow) |

---

## Project Structure

```
explainstock/
├── notebooks/
│   ├── 01_eda_m5.ipynb                  ← EDA on 2.7M rows
│   ├── 02_feature_engineering.ipynb     ← 27 features built
│   ├── 03_model_training.ipynb          ← XGBoost training
│   ├── 04_shap_explainability.ipynb     ← SHAP + faithfulness
│   ├── 05_dice_counterfactuals.ipynb    ← What-if scenarios
│   └── 06_llm_narrative.ipynb          ← LLM report generation
├── models/
│   └── xgb_ca1_foods_v1.json           ← Trained model
├── outputs/
│   ├── shap_plots/                      ← 12 research plots
│   └── reports/
│       └── explainstock_report.html    ← Demo dashboard
├── data/
│   ├── processed/                       ← Intermediate datasets
│   └── features/                        ← Engineered feature tables
└── README.md
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/ExplainStock.git
cd ExplainStock
python -m venv env
env\Scripts\activate        # Windows
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm shap jupyter kaggle groq pyarrow openpyxl
```

**Download M5 Dataset:**
```bash
kaggle competitions download -c m5-forecasting-accuracy
kaggle competitions download -c m5-forecasting-uncertainty
```
Place downloaded files in `data/raw/`.

---

## Dataset

Uses the [M5 Forecasting Competition](https://kaggle.com/competitions/m5-forecasting-accuracy) dataset — 42,840 hierarchical time series from Walmart US stores. Analysis focuses on CA_1 store, FOODS category (1,437 SKUs, 5+ years daily sales history). Data not included due to size — download via Kaggle API above.

---

## Roadmap

- [x] Month 1 — EDA → Feature Engineering → XGBoost → SHAP → Counterfactuals → LLM Reports
- [ ] Month 2 — LightGBM comparison model
- [ ] Month 2 — Streamlit interactive dashboard
- [ ] Month 2 — India festival calendar extension (Diwali, Navratri, payday signals)
- [ ] Month 3 — Formal research paper

---

## Motivation

Built as placement preparation targeting Amazon's Supply Chain Optimization Technologies (SCOT) team, which has publicly described the challenge of making automated inventory decisions interpretable to third-party sellers. The SNAP signal used in this project transfers directly to Amazon Fresh — Amazon became an authorized SNAP retailer in 2017.

---

*Built with Python · XGBoost · SHAP · LLaMA 3.3*
