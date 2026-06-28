# app.py — ExplainStock Dashboard
# Piece 1: Scaffold, imports, cached data/model loading

import streamlit as st
import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from datetime import datetime, timedelta

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="ExplainStock — Inventory Decisions",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styling pass (Piece 9) ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Tighten default Streamlit top padding */
    .block-container { padding-top: 2rem; }

    /* Section headers (st.markdown ### ...) get a consistent rule beneath them */
    h3 {
        border-bottom: 2px solid #F2F2F2;
        padding-bottom: 8px;
        margin-top: 1.8rem !important;
    }

    /* Metric labels in navy, values stay default for contrast */
    [data-testid="stMetricLabel"] {
        color: #131921;
        font-weight: 600;
    }

    /* Sidebar background slightly distinct from main content */
    [data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid #E8E8E8;
    }

    /* Primary buttons (download, generate) use Amazon orange consistently */
    .stDownloadButton button, .stButton button {
        border-color: #FF9900;
        color: #131921;
    }
    .stDownloadButton button:hover, .stButton button:hover {
        background-color: #FF9900;
        color: #131921;
        border-color: #FF9900;
    }

    /* Caption text slightly tighter line-height for dense info areas */
    .stCaption, [data-testid="stCaptionContainer"] {
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the LightGBM model (primary, per Step 13 decision: faster + more faithful)."""
    model = lgb.Booster(model_file='models/lgb_ca1_foods_v1.txt')
    return model

@st.cache_data
def load_test_predictions():
    """Load the real test-set predictions (40,236 rows, CA_1 FOODS, held-out 28 days)."""
    df = pd.read_parquet('data/features/test_predictions.parquet')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_results_store():
    """Load validated metrics from the rigor pass (Steps 1-13)."""
    with open('data/results_store.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_seller_reports():
    """Load pre-generated LLM narrative reports."""
    with open('outputs/reports/seller_reports.json', 'r') as f:
        return json.load(f)

@st.cache_resource
def get_groq_client():
    from groq import Groq
    from dotenv import load_dotenv
    import os
    load_dotenv()
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

groq_client = get_groq_client()

# ── Session state initialization ──────────────────
if 'live_report_text' not in st.session_state:
    st.session_state['live_report_text'] = None

def generate_seller_report(report_data, client):
    """Exact copy of the function from 06_llm_narrative.ipynb Cell 4."""
    system_prompt = """You are ExplainStock, an AI system built to explain 
automated inventory decisions to Amazon third-party sellers in plain English.

Your job is to take structured model output — predictions, SHAP feature 
importances, and counterfactual scenarios — and write a clear, helpful 
report that a seller with no data science background can understand and 
act on.

Rules:
- Write for a non-technical seller, not a data scientist
- Always explain WHY the system made its recommendation
- Always include 2-3 specific actions the seller can take
- Be concise — maximum 300 words
- Use simple language, no jargon
- Format with clear sections: Summary, Why This Recommendation, 
  What Could Change It, Your Action Items
- Use specific numbers from the data, never be vague"""

    user_prompt = f"""Generate an inventory decision report for this seller.

Here is the model output data:
{json.dumps(report_data, indent=2)}

Additional context:
- roll_mean_7 = average daily sales over last 7 days
- roll_mean_28 = average daily sales over last 28 days
- SHAP values measure how much each factor pushed the forecast up or down
- Counterfactuals show what the recommendation would be under different conditions
- SNAP days are government benefit days that increase FOODS demand by ~25%

Write the seller report now."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content

# ── Load everything once, cached ─────────────────────────────────────────────
model = load_model()
test_df = load_test_predictions()
results_store = load_results_store()
seller_reports = load_seller_reports()

# Get the EXACT feature list the model was trained on, directly from the booster
# (more robust than guessing an exclusion list by hand)
FEATURE_COLS = model.feature_name()

# Remove the two debug st.write() lines from Piece 1 — replace with the sidebar below

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 ExplainStock")
    st.caption("Explainable inventory decisions")
    st.divider()

    # Store selector — CA_1 only (locked decision: no trained model for CA_2/TX_1)
    st.selectbox(
        "Store",
        options=["CA_1 — California Store 1"],
        disabled=False,
        help="Currently trained on CA_1 (FOODS category) only. "
             "Additional stores are a planned extension — see roadmap.",
    )

    # Date picker — constrained to the real held-out test period
    min_date = test_df['date'].min().date()
    max_date = test_df['date'].max().date()

    selected_date = st.date_input(
        "Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        help=f"Held-out test period: {min_date} to {max_date} (28 days).",
    )

    st.divider()

    # Product search
    # Product search — show clean display names, map back to real IDs for lookups
    all_product_ids = sorted(test_df['id'].unique())
    display_to_real = {pid.replace('_CA_1_validation', ''): pid for pid in all_product_ids}
    display_names = sorted(display_to_real.keys())

    selected_display = st.selectbox(
        "Product",
        options=display_names,
        index=0,
        help=f"{len(display_names)} FOODS products available in CA_1.",
    )
    selected_product = display_to_real[selected_display]  # real ID used for all lookups below

    st.divider()
    st.caption("Built on the M5 Walmart dataset · CA_1 · FOODS category")
    st.caption(f"Model: LightGBM · Test RMSE: {results_store['model_training']['lgb_rmse']:.4f}")

# Make selections available to the rest of the app
st.session_state['selected_date'] = selected_date
st.session_state['selected_product'] = selected_product

# ── Main content header ───────────────────────────────────────────────────────
st.markdown(
    """
    <div style='background: linear-gradient(135deg, #131921 0%, #232F3E 100%);
                padding: 28px 32px; border-radius: 10px; margin-bottom: 8px;'>
        <h1 style='color: white; margin: 0; font-size: 2rem;'>📦 ExplainStock</h1>
        <p style='color: #D5D9DD; margin: 6px 0 0 0; font-size: 1rem;'>
            Explainable inventory decisions for Amazon SCOT-style seller transparency
        </p>
        <p style='color: #FF9900; margin: 10px 0 0 0; font-size: 0.85rem; font-weight: 600;'>
            LightGBM · SHAP · Custom Counterfactual Engine · LLaMA 3.3
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Section A: Forecast Chart ─────────────────────────────────────────────────
import plotly.graph_objects as go

st.markdown("### 📈 Forecast: Actual vs. Predicted")

product_data = test_df[test_df['id'] == selected_product].sort_values('date').copy()

if len(product_data) == 0:
    st.warning("No data available for this product in the test period.")
else:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=product_data['date'], y=product_data['units_sold'],
        mode='lines+markers', name='Actual',
        line=dict(color='#131921', width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=product_data['date'], y=product_data['predicted'],
        mode='lines+markers', name='Predicted (LightGBM)',
        line=dict(color='#FF9900', width=2, dash='dash'),
        marker=dict(size=6, symbol='diamond'),
    ))

    # Mark the selected date
    selected_row = product_data[product_data['date'] == pd.to_datetime(selected_date)]
    if len(selected_row) > 0:
        fig.add_shape(
            type="line",
            x0=selected_date, x1=selected_date,
            y0=0, y1=1, yref="paper",
            line=dict(dash="dot", color="grey", width=1.5),
        )
        fig.add_annotation(
            x=selected_date, y=1.05, yref="paper",
            text="Selected date", showarrow=False,
            font=dict(size=11, color="grey"),
        )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None, yaxis_title="Units Sold",
        plot_bgcolor='white',
        hovermode='x unified',
    )

    st.plotly_chart(fig, width='stretch')
    st.caption(
        f"📋 Held-out test period ({product_data['date'].min().date()} to "
        f"{product_data['date'].max().date()}) — actual ground truth available for "
        f"this entire window. Not a forward-looking forecast."
    )

    # ── Section B: Restock Decision Card ──────────────────────────────────────────
import math

st.markdown("### 📋 Restock Decision")

selected_row = product_data[product_data['date'] == pd.to_datetime(selected_date)]

if len(selected_row) == 0:
    st.warning("No prediction available for this exact date.")
else:
    row = selected_row.iloc[0]
    raw_prediction = max(row['predicted'], 0)

    # Reorder logic from documented Fix 1: 1.3x safety factor, ceiling, min 1
    SAFETY_FACTOR = 1.3
    reorder_qty = max(1, math.ceil(raw_prediction * SAFETY_FACTOR))

    # Urgency: based on current rolling trend vs. typical level
    roll_7 = row.get('roll_mean_7', np.nan)
    roll_28 = row.get('roll_mean_28', np.nan)
    if pd.notna(roll_7) and pd.notna(roll_28) and roll_28 > 0:
        trend_ratio = roll_7 / roll_28
    else:
        trend_ratio = 1.0

    # Urgency based on DEVIATION from this product's own normal level,
    # not absolute volume — a low-volume product at its normal level isn't urgent.
    if trend_ratio > 1.3:
        urgency, urgency_color = "HIGH", "#D32F2F"
    elif trend_ratio > 1.05 or trend_ratio < 0.7:
        urgency, urgency_color = "MEDIUM", "#F57F17"
    else:
        urgency, urgency_color = "LOW", "#2E7D32"

    if reorder_qty <= 1 and raw_prediction < 0.3:
        st.caption("ℹ️ Low-volume product — minimum order quantity applied (see Fix 1 in project docs).")

    # Estimated stockout: naive days-of-stock implied by current daily rate
    daily_rate = max(raw_prediction, 0.1)  # avoid divide-by-zero
    days_of_stock = reorder_qty / daily_rate
    stockout_date = pd.to_datetime(selected_date) + pd.Timedelta(days=days_of_stock)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Recommended Reorder", f"{reorder_qty} units")
        st.caption(f"Raw model prediction: {raw_prediction:.2f} units · ×1.3 safety factor")

    with col2:
        st.markdown(
            f"<div style='padding: 8px 0;'>"
            f"<span style='background-color:{urgency_color}; color:white; "
            f"padding: 4px 14px; border-radius: 4px; font-weight: 600; font-size: 0.9rem;'>"
            f"{urgency} URGENCY</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(f"7-day trend vs. 28-day average: {trend_ratio:.2f}x")

    with col3:
        st.metric("Est. Stockout Date", stockout_date.strftime("%b %d, %Y"))
        st.caption(f"Estimate only — reorder qty ÷ daily rate. Not validated to same standard as model metrics.")

    if row.get('is_snap', 0) == 1:
        st.info("📅 This is a SNAP benefit day — demand on full test set runs ~18.75% higher on these days (95% CI: 13.8%–23.8%).")

# ── Section C: SHAP Explanation ───────────────────────────────────────────────
import shap

st.markdown("### 🔍 Why This Recommendation?")

# Plain-English labels for technical feature names (per original spec)
FEATURE_LABELS = {
    'roll_mean_7': '7-day average sales',
    'roll_mean_14': '14-day average sales',
    'roll_mean_28': '28-day average sales',
    'roll_std_7': '7-day sales volatility',
    'roll_std_14': '14-day sales volatility',
    'roll_std_28': '28-day sales volatility',
    'lag_7': 'Sales 7 days ago',
    'lag_14': 'Sales 14 days ago',
    'lag_28': 'Sales 28 days ago',
    'sell_price': 'Current price',
    'price_vs_mean': 'Price vs. typical price',
    'price_change_pct': 'Recent price change',
    'price_lag_7': 'Price 7 days ago',
    'is_price_drop': 'Price drop',
    'is_snap': 'SNAP benefit day',
    'is_weekend': 'Weekend',
    'day_of_week': 'Day of week',
    'day_of_month': 'Day of month',
    'week_of_year': 'Week of year',
    'quarter': 'Quarter of year',
    'is_month_start': 'Start of month',
    'is_month_end': 'End of month',
    'is_event': 'Calendar event',
    'is_national_holiday': 'National holiday',
    'is_cultural_event': 'Cultural event',
    'is_religious_event': 'Religious event',
    'is_sporting_event': 'Sporting event',
}

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_shap_explainer(model)

if len(selected_row) == 0:
    st.warning("No data available for SHAP explanation on this date.")
else:
    row_features = selected_row[FEATURE_COLS]
    shap_values_row = explainer.shap_values(row_features, check_additivity=False)
    if isinstance(shap_values_row, list):
        shap_values_row = shap_values_row[0]
    shap_values_row = np.array(shap_values_row).flatten()

    shap_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'shap_value': shap_values_row,
    })
    shap_df['label'] = shap_df['feature'].map(lambda f: FEATURE_LABELS.get(f, f))
    shap_df['abs_value'] = shap_df['shap_value'].abs()
    top5 = shap_df.nlargest(5, 'abs_value').sort_values('shap_value')

    fig_shap = go.Figure()
    colors = ['#FF9900' if v > 0 else '#131921' for v in top5['shap_value']]
    fig_shap.add_trace(go.Bar(
        x=top5['shap_value'], y=top5['label'],
        orientation='h', marker_color=colors,
        text=[f"{v:+.3f}" for v in top5['shap_value']],
        textposition='outside',
    ))
    fig_shap.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Impact on prediction (units)",
        yaxis_title=None,
        plot_bgcolor='white',
    )
    st.plotly_chart(fig_shap, width='stretch')
    st.caption(
        "🟧 Orange = pushes prediction up · ⬛ Navy = pushes prediction down. "
        "Top feature varies by date — see project notes: a stable cluster of "
        "7–28 day rolling features (not one single feature) drives most predictions."
    )

    # ── Section D: Counterfactual What-If Sliders ─────────────────────────────────
st.markdown("### 🎛️ What If...?")

if len(selected_row) == 0:
    st.warning("No data available for counterfactual analysis on this date.")
else:
    base_row = selected_row.iloc[0]
    base_features = selected_row[FEATURE_COLS].copy()

    def run_counterfactual_scenario(base_features_df, changes: dict):
        """Custom counterfactual engine (per locked decision: not DiCE).
        Applies feature changes to a copy of the base row, re-predicts."""
        modified = base_features_df.copy()
        for feat, new_val in changes.items():
            if feat in modified.columns:
                modified[feat] = new_val
        new_pred = max(model.predict(modified)[0], 0)
        return new_pred

    base_pred = max(model.predict(base_features)[0], 0)

    # ── Model-backed scenarios ────────────────────────────────────────────────
    st.markdown(
        "<div style='background-color:#F0F4F8; padding:14px 18px; border-radius:8px; "
        "border-left:4px solid #131921;'>"
        "<strong>📊 Model-backed scenarios</strong><br>"
        "<span style='color:#555; font-size:0.85rem;'>These sliders feed your real "
        "LightGBM model and show its actual re-prediction.</span></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        price_change_pct = st.slider(
            "Price change", min_value=-20, max_value=20, value=0, step=5,
            format="%d%%", key="price_slider",
        )
        if price_change_pct != 0 and 'sell_price' in base_features.columns:
            new_price = base_row['sell_price'] * (1 + price_change_pct / 100)
            cf_pred_price = run_counterfactual_scenario(base_features, {'sell_price': new_price})
            delta_price = cf_pred_price - base_pred
            st.metric(
                "Predicted demand", f"{cf_pred_price:.2f} units",
                delta=f"{delta_price:+.2f} vs. base ({base_pred:.2f})",
            )
            if abs(delta_price) < 0.005:
                st.caption(
                    "ℹ️ No change: LightGBM makes piecewise-constant predictions — "
                    "this input shift didn't cross a decision threshold in the model's trees."
                )
        else:
            st.metric("Predicted demand", f"{base_pred:.2f} units", delta="No change")
            
    with col2:
        trend_change_pct = st.slider(
            "Demand trend change", min_value=-50, max_value=50, value=0, step=10,
            format="%d%%", key="trend_slider",
        )
        if trend_change_pct != 0 and 'roll_mean_7' in base_features.columns:
            new_roll7 = base_row['roll_mean_7'] * (1 + trend_change_pct / 100)
            changes = {'roll_mean_7': new_roll7}
            if 'lag_7' in base_features.columns:
                changes['lag_7'] = base_row['lag_7'] * (1 + trend_change_pct / 100)
            cf_pred_trend = run_counterfactual_scenario(base_features, changes)
            delta_trend = cf_pred_trend - base_pred
            st.metric(
                "Predicted demand", f"{cf_pred_trend:.2f} units",
                delta=f"{delta_trend:+.2f} vs. base ({base_pred:.2f})",
            )
            if abs(delta_trend) < 0.005:
                st.caption(
                    "ℹ️ No change: LightGBM makes piecewise-constant predictions — "
                    "this input shift didn't cross a decision threshold in the model's trees."
                )
        else:
            st.metric("Predicted demand", f"{base_pred:.2f} units", delta="No change")

    st.divider()

    # ── Stockout estimate adjustment (heuristic only, NOT model-backed) ───────
    st.markdown(
        "<div style='background-color:#FFF8E1; padding:14px 18px; border-radius:8px; "
        "border-left:4px solid #F57F17;'>"
        "<strong>⏱️ Stockout estimate adjustment</strong><br>"
        "<span style='color:#7A5C00; font-size:0.85rem;'>This slider does NOT feed the "
        "model. It only adjusts the heuristic stockout-date calculation from the Restock "
        "Decision card above — same unvalidated estimate, just shifted by lead time.</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    lead_time_days = st.slider(
        "Lead time adjustment", min_value=-5, max_value=5, value=0, step=1,
        format="%d days", key="lead_time_slider",
    )

    if lead_time_days != 0:
        adjusted_stockout = stockout_date + pd.Timedelta(days=lead_time_days)
        st.metric(
            "Adjusted stockout estimate",
            adjusted_stockout.strftime("%b %d, %Y"),
            delta=f"{lead_time_days:+d} days vs. base estimate ({stockout_date.strftime('%b %d')})",
        )
    else:
        st.metric("Adjusted stockout estimate", stockout_date.strftime("%b %d, %Y"), delta="No change")

            # ── Bottom Section: LLM Narrative Report ──────────────────────────────────────
st.divider()
st.markdown("### 📝 Seller Report")

# Match against the clean display name, since seller_reports.json is keyed by
# the short product ID (e.g. "FOODS_1_072"), not the full "_CA_1_validation" id
short_id = selected_display if 'selected_display' in dir() else selected_product.replace('_CA_1_validation', '')

if short_id in seller_reports:
    report_entry = seller_reports[short_id]
    report_text = report_entry.get('report', '')
    report_data = report_entry.get('data', {})

    st.markdown(
        f"<div style='background-color:#FAFAFA; border:1px solid #E0E0E0; "
        f"border-radius:8px; padding:20px 24px;'>"
        f"<pre style='white-space:pre-wrap; font-family:inherit; margin:0; "
        f"font-size:0.95rem; line-height:1.5;'>{report_text}</pre></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Generated by LLaMA 3.3 via Groq · Report date: {report_data.get('date', 'N/A')} · "
        f"Pre-generated for {len(seller_reports)} sample products (Day 9 batch run)."
    )
else:
    st.info(
        f"📋 No pre-generated narrative report exists for **{short_id}**. "
        f"Reports were generated for a 5-product sample spanning different demand levels "
        f"({', '.join(seller_reports.keys())}) as a proof of concept, not the full catalog."
    )

    if st.button("🤖 Generate live report for this product", key="generate_live_report"):
        with st.spinner("Calling LLaMA 3.3 via Groq..."):
            try:
                # Build report_data dict directly from data already live in this session
                # (adapter for generate_seller_report — skips build_report_data, which
                # expects pre-batched multi-row dataframes this single-row dashboard
                # view doesn't construct)
                live_report_data = {
                    "product_id": short_id,
                    "store": "CA_1",
                    "date": str(pd.to_datetime(selected_date).date()),
                    "reorder_qty": reorder_qty,
                    "actual_sold": round(float(base_row['units_sold']), 1),
                    "sell_price": round(float(base_row.get('sell_price', 0)), 2),
                    "is_snap_day": bool(base_row.get('is_snap', 0)),
                    "is_event_day": bool(base_row.get('is_event', 0)),
                    "trend_7day_avg": round(float(base_row.get('roll_mean_7', 0)), 4),
                    "trend_28day_avg": round(float(base_row.get('roll_mean_28', 0)), 4),
                    "trend_direction": "upward" if base_row.get('roll_mean_7', 0) > base_row.get('roll_mean_28', 0) else "downward",
                    "top_features": dict(zip(top5['feature'], top5['shap_value'].round(4))),
                    "counterfactuals": [
                        {"scenario": "Price change", "base_pred": round(base_pred, 4),
                         "cf_pred": round(cf_pred_price, 4) if 'cf_pred_price' in dir() else round(base_pred, 4),
                         "delta": round((cf_pred_price - base_pred), 4) if 'cf_pred_price' in dir() else 0.0,
                         "delta_pct": round(((cf_pred_price - base_pred) / max(base_pred, 0.01)) * 100, 2) if 'cf_pred_price' in dir() else 0.0},
                    ],
                    "snap_demand_lift_pct": 18.75,  # from validated Step 5.5 finding
                    "price_elasticity": -0.3764,    # from validated Step 6.5 finding
                }
# Compute each scenario's prediction ONCE, then derive delta/delta_pct
                cf_pred_snap = run_counterfactual_scenario(base_features, {'is_snap': 0.0})

                trend_up_changes = {
                    'roll_mean_7': float(base_row.get('roll_mean_7', 0)) * 1.5,
                    'lag_7': float(base_row.get('lag_7', 0)) * 1.5,
                }
                cf_pred_trend_up = run_counterfactual_scenario(base_features, trend_up_changes)

                cf_pred_price_for_report = cf_pred_price if 'cf_pred_price' in dir() else base_pred

                def _cf_entry(scenario_name, cf_pred_value):
                    delta = cf_pred_value - base_pred
                    delta_pct = (delta / max(base_pred, 0.01)) * 100
                    return {
                        "scenario": scenario_name,
                        "base_pred": round(base_pred, 4),
                        "cf_pred": round(cf_pred_value, 4),
                        "delta": round(delta, 4),
                        "delta_pct": round(delta_pct, 2),
                    }

                live_report_data["counterfactuals"] = [
                    _cf_entry("Remove SNAP day", cf_pred_snap),
                    _cf_entry("Demand trend +50%", cf_pred_trend_up),
                    _cf_entry("Price change (slider)", cf_pred_price_for_report),
                ]
                st.session_state['live_report_text'] = generate_seller_report(live_report_data, groq_client)
                live_report_text = st.session_state['live_report_text']

                st.markdown(
                    f"<div style='background-color:#FAFAFA; border:1px solid #E0E0E0; "
                    f"border-radius:8px; padding:20px 24px; margin-top:12px;'>"
                    f"<pre style='white-space:pre-wrap; font-family:inherit; margin:0; "
                    f"font-size:0.95rem; line-height:1.5;'>{live_report_text}</pre></div>",
                    unsafe_allow_html=True,
                )
                st.caption("⚡ Generated live via Groq (LLaMA 3.3) — not pre-cached, may take a few seconds to regenerate on rerun.")
            except Exception as e:
                st.error(f"Live generation failed: {e}")

                # ── PDF Export ─────────────────────────────────────────────────────────────────
st.divider()

def build_pdf_report():
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    import io
    import re

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                             topMargin=0.7*inch, bottomMargin=0.7*inch)
    styles = getSampleStyleSheet()

    navy = colors.HexColor("#131921")
    orange = colors.HexColor("#FF9900")

    title_style = ParagraphStyle('TitleCustom', parent=styles['Title'],
                                   textColor=navy, fontSize=20)
    h2_style = ParagraphStyle('H2Custom', parent=styles['Heading2'],
                                textColor=navy, fontSize=13, spaceBefore=14)
    body_style = ParagraphStyle('BodyCustom', parent=styles['Normal'], fontSize=10, leading=14)
    caption_style = ParagraphStyle('CaptionCustom', parent=styles['Normal'],
                                     fontSize=8, textColor=colors.grey)

    story = []
    story.append(Paragraph("ExplainStock — Inventory Decision Report", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"{short_id} &nbsp;&middot;&nbsp; CA_1 &nbsp;&middot;&nbsp; "
        f"{pd.to_datetime(selected_date).strftime('%B %d, %Y')}", body_style))
    story.append(Spacer(1, 16))

    # Restock decision table
    story.append(Paragraph("Restock Decision", h2_style))
    decision_data = [
        ["Recommended Reorder", f"{reorder_qty} units"],
        ["Urgency", urgency],
        ["Est. Stockout Date", stockout_date.strftime("%b %d, %Y")],
        ["Raw Model Prediction", f"{raw_prediction:.2f} units"],
    ]
    t = Table(decision_data, colWidths=[2.3*inch, 3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F2F2F2")),
        ('TEXTCOLOR', (0, 0), (0, -1), navy),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Paragraph(
        "Estimate only — reorder qty ÷ daily rate. Not validated to same standard as model metrics.",
        caption_style))
    story.append(Spacer(1, 14))

    # Top SHAP factors table
    story.append(Paragraph("Why This Recommendation — Top Factors", h2_style))
    shap_table_data = [["Factor", "Impact (units)"]] + [
        [row['label'], f"{row['shap_value']:+.3f}"]
        for _, row in top5.sort_values('shap_value', ascending=False).iterrows()
    ]
    t2 = Table(shap_table_data, colWidths=[3.3*inch, 2*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), navy),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F8F8")]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 14))

    # Seller report text, if available
    story.append(Paragraph("Seller Report", h2_style))
    if short_id in seller_reports:
        report_src = seller_reports[short_id].get('report', '')
    elif st.session_state.get('live_report_text'):
        report_src = st.session_state['live_report_text']
    else:
        report_src = "No narrative report was generated for this product in this session."

    # Strip markdown symbols for clean PDF body text
    clean_text = re.sub(r'^#+\s*', '', report_src, flags=re.MULTILINE)
    clean_text = clean_text.replace('**', '')
    for para in clean_text.split('\n\n'):
        if para.strip():
            story.append(Paragraph(para.strip().replace('\n', '<br/>'), body_style))
            story.append(Spacer(1, 8))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Built on the M5 Walmart dataset · CA_1 · FOODS category · "
        f"Model: LightGBM · Test RMSE: {results_store['model_training']['lgb_rmse']:.4f}",
        caption_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

if len(selected_row) > 0:
    pdf_buffer = build_pdf_report()
    st.download_button(
        label="📄 Export this report as PDF",
        data=pdf_buffer,
        file_name=f"explainstock_{short_id}_{pd.to_datetime(selected_date).strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        key=f"pdf_export_{short_id}_{selected_date}",
    )
else:
    st.caption("PDF export unavailable — no prediction data for this product/date combination.")