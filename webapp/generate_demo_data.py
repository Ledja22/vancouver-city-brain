"""
Generate demo data for CityBrain dashboard testing.
Run this ONCE to create sample CSVs in webapp/data/.
Replace with real data exported from Colab when ready.
"""

import pandas as pd
import numpy as np
import pathlib

DATA_DIR = pathlib.Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)
N = 2000  # demo road segments

# Vancouver bounding box
lat = np.random.uniform(49.20, 49.30, N)
lon = np.random.uniform(-123.22, -123.02, N)

# True labels: 0=Low, 1=Medium, 2=High
true_label = np.random.choice([0, 1, 2], N, p=[0.45, 0.35, 0.20])

# Generate correlated predictions (not perfect, ~55% macro F1)
pred_label = true_label.copy()
flip_mask = np.random.random(N) < 0.35  # 35% error rate
pred_label[flip_mask] = np.random.choice([0, 1, 2], flip_mask.sum())

# Probabilities
probs = np.random.dirichlet([2, 2, 2], N)
# Make probs somewhat consistent with predictions
for i in range(N):
    probs[i, pred_label[i]] += 0.3
probs = probs / probs.sum(axis=1, keepdims=True)

# Features
neighbourhoods = np.random.choice([
    "Kitsilano", "Mount Pleasant", "Downtown", "Fairview",
    "Grandview-Woodland", "Hastings-Sunrise", "Renfrew-Collingwood",
    "Kensington-Cedar Cottage", "Riley Park", "Dunbar-Southlands",
    "Kerrisdale", "Marpole", "Oakridge", "Shaughnessy",
    "South Cambie", "Strathcona", "Sunset", "Victoria-Fraserview",
    "West End", "West Point Grey",
], N)

# Model-specific predictions (slightly different)
pred_fusion = true_label.copy()
fusion_mask = np.random.random(N) < 0.40
pred_fusion[fusion_mask] = np.random.choice([0, 1, 2], fusion_mask.sum())
pred_xgb = true_label.copy()
xgb_mask = np.random.random(N) < 0.38
pred_xgb[xgb_mask] = np.random.choice([0, 1, 2], xgb_mask.sum())
pred_stacked = true_label.copy()
stack_mask = np.random.random(N) < 0.36
pred_stacked[stack_mask] = np.random.choice([0, 1, 2], stack_mask.sum())

export = pd.DataFrame({
    "lat": lat,
    "lon": lon,
    "true_label": true_label,
    "pred_label": pred_label,
    "prob_low": probs[:, 0],
    "prob_medium": probs[:, 1],
    "prob_high": probs[:, 2],
    "traffic_load": np.random.exponential(5, N),
    "est_pavement_age": np.random.uniform(0, 40, N),
    "streetuse": np.random.choice(["Arterial", "Collector", "Local"], N),
    "length_m": np.random.uniform(20, 500, N),
    "water_main_avg_age": np.random.uniform(10, 80, N),
    "is_truck_route": np.random.choice([0, 1], N, p=[0.85, 0.15]),
    "is_snow_route": np.random.choice([0, 1], N, p=[0.80, 0.20]),
    "neighbourhood": neighbourhoods,
    "tree_count_30m": np.random.poisson(3, N),
    "sewer_combined_pct": np.random.uniform(0, 1, N),
    "slope_pct": np.random.uniform(0, 15, N),
    "elevation_m": np.random.uniform(0, 150, N),
    "complaint_total": np.random.poisson(2, N),
    "is_bikeway": np.random.choice([0, 1], N, p=[0.88, 0.12]),
    "utility_density": np.random.uniform(0, 20, N),
    "drainage_risk": np.random.uniform(0, 1, N),
    "ROW_width": np.random.uniform(8, 30, N),
    "pred_fusion": pred_fusion,
    "pred_xgb": pred_xgb,
    "pred_stacked": pred_stacked,
    "pred_tuned": pred_label,
    "fusion_prob_low": np.random.dirichlet([2, 2, 2], N)[:, 0],
    "fusion_prob_med": np.random.dirichlet([2, 2, 2], N)[:, 1],
    "fusion_prob_high": np.random.dirichlet([2, 2, 2], N)[:, 2],
    "xgb_prob_low": np.random.dirichlet([2, 2, 2], N)[:, 0],
    "xgb_prob_med": np.random.dirichlet([2, 2, 2], N)[:, 1],
    "xgb_prob_high": np.random.dirichlet([2, 2, 2], N)[:, 2],
    "stacked_prob_low": np.random.dirichlet([2, 2, 2], N)[:, 0],
    "stacked_prob_med": np.random.dirichlet([2, 2, 2], N)[:, 1],
    "stacked_prob_high": np.random.dirichlet([2, 2, 2], N)[:, 2],
})
export.to_csv(DATA_DIR / "citybrain_dashboard_data.csv", index=False)

# SHAP data
shap_data = pd.DataFrame({
    "feature": [
        "sl_risk_7", "sl_high_15", "sl_risk_15", "sl_risk_3", "source",
        "sl_high_7", "pothole_total", "is_bikeway", "length",
        "tree_avg_height", "sewer_combined_pct", "sl_high_3",
        "sl_med_7", "tree_avg_diameter", "permit_count_200m",
    ],
    "importance": [
        0.2193, 0.1251, 0.1173, 0.1139, 0.1066,
        0.0821, 0.0724, 0.0703, 0.0642,
        0.0571, 0.0503, 0.0491,
        0.0486, 0.0474, 0.0452,
    ],
})
shap_data.to_csv(DATA_DIR / "citybrain_shap.csv", index=False)

# Version history
versions = pd.DataFrame({
    "version": ["v1", "v2", "v3", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"],
    "f1": [0.3915, 0.4411, 0.4495, 0.4941, 0.4941, 0.5209, 0.5183, 0.5207, 0.5259, 0.5322, 0.5040, 0.5452, 0.5332, 0.5446],
    "key_change": [
        "XGBoost baseline",
        "Focal Loss + feature engineering",
        "Gated Attention Fusion",
        "Rebalanced 3-class labels",
        "CrossAttention + Ensemble",
        "3-model ensemble",
        "2-branch (removed temporal)",
        "Clean feature separation + spatial lag",
        "5-Fold CV Stacking",
        "18 infrastructure features",
        "ResidualBlock + Transformer + SMOTE-ENN",
        "10-Fold Stacking + Optuna HPO",
        "No resampling (regression)",
        "SMOTE-only + regularised meta-learner",
    ],
})
versions.to_csv(DATA_DIR / "citybrain_versions.csv", index=False)

# Per-model F1
model_f1s = pd.DataFrame({
    "model": ["Road-MLP", "Tabular-MLP", "Fusion", "XGBoost", "CatBoost", "LightGBM", "ExtraTrees", "Stacked", "Tuned"],
    "f1": [0.3812, 0.4523, 0.4891, 0.5102, 0.5045, 0.4987, 0.4876, 0.5411, 0.5446],
})
model_f1s.to_csv(DATA_DIR / "citybrain_model_f1s.csv", index=False)

print(f"Demo data generated in {DATA_DIR}/")
print("Files: citybrain_dashboard_data.csv, citybrain_shap.csv, citybrain_versions.csv, citybrain_model_f1s.csv")
