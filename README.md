# Vancouver CityBrain

A data-driven pavement risk assessment system for Vancouver, developed as a COMP 9130 final project. The repository combines geospatial feature engineering, tabular and neural network models, multi-model stacking, and an interactive dashboard to prioritize pavement maintenance risk.

## Project Overview

Vancouver CityBrain predicts pavement risk using open City of Vancouver infrastructure, traffic, complaint, and weather datasets. The model is trained to classify pavement condition into three risk categories: `Low`, `Medium`, and `High`.

Key capabilities:
- Multi-source geospatial feature engineering
- Road-level and tabular feature fusion using a transformer-style architecture
- 10-fold stacking ensemble with an XGBoost meta-learner
- Ordinal-aware loss functions for safer risk prediction
- Interactive dashboard for model outputs, version comparison, and SHAP analysis

## Highlights

- **Target**: 3-class pavement risk classification for road pavement condition
- **Final model**: `CityBrain v15` with SMOTE-only resampling and a regularised XGBoost meta-learner
- **Core architecture**: Road-MLP (12d) + Tabular-MLP (44d) → Cross-Attention Fusion → 10-Fold model stacking
- **Final performance**: Macro F1 = **0.5541** on the held-out validation/test dataset
- **Improvement**: +0.1626 absolute F1 gain over the baseline v1 model (0.3915 → 0.5541)

## Repository Structure

```text
vancouver-city-brain/
├── code/
│   ├── CityBrain_v_finished.ipynb         # Final modelling pipeline
│   ├── EDA/                               # Exploratory data analysis notebooks
│   │   ├── CityBrain_InfraEDA.ipynb
│   │   └── CityBrain_initial_EDA.ipynb
│   └── Improve_process/                   # Model iteration history
│       ├── CityBrain_v1_Baseline.ipynb
│       ├── CityBrain_v10_Stacking.ipynb
│       ├── CityBrain_v11_RichFeatures.ipynb
│       ├── CityBrain_v12_Improved.ipynb
│       ├── CityBrain_v13_StackingPlus.ipynb
│       ├── CityBrain_v2_Enhanced.ipynb
│       ├── CityBrain_v3_GatedFusion.ipynb
│       ├── CityBrain_v5_Rebalanced3Class.ipynb
│       ├── CityBrain_v6_Rebalanced3Class.ipynb
│       ├── CityBrain_v7_Feature_enigineering.ipynb
│       ├── CityBrain_v8_TwoBranch.ipynb
│       └── CityBrain_v9_CleanFeatures.ipynb
├── figures/                               # Visualizations and dashboard screenshots
├── webapp/
│   ├── app.py                             # Streamlit dashboard application
│   ├── requirements.txt                   # Webapp dependencies
│   └── data/                              # Dashboard CSV data exports
└── README.md                              # Project documentation
```

## Results

The final pipeline is supported by a version history with gradual model improvements:

| Version | Macro F1 | Key change |
|--------|----------|------------|
| v1 | 0.3915 | XGBoost baseline |
| v5 | 0.4941 | 3-class label rebalancing |
| v11 | 0.5322 | +18 infrastructure features |
| v13 | 0.5452 | 10-fold stacking + Optuna HPO |
| v15 | 0.5541 | SMOTE-only + regularised meta-learner |

Major results:

- **Final model macro F1**: 0.5541
- **Relative performance lift**: +41.6% from the baseline v1 model
- **Primary risk classes**: Low / Medium / High, with class-specific penalties for severe misclassification

## Technical Summary

The final model architecture is built around a hybrid fusion and ensemble pipeline:

- **Road input branch**: 12 road-specific features, encoded with a 128→64 MLP
- **Tabular input branch**: 44 engineered infrastructure, weather, complaint, and spatial features, encoded with a 256→128 MLP
- **Fusion layer**: cross-attention fusion that combines road and tabular embeddings
- **Stacking ensemble**: predictions from the fusion model and several tree-based learners are combined via 10-fold stacking
- **Meta-learner**: regularised XGBoost receives stacked out-of-fold predictions and learns the final combination
- **Threshold tuning**: differential evolution optimizes class thresholds for the ordinal risk labels

Additional modeling design:
- SMOTE-only resampling for balanced training data
- Custom loss functions for focal weighting, ordinal penalty, and cost-sensitive misclassification
- Geospatial feature extraction from infrastructure layers, street network, and complaint density

## Getting Started

### 1. Install dependencies

```bash
cd vancouver-city-brain/webapp
python -m pip install -r requirements.txt
```

The notebook pipelines also require:
- `pandas`
- `numpy`
- `scikit-learn`
- `torch`
- `xgboost` / `lightgbm` / `catboost`
- `scipy`

### 2. Prepare data

Real-world data is not included in the repository. The notebook expects preprocessed CSV exports in the project data directory.

The core data used by the project includes:
- Pavement condition labels and geometry
- Public street network and street-use attributes
- Right-of-way widths
- Repair and infrastructure project data
- Water mains, sewer, tree, permit, bus, and utility datasets

### 3. Run the notebook

Open `code/CityBrain_v_finished.ipynb` and execute the cells sequentially. The notebook covers:
- data loading and cleaning
- spatial and infrastructure feature engineering
- model training, evaluation, and threshold tuning
- result export for the dashboard

### 4. Launch the dashboard

```bash
cd vancouver-city-brain/webapp
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Notes

- The dashboard reads precomputed CSV data from `webapp/data/`.
- The final model is designed to protect against severe misclassification by penalizing distance between predicted and true risk levels.
- The repository documents model evolution across versions v1 through v15.

## License and Credits

This project was developed for academic purposes as a COMP 9130 final project.
All data and code are intended for research and analysis of public infrastructure risk.
