# Vancouver CityBrain

A COMP 9130 final project that builds a pavement risk assessment pipeline for Vancouver roads. The project uses City of Vancouver open data to predict pavement condition risk and prioritize maintenance decisions with an end-to-end notebook implementation.

## Problem Description and Motivation

City pavement maintenance requires strategic decision-making to allocate limited budgets and improve safety. Traditional inspections are costly and often reactive. This project aims to make pavement risk prediction proactive by using data-driven features from infrastructure, street geometry, complaints, weather, and repair history.

Motivation:
- Reduce costly pavement failures by identifying high-risk segments early
- Support asset management decisions with objective, spatially-aware risk predictions
- Leverage public City of Vancouver data to improve transparency and explainability

## Dataset Description and Source

The modeling pipeline is built from multiple City of Vancouver open datasets, including:

- **Pavement condition** — road condition labels and segment geometry
- **Public streets / street use** — road classifications and street network geometry
- **Right-of-way widths** — corridor widths and street corridor features
- **Water mains** — pipe count, average age, and density around segments
- **Sewer system** — drainage risk and sewer infrastructure context
- **Street trees** — tree count, diameter, height, and root pressure proxies
- **Building permits** — development activity proxies via permit count and value
- **Utility assets** — manhole/catch basin density and utility infrastructure exposure
- **Bus / truck / snow routes** — road usage, maintenance priority, and traffic context

Source: https://opendata.vancouver.ca/

## Setup Instructions and How to Run

### 1. Install dependencies

Install the root dependencies used by the notebook pipeline:

```bash
cd /Users/ledjahalltari/vancouver-city-brain
python -m pip install -r requirements.txt
```

If you want to run the dashboard separately, install its dependencies from `webapp/requirements.txt`.

### 2. Prepare data

Place the required CSV datasets into the local `data/` folder, or mount them in Google Drive for Colab use.

The finished notebook `code/CityBrain_v_finished.ipynb` looks for:
- `data/pavement_enriched.csv` if available
- otherwise raw CSVs such as `pavement_condition.csv`, `public_streets.csv`, `right_of_way_widths.csv`, `city_project_package_street.csv`, and other infrastructure CSVs

Update the data path in the notebook if needed:

```python
DATA_DIR = '/content/drive/MyDrive/AI-FinalProject/data'
```

or locally:

```python
DATA_DIR = './data'
```

### 3. Run the finished notebook

Open `code/CityBrain_v_finished.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.

Execute the notebook cells sequentially. The notebook covers:
1. imports and environment setup
2. pavement data loading and label mapping
3. geospatial feature engineering for road, infrastructure, complaint, weather, and spatial lag features
4. training, fusion, stacking, and threshold optimization
5. evaluation, metrics reporting, and dashboard export generation

The notebook is the final code artifact for the completed model pipeline.

## Results Summary

The repository tracks model progress through versioned notebooks. Key results include:

- **v1**: XGBoost baseline — Macro F1 = 0.3915
- **v5**: 3-class rebalancing — Macro F1 = 0.4941
- **v11**: added 18 infrastructure features — Macro F1 = 0.5322
- **v13**: 10-fold stacking + Optuna HPO — Macro F1 = 0.5452
- **v15**: SMOTE-only resampling + regularised XGBoost meta-learner — Macro F1 = 0.5541

The final notebook implements the completed version of the pipeline and delivers these improvements through:
- hybrid road/tabular feature fusion
- cross-attention-style fusion of embeddings
- 10-fold stacking ensemble of neural and tree learners
- regularised XGBoost meta-learner
- threshold tuning for ordinal risk labels
- SMOTE-only balancing and custom loss functions

### Final metrics

- **Macro F1**: 0.5541
- **Relative performance lift versus baseline**: +41.6%

## Team Member Contributions

 - **Savina Cai** — primary author of the final notebook pipeline, responsible for code architecture, data ingestion, stacking ensemble training, custom loss implementation, threshold optimization, and quantitative evaluation.
 - **Ledja Halltari** — contributor to data preprocessing and code workflow validation, geospatial feature engineering, hybrid fusion model development, pipeline reproducibility checks, and review of modeling/code process changes.

## Repository Contents

```text
vancouver-city-brain/
├── code/
│   ├── CityBrain_v_finished.ipynb         # Final modelling pipeline
│   ├── EDA/                               # Exploratory data analysis notebooks
│   └── Improve_process/                   # Model iteration history notebooks
├── figures/                               # Visualizations and dashboard screenshots
├── webapp/                                # Streamlit dashboard application
├── data/                                  # Local dataset inputs for notebooks
├── requirements.txt                       # Notebook pipeline dependencies
└── README.md                              # Project documentation
```

## Notes

- This README is focused on the final notebook pipeline and code changes, not the dashboard app.
- Run `code/CityBrain_v_finished.ipynb` before using the dashboard exports in `webapp/data/`.
