# NBA Lineup Net Rating Predictor

A machine learning project that predicts 5-player NBA lineup performance using player-level advanced stats and lineup chemistry features, built to mirror roster construction and in-game strategy analysis done by NBA analytics departments.

## What it does

- Pulls 10,000+ lineup combinations across 5 NBA seasons (2019–2024) directly from the NBA Stats API
- Engineers synergy features measuring how much a lineup outperforms or underperforms the sum of its individual players
- Trains and compares Ridge Regression, Random Forest, and XGBoost models to predict lineup net rating
- Applies SHAP analysis to explain which player combinations and stats drive lineup performance
- Deploys an interactive Streamlit dashboard for exploring top lineups, team breakdowns, and synergy leaders across all 30 NBA teams

## Key finding

SHAP analysis revealed that effective field goal percentage (EFG%) is the dominant driver of lineup net rating — more predictive than individual player ratings alone. Lineups with the highest synergy scores (actual net rating exceeding what individual stats would predict) consistently featured strong defensive rebounding and low turnover rates regardless of star power.

## Model results

| Model | RMSE | R² |
|---|---|---|
| Ridge Regression | 15.35 | 0.517 |
| Random Forest | 16.13 | 0.466 |
| XGBoost | 16.07 | 0.470 |

Train/test split by season — trained on 2019–20 through 2022–23, tested on 2023–24.

## Tech stack

| Tool | How it was used |
|---|---|
| `nba_api` | Pulled live lineup and player stats from NBA Stats API |
| `pandas` / `numpy` | Feature engineering and synergy score computation |
| `scikit-learn` | Ridge Regression, StandardScaler, train/test evaluation |
| `XGBoost` | Gradient boosting model for non-linear synergy patterns |
| `SHAP` | Feature importance and model interpretability |
| `Streamlit` | Interactive dashboard for lineup exploration |
| `Plotly` / `matplotlib` | Visualizations inside dashboard and EDA |

## Project structure

```
nba-lineup-model/
├── data/
│   ├── raw/                   # pulled from NBA Stats API (not committed)
│   └── processed/             # engineered features, saved model
├── notebooks/                 # EDA charts (distributions, correlations, SHAP)
├── src/
│   ├── pull_data.py           # pulls lineup + player data from nba_api
│   ├── feature_engineering.py # builds synergy features, joins player stats
│   ├── model.py               # trains Ridge, Random Forest, XGBoost
│   └── shap_analysis.py       # SHAP feature importance plots
└── app/
    └── dashboard.py           # Streamlit dashboard
```

## Run locally

```bash
git clone https://github.com/shobithabhaskar18/nba-lineup-model.git
cd nba-lineup-model
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/pull_data.py
python src/feature_engineering.py
python src/model.py
streamlit run app/dashboard.py
```

## Dashboard features

- **Top Lineups** — filter by season and minimum minutes to see the highest performing lineups
- **Team Explorer** — select any NBA team and season to explore all their lineup combinations with net rating and synergy scores
- **Synergy Leaders** — lineups that most outperform what individual player stats would predict, with actual vs expected net rating comparison
