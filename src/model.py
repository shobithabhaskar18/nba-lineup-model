import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

FEATURES = [
    "AVG_PLAYER_NET_RATING",
    "AVG_PLAYER_TS_PCT",
    "AVG_PLAYER_PIE",
    "EFG_PCT",
    "AST_PCT",
    "OREB_PCT",
    "DREB_PCT",
    "TM_TOV_PCT",
    "PACE",
    "MIN",
    "P1_OFF_RATING", "P1_DEF_RATING", "P1_TS_PCT", "P1_AST_PCT", "P1_PIE",
    "P2_OFF_RATING", "P2_DEF_RATING", "P2_TS_PCT", "P2_AST_PCT", "P2_PIE",
    "P3_OFF_RATING", "P3_DEF_RATING", "P3_TS_PCT", "P3_AST_PCT", "P3_PIE",
    "P4_OFF_RATING", "P4_DEF_RATING", "P4_TS_PCT", "P4_AST_PCT", "P4_PIE",
    "P5_OFF_RATING", "P5_DEF_RATING", "P5_TS_PCT", "P5_AST_PCT", "P5_PIE",
]

TARGET = "NET_RATING"

def load_data():
    df = pd.read_parquet("data/processed/lineups_featured.parquet")
    return df

def split_data(df):
    # Split by season — train on older seasons, test on most recent
    # This prevents data leakage and mirrors real-world usage
    train = df[df["SEASON"] != "2023-24"].copy()
    test = df[df["SEASON"] == "2023-24"].copy()

    available = [f for f in FEATURES if f in df.columns]

    X_train = train[available].fillna(0)
    X_test = test[available].fillna(0)
    y_train = train[TARGET]
    y_test = test[TARGET]

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Features used: {len(available)}")
    return X_train, X_test, y_train, y_test, available

def evaluate(name, y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name:30s} RMSE: {rmse:.2f}  R2: {r2:.3f}")
    return rmse, r2

def train_models(X_train, X_test, y_train, y_test):
    results = {}

    # Model 1 — Ridge Regression (baseline)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge = Ridge(alpha=10)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    rmse, r2 = evaluate("Ridge Regression", y_test, ridge_pred)
    results["ridge"] = {"model": ridge, "rmse": rmse, "r2": r2, "scaler": scaler}

    # Model 2 — Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                               min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rmse, r2 = evaluate("Random Forest", y_test, rf_pred)
    results["rf"] = {"model": rf, "rmse": rmse, "r2": r2}

    # Model 3 — XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    rmse, r2 = evaluate("XGBoost", y_test, xgb_pred)
    results["xgb"] = {"model": xgb_model, "rmse": rmse, "r2": r2}

    return results

def save_best_model(results, feature_names):
    best = min(results, key=lambda k: results[k]["rmse"])
    print(f"\nBest model: {best} (RMSE: {results[best]['rmse']:.2f})")
    joblib.dump(results[best]["model"], "data/processed/best_model.pkl")
    joblib.dump(feature_names, "data/processed/feature_names.pkl")
    print("Model saved to data/processed/best_model.pkl")

def main():
    df = load_data()
    X_train, X_test, y_train, y_test, features = split_data(df)
    results = train_models(X_train, X_test, y_train, y_test)
    save_best_model(results, features)

if __name__ == "__main__":
    main()