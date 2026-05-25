import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

FEATURES = [
    "AVG_PLAYER_NET_RATING", "AVG_PLAYER_TS_PCT", "AVG_PLAYER_PIE",
    "EFG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT", "TM_TOV_PCT", "PACE", "MIN",
    "P1_OFF_RATING", "P1_DEF_RATING", "P1_TS_PCT", "P1_AST_PCT", "P1_PIE",
    "P2_OFF_RATING", "P2_DEF_RATING", "P2_TS_PCT", "P2_AST_PCT", "P2_PIE",
    "P3_OFF_RATING", "P3_DEF_RATING", "P3_TS_PCT", "P3_AST_PCT", "P3_PIE",
    "P4_OFF_RATING", "P4_DEF_RATING", "P4_TS_PCT", "P4_AST_PCT", "P4_PIE",
    "P5_OFF_RATING", "P5_DEF_RATING", "P5_TS_PCT", "P5_AST_PCT", "P5_PIE",
]

def main():
    df = pd.read_parquet("data/processed/lineups_featured.parquet")
    model = joblib.load("data/processed/best_model.pkl")
    feature_names = joblib.load("data/processed/feature_names.pkl")

    test = df[df["SEASON"] == "2023-24"].copy()
    X_test = test[feature_names].fillna(0)

    # Scale using same approach as training
    from sklearn.preprocessing import StandardScaler
    train = df[df["SEASON"] != "2023-24"].copy()
    X_train = train[feature_names].fillna(0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names
    )

    print("Computing SHAP values...")
    explainer = shap.LinearExplainer(model, X_test_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    # Plot 1 — overall feature importance
    plt.figure()
    shap.summary_plot(
        shap_values, X_test_scaled,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.title("Top 15 Features by SHAP Importance")
    plt.tight_layout()
    plt.savefig("notebooks/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_importance.png")

    # Plot 2 — beeswarm (direction of impact)
    plt.figure()
    shap.summary_plot(
        shap_values, X_test_scaled,
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    plt.title("SHAP Feature Impact Direction")
    plt.tight_layout()
    plt.savefig("notebooks/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_beeswarm.png")

    # Print top features as text
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names
    ).sort_values(ascending=False)

    print("\nTop 10 most important features:")
    print(mean_shap.head(10).to_string())

if __name__ == "__main__":
    main()