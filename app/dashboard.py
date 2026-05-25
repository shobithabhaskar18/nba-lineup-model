import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="NBA Lineup Predictor", layout="wide")

FEATURES = [
    "AVG_PLAYER_NET_RATING", "AVG_PLAYER_TS_PCT", "AVG_PLAYER_PIE",
    "EFG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT", "TM_TOV_PCT", "PACE", "MIN",
    "P1_OFF_RATING", "P1_DEF_RATING", "P1_TS_PCT", "P1_AST_PCT", "P1_PIE",
    "P2_OFF_RATING", "P2_DEF_RATING", "P2_TS_PCT", "P2_AST_PCT", "P2_PIE",
    "P3_OFF_RATING", "P3_DEF_RATING", "P3_TS_PCT", "P3_AST_PCT", "P3_PIE",
    "P4_OFF_RATING", "P4_DEF_RATING", "P4_TS_PCT", "P4_AST_PCT", "P4_PIE",
    "P5_OFF_RATING", "P5_DEF_RATING", "P5_TS_PCT", "P5_AST_PCT", "P5_PIE",
]

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/lineups_featured.parquet")
    return df

@st.cache_resource
def load_model(df):
    model = joblib.load("data/processed/best_model.pkl")
    feature_names = joblib.load("data/processed/feature_names.pkl")
    train = df[df["SEASON"] != "2023-24"].copy()
    X_train = train[feature_names].fillna(0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return model, scaler, feature_names

def predict_lineup(row, model, scaler, feature_names):
    X = row[feature_names].fillna(0).values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)[0]

df = load_data()
model, scaler, feature_names = load_model(df)

st.title("NBA Lineup Net Rating Predictor")
st.markdown("Predict how any 5-player lineup will perform based on individual player stats and lineup chemistry.")

tab1, tab2, tab3 = st.tabs(["Top Lineups", "Team Explorer", "Synergy Leaders"])

with tab1:
    st.subheader("Top Predicted Lineups")
    season_filter = st.selectbox("Season", sorted(df["SEASON"].unique(), reverse=True))
    min_minutes = st.slider("Minimum minutes played", 20, 200, 50)

    filtered = df[(df["SEASON"] == season_filter) & (df["MIN"] >= min_minutes)].copy()

    display_cols = ["GROUP_NAME", "MIN", "NET_RATING", "OFF_RATING",
                    "DEF_RATING", "SYNERGY_SCORE"]
    top = filtered.nlargest(20, "NET_RATING")[display_cols].reset_index(drop=True)
    top.columns = ["Lineup", "Minutes", "Net Rating", "Off Rating",
                   "Def Rating", "Synergy Score"]
    top["Net Rating"] = top["Net Rating"].round(1)
    top["Synergy Score"] = top["Synergy Score"].round(2)
    st.dataframe(top, width='stretch')

with tab2:
    st.subheader("Team Lineup Explorer")
    season2 = st.selectbox("Season ", sorted(df["SEASON"].unique(), reverse=True))
    teams = sorted(df["TEAM_ABBREVIATION"].unique())
    team = st.selectbox("Team", teams)

    team_df = df[(df["SEASON"] == season2) &
                 (df["TEAM_ABBREVIATION"] == team)].copy()
    team_df = team_df.sort_values("NET_RATING", ascending=False)

    display = team_df[["GROUP_NAME", "MIN", "NET_RATING",
                        "SYNERGY_SCORE"]].reset_index(drop=True)
    display.columns = ["Lineup", "Minutes", "Net Rating", "Synergy Score"]
    display["Net Rating"] = display["Net Rating"].round(1)
    display["Synergy Score"] = display["Synergy Score"].round(2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Lineups", len(team_df))
    col2.metric("Best Net Rating", f"{team_df['NET_RATING'].max():.1f}")
    col3.metric("Avg Synergy Score", f"{team_df['SYNERGY_SCORE'].mean():.2f}")

    st.dataframe(display, width='stretch')

with tab3:
    st.subheader("Synergy Leaders — Lineups Outperforming Individual Talent")
    season3 = st.selectbox("Season  ", sorted(df["SEASON"].unique(), reverse=True))
    min_min3 = st.slider("Minimum minutes ", 20, 200, 40)

    syn_df = df[(df["SEASON"] == season3) &
                (df["MIN"] >= min_min3)].copy()
    syn_df = syn_df.sort_values("SYNERGY_SCORE", ascending=False)

    top_syn = syn_df[["GROUP_NAME", "TEAM_ABBREVIATION", "MIN",
                       "NET_RATING", "AVG_PLAYER_NET_RATING",
                       "SYNERGY_SCORE"]].head(20).reset_index(drop=True)
    top_syn.columns = ["Lineup", "Team", "Minutes", "Actual Net Rating",
                        "Expected Net Rating", "Synergy Score"]
    for col in ["Actual Net Rating", "Expected Net Rating", "Synergy Score"]:
        top_syn[col] = top_syn[col].round(2)

    st.dataframe(top_syn, width='stretch')

    fig, ax = plt.subplots(figsize=(10, 5))
    top10 = syn_df.head(10)
    names = [g[:40] for g in top10["GROUP_NAME"]]
    ax.barh(names, top10["SYNERGY_SCORE"].values, color="#1d9e75")
    ax.set_xlabel("Synergy Score")
    ax.set_title(f"Top 10 Synergy Lineups — {season3}")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)