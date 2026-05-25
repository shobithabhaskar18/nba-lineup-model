import pandas as pd
import numpy as np

def load_data():
    lineups = pd.read_parquet("data/raw/lineups_raw.parquet")
    players = pd.read_parquet("data/raw/players_raw.parquet")
    return lineups, players

def extract_players(lineups):
    print("Extracting player names from GROUP_NAME...")
    split = lineups["GROUP_NAME"].str.split(" - ", expand=True)
    split.columns = [f"PLAYER_{i+1}" for i in range(split.shape[1])]
    lineups = pd.concat([lineups, split], axis=1)
    return lineups

def filter_minutes(lineups, min_minutes=20):
    print(f"Filtering lineups with less than {min_minutes} minutes...")
    before = len(lineups)
    lineups = lineups[lineups["MIN"] >= min_minutes].reset_index(drop=True)
    after = len(lineups)
    print(f"Removed {before - after} rows. Remaining: {after}")
    return lineups

def build_name_map(players):
    print("Building abbreviated name lookup...")
    name_map = {}
    for full_name in players["PLAYER_NAME"].unique():
        parts = full_name.strip().split(" ")
        if len(parts) >= 2:
            abbreviated = parts[0][0] + ". " + " ".join(parts[1:])
            name_map[abbreviated] = full_name
    return name_map

def get_player_features(players):
    print("Building player feature lookup...")
    cols = [
        "PLAYER_NAME", "SEASON", "OFF_RATING", "DEF_RATING",
        "NET_RATING", "TS_PCT", "AST_PCT", "OREB_PCT",
        "DREB_PCT", "TM_TOV_PCT", "PIE"
    ]
    player_features = players[cols].copy()
    return player_features

def join_player_stats(lineups, player_features, name_map):
    print("Joining player stats to lineups...")
    for i in range(1, 6):
        player_col = f"PLAYER_{i}"
        lineups[player_col] = lineups[player_col].map(
            lambda x: name_map.get(x, x) if pd.notna(x) else x
        )
        renamed = player_features.rename(columns={
            "PLAYER_NAME": player_col,
            "OFF_RATING": f"P{i}_OFF_RATING",
            "DEF_RATING": f"P{i}_DEF_RATING",
            "NET_RATING": f"P{i}_NET_RATING",
            "TS_PCT": f"P{i}_TS_PCT",
            "AST_PCT": f"P{i}_AST_PCT",
            "OREB_PCT": f"P{i}_OREB_PCT",
            "DREB_PCT": f"P{i}_DREB_PCT",
            "TM_TOV_PCT": f"P{i}_TOV_PCT",
            "PIE": f"P{i}_PIE"
        })
        lineups = lineups.merge(
            renamed,
            on=[player_col, "SEASON"],
            how="left"
        )
    return lineups

def add_synergy_features(df):
    print("Adding synergy features...")
    player_net_cols = [f"P{i}_NET_RATING" for i in range(1, 6)]
    player_ts_cols = [f"P{i}_TS_PCT" for i in range(1, 6)]
    player_pie_cols = [f"P{i}_PIE" for i in range(1, 6)]

    df["AVG_PLAYER_NET_RATING"] = df[player_net_cols].mean(axis=1)
    df["AVG_PLAYER_TS_PCT"] = df[player_ts_cols].mean(axis=1)
    df["AVG_PLAYER_PIE"] = df[player_pie_cols].mean(axis=1)

    df["SYNERGY_SCORE"] = df["NET_RATING"] - df["AVG_PLAYER_NET_RATING"]

    return df

def main():
    lineups, players = load_data()

    lineups = extract_players(lineups)
    lineups = filter_minutes(lineups, min_minutes=20)

    name_map = build_name_map(players)
    player_features = get_player_features(players)
    lineups = join_player_stats(lineups, player_features, name_map)
    lineups = add_synergy_features(lineups)

    lineups.to_parquet("data/processed/lineups_featured.parquet", index=False)

    print(f"\nFinal dataset shape: {lineups.shape}")
    print(f"\nSample synergy scores:")
    print(lineups[["GROUP_NAME", "NET_RATING", "AVG_PLAYER_NET_RATING", "SYNERGY_SCORE"]].head(10))
    print(f"\nNull counts in key columns:")
    key_cols = ["NET_RATING", "AVG_PLAYER_NET_RATING", "SYNERGY_SCORE"]
    print(lineups[key_cols].isnull().sum())

if __name__ == "__main__":
    main()