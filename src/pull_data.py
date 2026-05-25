import time
import pandas as pd
from nba_api.stats.endpoints import LeagueDashLineups, LeagueDashPlayerStats

SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24"
]

def pull_lineup_data(season):
    print(f"Pulling lineup data for {season}...")
    lineups = LeagueDashLineups(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Per100Possessions"
    )
    df = lineups.get_data_frames()[0]
    df["SEASON"] = season
    return df

def pull_player_data(season):
    print(f"Pulling player data for {season}...")
    players = LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Per100Possessions"
    )
    df = players.get_data_frames()[0]
    df["SEASON"] = season
    return df

def main():
    all_lineups = []
    all_players = []

    for season in SEASONS:
        try:
            lineup_df = pull_lineup_data(season)
            all_lineups.append(lineup_df)
            time.sleep(1)

            player_df = pull_player_data(season)
            all_players.append(player_df)
            time.sleep(1)

        except Exception as e:
            print(f"Error pulling {season}: {e}")
            continue

    if not all_lineups or not all_players:
        print("No data pulled. Check errors above.")
        return

    lineups_combined = pd.concat(all_lineups, ignore_index=True)
    players_combined = pd.concat(all_players, ignore_index=True)

    lineups_combined.to_parquet("data/raw/lineups_raw.parquet", index=False)
    players_combined.to_parquet("data/raw/players_raw.parquet", index=False)

    print(f"Done! Lineups shape: {lineups_combined.shape}")
    print(f"Done! Players shape: {players_combined.shape}")

if __name__ == "__main__":
    main()