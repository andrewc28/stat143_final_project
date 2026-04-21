import pandas as pd
import numpy as np

DATA = "data/"
SEASON = 2025

# Load raw files
games = pd.read_csv(DATA + "MRegularSeasonDetailedResults.csv")
confs = pd.read_csv(DATA + "MTeamConferences.csv")
teams = pd.read_csv(DATA + "MTeams.csv")
seeds = pd.read_csv(DATA + "MNCAATourneySeeds.csv")
tourney = pd.read_csv(DATA + "MNCAATourneyDetailedResults.csv")

# Filter to current season
games = games[games["Season"] == SEASON].copy()
confs = confs[confs["Season"] == SEASON].copy()
tourney = tourney[tourney["Season"] == SEASON].copy()
seeds = seeds[seeds["Season"] == SEASON].copy()

#   Build team index  
# Only keep teams that appear in this season's games or tournament
team_ids = pd.unique(
    pd.concat([games["WTeamID"], games["LTeamID"],
               tourney["WTeamID"], tourney["LTeamID"]])
)
team_ids = np.sort(team_ids)
team_to_idx = {tid: i for i, tid in enumerate(team_ids)}
T = len(team_ids)

#   Build conference index  
# Some teams may be missing from MTeamConferences; fill with "Unknown"
conf_series = confs.set_index("TeamID")["ConfAbbrev"]
conf_labels = []
for tid in team_ids:
    conf_labels.append(conf_series.get(tid, "Unknown"))

unique_confs = sorted(set(conf_labels))
conf_to_idx = {c: i for i, c in enumerate(unique_confs)}
L = len(unique_confs)

team_conf_idx = np.array([conf_to_idx[c] for c in conf_labels])  # shape (T,)

#   Convert raw game rows to (home, away, home_score, away_score, neutral)  
# WLoc: H = winner was home, A = winner was away, N = neutral
def parse_games(df):
    rows = []
    for _, r in df.iterrows():
        wid, lid = r["WTeamID"], r["LTeamID"]
        ws, ls = r["WScore"], r["LScore"]
        loc = r.get("WLoc", "N")

        if loc == "H":
            home_id, away_id = wid, lid
            home_score, away_score = ws, ls
            neutral = False
        elif loc == "A":
            home_id, away_id = lid, wid
            home_score, away_score = ls, ws
            neutral = False
        else:  # N
            home_id, away_id = wid, lid
            home_score, away_score = ws, ls
            neutral = True

        if home_id not in team_to_idx or away_id not in team_to_idx:
            continue

        rows.append({
            "home_team": team_to_idx[home_id],
            "away_team": team_to_idx[away_id],
            "home_conf": team_conf_idx[team_to_idx[home_id]],
            "away_conf": team_conf_idx[team_to_idx[away_id]],
            "home_score": int(home_score),
            "away_score": int(away_score),
            "neutral": neutral,
        })
    return pd.DataFrame(rows)

train_df = parse_games(games)
tourney_df = parse_games(tourney)

#   Tournament teams and bracket pairs  
# Seeds file gives us which teams are in the 2025 tournament
tourney_team_ids = seeds["TeamID"].values
tourney_team_idxs = np.array([team_to_idx[tid] for tid in tourney_team_ids
                               if tid in team_to_idx])

print(f"Teams:       {T}")
print(f"Conferences: {L}")
print(f"Train games: {len(train_df)}")
print(f"Tourney games available: {len(tourney_df)}")
print(f"Tourney teams in bracket: {len(tourney_team_idxs)}")
print()
print("Train sample:")
print(train_df.head())

# Save processed data
train_df.to_csv(DATA + "train_games.csv", index=False)
tourney_df.to_csv(DATA + "tourney_games.csv", index=False)
np.save(DATA + "team_ids.npy", team_ids)
np.save(DATA + "team_conf_idx.npy", team_conf_idx)

# Save mappings for later use
import json
with open(DATA + "team_to_idx.json", "w") as f:
    json.dump({int(k): v for k, v in team_to_idx.items()}, f)
with open(DATA + "conf_to_idx.json", "w") as f:
    json.dump(conf_to_idx, f)

print("\nSaved: train_games.csv, tourney_games.csv, team_ids.npy, team_conf_idx.npy")
