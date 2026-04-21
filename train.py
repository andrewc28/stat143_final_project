import numpy as np
import pandas as pd
import json
from model import PoissonFactorizationVI

DATA = "data/"
N_RUNS = 100
K1, K2 = 10, 10

#   Load data  
train = pd.read_csv(DATA + "train_games.csv")
seeds = pd.read_csv(DATA + "MNCAATourneySeeds.csv")
seeds = seeds[seeds["Season"] == 2025]

with open(DATA + "team_to_idx.json") as f:
    team_to_idx = {int(k): v for k, v in json.load(f).items()}
with open(DATA + "conf_to_idx.json") as f:
    conf_to_idx = json.load(f)

team_conf = np.load(DATA + "team_conf_idx.npy")
team_ids  = np.load(DATA + "team_ids.npy")
T = len(team_ids)
L = len(conf_to_idx)

home    = train["home_team"].values
away    = train["away_team"].values
h_conf  = train["home_conf"].values
a_conf  = train["away_conf"].values
ys_H    = train["home_score"].values.astype(float)
ys_A    = train["away_score"].values.astype(float)
neutral = train["neutral"].values

#   Tournament teams  
tourney_tids = seeds["TeamID"].values
tourney_idxs = np.array([team_to_idx[t] for t in tourney_tids if t in team_to_idx])
tourney_confs = team_conf[tourney_idxs]

# All 68x67 ordered pairs for full bracket prediction
pairs = [(i, j) for i in range(len(tourney_idxs))
                for j in range(len(tourney_idxs)) if i != j]
p_home = np.array([tourney_idxs[i] for i, j in pairs])
p_away = np.array([tourney_idxs[j] for i, j in pairs])
p_hc   = team_conf[p_home]
p_ac   = team_conf[p_away]

#   Run VI N_RUNS times and average predictions  
all_probs = np.zeros(len(pairs))

for run in range(N_RUNS):
    print(f"Run {run+1}/{N_RUNS}")
    model = PoissonFactorizationVI(T=T, L=L, K1=K1, K2=K2)
    model.fit(home, away, h_conf, a_conf, ys_H, ys_A, neutral,
              max_iter=3_000, tol=1e-6, seed=run, verbose=(run == 0))

    probs, _, _ = model.predict_proba(p_home, p_away, p_hc, p_ac)
    all_probs += probs

all_probs /= N_RUNS

#   Build results dataframe  
idx_to_team = {v: k for k, v in team_to_idx.items()}
teams_df = pd.read_csv(DATA + "MTeams.csv").set_index("TeamID")

rows = []
for (i, j), prob in zip(pairs, all_probs):
    tid_h = idx_to_team[p_home[pairs.index((i,j))]]
    tid_a = idx_to_team[p_away[pairs.index((i,j))]]
    rows.append({
        "team1_id":   tid_h,
        "team1_name": teams_df.loc[tid_h, "TeamName"] if tid_h in teams_df.index else tid_h,
        "team2_id":   tid_a,
        "team2_name": teams_df.loc[tid_a, "TeamName"] if tid_a in teams_df.index else tid_a,
        "p_team1_wins": round(prob, 4),
    })

results = pd.DataFrame(rows)
results.to_csv(DATA + "predictions.csv", index=False)
print("\nSaved predictions.csv")
print(results.head(10))
