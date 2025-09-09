import zipfile
import pandas as pd
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
ZIP_PATH = r"C:\Users\jeffw\MLB MVP Prediction Project\Lahman_1871-2024_csv.zip"
PREFIX   = "lahman_1871-2024_csv/"   # adjust if your folder name inside the zip differs
START_YEAR = 1980

# -----------------------------
# LOAD
# -----------------------------
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    batting  = pd.read_csv(z.open(PREFIX + "Batting.csv"))
    pitching = pd.read_csv(z.open(PREFIX + "Pitching.csv"))
    fielding = pd.read_csv(z.open(PREFIX + "Fielding.csv"))
    teams    = pd.read_csv(z.open(PREFIX + "Teams.csv"))
    awards   = pd.read_csv(z.open(PREFIX + "AwardsSharePlayers.csv"))

# -----------------------------
# FILTER YEARS
# -----------------------------
batting  = batting[batting['yearID']  >= START_YEAR].copy()
pitching = pitching[pitching['yearID'] >= START_YEAR].copy()
fielding = fielding[fielding['yearID'] >= START_YEAR].copy()
teams    = teams[teams['yearID']      >= START_YEAR].copy()
awards   = awards[awards['yearID']    >= START_YEAR].copy()

# -----------------------------
# TEAM WIN PCT
# -----------------------------
teams['WinPct'] = teams['W'] / (teams['W'] + teams['L'])
teams_simple = teams[['yearID','teamID','lgID','WinPct']].copy()

# -----------------------------
# PRIMARY TEAM PER PLAYER-YEAR
# (so we can attach team WinPct to a single team)
# -----------------------------
# Hitters: use At Bats as weight; fallback to Games
bat_ab = batting.groupby(['playerID','yearID','teamID'], as_index=False).agg({'AB':'sum','G':'sum'})
bat_ab['bat_weight'] = bat_ab['AB'].fillna(0)
bat_ab.loc[bat_ab['bat_weight']==0, 'bat_weight'] = bat_ab.loc[bat_ab['bat_weight']==0, 'G']

# Pitchers: use Outs Pitched (IPouts) as weight; fallback to Games Pitched (G)
pit_ip = pitching.groupby(['playerID','yearID','teamID'], as_index=False).agg({'IPouts':'sum','G':'sum'})
pit_ip['pit_weight'] = pit_ip['IPouts'].fillna(0)
pit_ip.loc[pit_ip['pit_weight']==0, 'pit_weight'] = pit_ip.loc[pit_ip['pit_weight']==0, 'G']

# Choose primary team by max weight per player-year
bat_primary = bat_ab.sort_values(['playerID','yearID','bat_weight'], ascending=[True,True,False]) \
                    .drop_duplicates(['playerID','yearID'])[['playerID','yearID','teamID']] \
                    .rename(columns={'teamID':'bat_teamID'})
pit_primary = pit_ip.sort_values(['playerID','yearID','pit_weight'], ascending=[True,True,False]) \
                    .drop_duplicates(['playerID','yearID'])[['playerID','yearID','teamID']] \
                    .rename(columns={'teamID':'pit_teamID'})

# Merge to decide one primary team: prefer hitter team if available, else pitcher team
primary_team = pd.merge(bat_primary, pit_primary, on=['playerID','yearID'], how='outer')
primary_team['teamID'] = primary_team['bat_teamID'].fillna(primary_team['pit_teamID'])
primary_team = primary_team[['playerID','yearID','teamID']].dropna(subset=['teamID'])

# Attach league for ranking convenience (from Teams)
primary_team = primary_team.merge(teams_simple[['yearID','teamID','lgID']], on=['yearID','teamID'], how='left')

# -----------------------------
# AGGREGATE PLAYER STATS TO PLAYER-YEAR
# -----------------------------
bat_agg = batting.groupby(['playerID','yearID'], as_index=False).sum(numeric_only=True)
pit_agg = pitching.groupby(['playerID','yearID'], as_index=False).sum(numeric_only=True)
fld_agg = fielding.groupby(['playerID','yearID'], as_index=False).agg(
    G_fld=('G','sum'), PO=('PO','sum'), A=('A','sum'), E=('E','sum'), DP=('DP','sum')
)
fld_agg['FieldPct'] = (fld_agg['PO'] + fld_agg['A']) / ((fld_agg['PO'] + fld_agg['A'] + fld_agg['E']).replace(0, pd.NA))

# Combine batting + pitching + fielding
player_stats = bat_agg.merge(pit_agg, on=['playerID','yearID'], how='outer', suffixes=('_bat','_pit'))
player_stats = player_stats.merge(fld_agg, on=['playerID','yearID'], how='left')

# Attach primary team & WinPct
player_stats = player_stats.merge(primary_team[['playerID','yearID','teamID','lgID']], on=['playerID','yearID'], how='left')
player_stats = player_stats.merge(teams_simple[['yearID','teamID','WinPct']], on=['yearID','teamID'], how='left')

# -----------------------------
# AWARDS: build vote share, pivot to columns
# -----------------------------
awards = awards[awards['awardID'].isin(['Most Valuable Player','Cy Young Award'])].copy()
awards['voteShare'] = awards['pointsWon'] / awards['pointsMax']
awards_narrow = awards[['playerID','yearID','lgID','awardID','pointsWon','pointsMax','votesFirst','voteShare']].copy()

awards_pivot = awards_narrow.pivot_table(
    index=['playerID','yearID','lgID'],
    columns='awardID',
    values=['pointsWon','pointsMax','votesFirst','voteShare'],
    aggfunc='max',  # one row per player/award/year; max is safe
    fill_value=0
)
awards_pivot.columns = ['_'.join(col).replace(' ','_') for col in awards_pivot.columns]
awards_pivot = awards_pivot.reset_index()

# Merge awards into player stats
full = player_stats.merge(awards_pivot, on=['playerID','yearID','lgID'], how='left')

# -----------------------------
# LABELS: Top-5 per year/league for MVP & CY (based on voteShare)
# -----------------------------
def top5_flag(df, share_col):
    df = df.copy()
    df[share_col] = df[share_col].fillna(0)
    df['rank_tmp'] = df.groupby(['yearID','lgID'])[share_col].rank(ascending=False, method='first')
    return (df['rank_tmp'] <= 5).astype(int)

full['is_top5_MVP'] = top5_flag(full, 'voteShare_Most_Valuable_Player')
full['is_top5_CY']  = top5_flag(full, 'voteShare_Cy_Young_Award')
full.drop(columns=['rank_tmp'], errors='ignore', inplace=True)

# -----------------------------
# OPTIONAL: tidy up a lean feature set for modeling
# -----------------------------
keep_cols = [
    'playerID','yearID','teamID','lgID','WinPct',
    # batting (common useful ones if present)
    'G_bat','AB','H','HR','RBI','BB','SO','SB','CS','R','2B','3B','IBB','HBP','SH','SF',
    # pitching (common)
    'G_pit','GS','IPouts','SO_pit','BB_pit','ER','HR_pit','SV','W_pit','L_pit','H_pit','HBP_pit','WP','BK',
    # fielding
    'G_fld','PO','A','E','DP','FieldPct',
    # awards targets
    'pointsWon_Most_Valuable_Player','pointsMax_Most_Valuable_Player','voteShare_Most_Valuable_Player','votesFirst_Most_Valuable_Player',
    'pointsWon_Cy_Young_Award','pointsMax_Cy_Young_Award','voteShare_Cy_Young_Award','votesFirst_Cy_Young_Award',
    'is_top5_MVP','is_top5_CY'
]
# keep only columns that exist
keep_cols = [c for c in keep_cols if c in full.columns]
dataset = full[keep_cols].copy()

import numpy as np

# -----------------------------
# Add OPS (On-Base Plus Slugging)
# -----------------------------
full['1B'] = full['H_bat'] - full['2B'] - full['3B'] - full['HR_bat']

# OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
full['OBP'] = (full['H_bat'] + full['BB_bat'] + full['HBP_bat']) / (
    full['AB'] + full['BB_bat'] + full['HBP_bat'] + full['SF_bat']
).replace(0, np.nan)

# SLG = (1B + 2B*2 + 3B*3 + HR*4) / AB
full['SLG'] = (full['1B'] + 2*full['2B'] + 3*full['3B'] + 4*full['HR_bat']) / (
    full['AB'].replace(0, np.nan)
)

# OPS = OBP + SLG
full['OPS'] = full['OBP'] + full['SLG']

# Fill any NaN values with 0
full[['OBP','SLG','OPS']] = full[['OBP','SLG','OPS']].fillna(0)


# Save everything
full.to_csv("player_season_full_1980_present.csv", index=False)
dataset.to_csv("player_season_features_1980_present.csv", index=False)
print("✅ Saved:\n - player_season_full_1980_present.csv (all merged columns)\n - player_season_features_1980_present.csv (lean features)")
