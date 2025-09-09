# file: enrich_with_fangraphs.py
import pandas as pd
import numpy as np
from pybaseball import batting_stats, pitching_stats, playerid_reverse_lookup

START_YEAR = 1980
END_YEAR   = 2024  # adjust if needed

# 1) Load your merged base (from your earlier pipeline)
base = pd.read_csv("player_season_full_1980_present.csv")
base["yearID"] = base["yearID"].astype(int)

# 2) Map Lahman/BBRef IDs -> FanGraphs IDs
#    Lahman playerID matches bbref-style ids like "troutmi01".
all_ids = base["playerID"].dropna().unique().tolist()

# do lookup in chunks to be gentle
def map_bbref_to_fg(ids, chunk=500):
    frames = []
    for i in range(0, len(ids), chunk):
        sub = ids[i:i+chunk]
        m = playerid_reverse_lookup(sub, key_type="bbref")
        frames.append(m)
    return pd.concat(frames, ignore_index=True)

id_map = map_bbref_to_fg(all_ids)
# columns typically: key_bbref, key_fangraphs, key_mlbam, name_first, name_last, ...
id_map = id_map[["key_bbref", "key_fangraphs"]].dropna().drop_duplicates()
id_map.rename(columns={"key_bbref":"playerID", "key_fangraphs":"IDfg"}, inplace=True)

# attach FanGraphs ID to base
enriched = base.merge(id_map, on="playerID", how="left")

# 3) Pull FanGraphs batting & pitching (1980–END_YEAR)
print("Downloading FanGraphs batting leaderboard…")
bat_fg = batting_stats(START_YEAR, END_YEAR)  # includes Season, IDfg
# keep only what we'll use (guard for column presence)
bat_keep = ["Season","IDfg"]
for col in ["WAR","wRC+","OPS","OBP","SLG"]:
    if col in bat_fg.columns:
        bat_keep.append(col)
bat_fg = bat_fg[bat_keep].copy()
bat_fg.rename(columns={
    "Season":"yearID",
    "WAR":"bat_WAR_fg",
    "wRC+":"bat_wRC_plus",
    "OPS":"bat_OPS",
    "OBP":"bat_OBP",
    "SLG":"bat_SLG",
}, inplace=True)
bat_fg["yearID"] = bat_fg["yearID"].astype(int)

print("Downloading FanGraphs pitching leaderboard…")
pit_fg = pitching_stats(START_YEAR, END_YEAR)
pit_keep = ["Season","IDfg"]
for col in ["WAR","FIP","K%","BB%","ERA-","xFIP"]:
    if col in pit_fg.columns:
        pit_keep.append(col)
pit_fg = pit_fg[pit_keep].copy()
pit_fg.rename(columns={
    "Season":"yearID",
    "WAR":"pit_WAR_fg",
    "FIP":"pit_FIP",
    "K%":"pit_Kpct",
    "BB%":"pit_BBpct",
    "ERA-":"pit_ERA_minus",
    "xFIP":"pit_xFIP",
}, inplace=True)
pit_fg["yearID"] = pit_fg["yearID"].astype(int)

# 4) Merge FanGraphs features into your base by (IDfg, year)
enriched = enriched.merge(bat_fg, on=["IDfg","yearID"], how="left")
enriched = enriched.merge(pit_fg, on=["IDfg","yearID"], how="left")

# 5) Optional: derive OPS+ proxy later; for now we rely on wRC+ (better than OPS+ for offense)
# fill NaNs for new numeric features only
fg_num = ["bat_WAR_fg","bat_wRC_plus","bat_OPS","bat_OBP","bat_SLG",
          "pit_WAR_fg","pit_FIP","pit_Kpct","pit_BBpct","pit_ERA_minus","pit_xFIP"]
fg_num = [c for c in fg_num if c in enriched.columns]
enriched[fg_num] = enriched[fg_num].astype(float)

# 6) Write a lean features file for training
# load your previous lean selection if you have it; otherwise build a new one
keep_core = [
    "playerID","yearID","teamID","lgID","WinPct",
    # batting counts you already had (if present)
    "G_bat","AB","H","HR","RBI","BB","SO","SB","CS","R","2B","3B","HBP","SF",
    # pitching you already had (if present)
    "G_pit","GS","IPouts","SO_pit","BB_pit","ER","HR_pit","SV","W_pit","L_pit",
    # fielding
    "G_fld","PO","A","E","DP","FieldPct",
]
# add FG features that exist
keep_fg = [c for c in [
    "bat_WAR_fg","bat_wRC_plus","bat_OPS","bat_OBP","bat_SLG",
    "pit_WAR_fg","pit_FIP","pit_Kpct","pit_BBpct","pit_ERA_minus","pit_xFIP"
] if c in enriched.columns]

# award labels (from your earlier pipeline)
labels = [c for c in enriched.columns if c in
          ["is_top5_MVP","is_top5_CY",
           "voteShare_Most_Valuable_Player","voteShare_Cy_Young_Award"]]

cols = [c for c in keep_core + keep_fg + labels if c in enriched.columns]
final = enriched[cols].copy()

final.to_csv("player_season_features_with_fg_1980_present.csv", index=False)
print("✅ Wrote: player_season_features_with_fg_1980_present.csv")
