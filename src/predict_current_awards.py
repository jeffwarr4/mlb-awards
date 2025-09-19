# src/predict_current_awards.py
import warnings
warnings.filterwarnings("ignore")

import os
import time
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from requests.exceptions import HTTPError
from pybaseball import (
    batting_stats,
    pitching_stats,
    standings,
    team_batting,
    team_pitching,
)

CURRENT_YEAR = 2025

# -----------------------------
# Helpers & constants
# -----------------------------
TEAM_LEAGUE_2025: Dict[str, str] = {
    # AL
    "NYY":"AL","BOS":"AL","TOR":"AL","BAL":"AL","TBR":"AL",
    "CLE":"AL","MIN":"AL","DET":"AL","KCR":"AL","CHW":"AL",
    "HOU":"AL","SEA":"AL","OAK":"AL","TEX":"AL","LAA":"AL",
    # NL
    "LAD":"NL","SFG":"NL","SDP":"NL","ARI":"NL","COL":"NL",
    "CHC":"NL","STL":"NL","CIN":"NL","MIL":"NL","PIT":"NL",
    "ATL":"NL","NYM":"NL","PHI":"NL","WSN":"NL","MIA":"NL",
}
TEAM_ABBR_CANON = {
    "WSH":"WSN", "TBR":"TBR", "TBD":"TBR", "KCA":"KCR", "ANA":"LAA",
    "CHC":"CHC", "CHW":"CHW", "NYY":"NYY", "NYM":"NYM", "SF":"SFG", "SD":"SDP",
    "TB":"TBR", "KC":"KCR", "WSN":"WSN", "LAA":"LAA", "LAD":"LAD",
    "HOU":"HOU","SEA":"SEA","OAK":"OAK","TEX":"TEX",
    "ATL":"ATL","PHI":"PHI","MIA":"MIA","BAL":"BAL","BOS":"BOS","CLE":"CLE","DET":"DET","MIN":"MIN","TOR":"TOR",
    "STL":"STL","CIN":"CIN","MIL":"MIL","PIT":"PIT","ARI":"ARI","COL":"COL","SFG":"SFG","SDP":"SDP"
}

FG_MAX_RETRIES = 4
FG_SLEEP = 8  # seconds between retries

def baseball_ip_to_outs(ip: float) -> int:
    """Convert baseball IP (e.g., 123.2) to outs (integer)."""
    if pd.isna(ip):
        return 0
    whole = int(np.floor(ip))
    frac = ip - whole
    # 0.0 -> 0 outs, 0.1 -> 1 out, 0.2 -> 2 outs
    frac_outs = int(round(frac * 10))
    return whole * 3 + frac_outs

def safe_fill(df: pd.DataFrame, cols: List[str], val=0):
    present = [c for c in cols if c in df.columns]
    if present:
        df[present] = df[present].fillna(val)

def ensure_columns(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c not in df.columns:
            df[c] = 0

def _normalize_team(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].map(lambda x: TEAM_ABBR_CANON.get(str(x).strip(), str(x).strip()))
    return df

def _retry_fangraphs(fn, *args, **kwargs):
    last = None
    for i in range(FG_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except HTTPError as e:
            last = e
            code = getattr(e.response, "status_code", None)
            print(f"[FG] HTTP {code}; retry {i+1}/{FG_MAX_RETRIES} after {FG_SLEEP}s")
            time.sleep(FG_SLEEP)
        except Exception as e:
            last = e
            print(f"[FG] Error {e}; retry {i+1}/{FG_MAX_RETRIES} after {FG_SLEEP}s")
            time.sleep(FG_SLEEP)
    raise last

def safe_batting_stats(year: int) -> pd.DataFrame:
    """Batting with retries; returns empty DF on persistent failure."""
    try:
        return _retry_fangraphs(batting_stats, year)
    except Exception as e:
        print(f"[FG] batting_stats blocked for {year}: {e}. Using empty frame.")
        return pd.DataFrame()

def safe_pitching_stats(year: int) -> pd.DataFrame:
    """Pitching with retries; returns empty DF on persistent failure."""
    try:
        return _retry_fangraphs(pitching_stats, year)
    except Exception as e:
        print(f"[FG] pitching_stats blocked for {year}: {e}. Using empty frame.")
        return pd.DataFrame()

def compute_team_winpct(year: int) -> pd.DataFrame:
    """
    Return DataFrame with columns ['Team','WinPct'] for the given year.
    Tries pybaseball.standings() first (handles list/df), then FG team leaderboards.
    Falls back to 0.500 if everything fails.
    """
    # --- Try standings() ---
    try:
        st = standings(year)
        # standings can return a list of DataFrames (e.g., divisions or leagues)
        if isinstance(st, list):
            frames = []
            for part in st:
                if isinstance(part, pd.DataFrame) and not part.empty:
                    frames.append(part)
            if frames:
                st = pd.concat(frames, ignore_index=True)
            else:
                st = None

        if isinstance(st, pd.DataFrame) and not st.empty:
            cols = {c.lower(): c for c in st.columns}
            team_col = cols.get('tm') or cols.get('team')
            wp_col   = cols.get('w-l%') or cols.get('win%') or cols.get('w-l pct') or cols.get('w-l_pct')
            w_col    = cols.get('w')
            l_col    = cols.get('l')

            if team_col is None:
                raise ValueError("standings(): could not find team column")
            df = st[[team_col]].copy().rename(columns={team_col: 'Team'})

            # Prefer a direct winning% column if present
            if wp_col is not None:
                df['WinPct'] = pd.to_numeric(st[wp_col], errors='coerce')
            elif w_col is not None and l_col is not None:
                W = pd.to_numeric(st[w_col], errors='coerce')
                L = pd.to_numeric(st[l_col], errors='coerce')
                df['WinPct'] = W / (W + L)
            else:
                raise ValueError("standings(): missing W/L or W-L%")

            df = _normalize_team(df, 'Team').dropna()
            if not df.empty:
                return df[['Team','WinPct']]
    except Exception as e:
        print(f"⚠️ standings() WinPct failed for {year}: {e}")

    # --- Try FanGraphs team leaderboards (batting, then pitching) ---
    for pull in (team_batting, team_pitching):
        try:
            tb = pull(year)
            if isinstance(tb, pd.DataFrame) and not tb.empty:
                cols = {c.lower(): c for c in tb.columns}
                team_col = cols.get('team') or cols.get('tm')
                w_col    = cols.get('w')
                l_col    = cols.get('l')
                if team_col and w_col and l_col:
                    df = tb[[team_col, w_col, l_col]].copy()
                    df.rename(columns={team_col: 'Team', w_col: 'W', l_col: 'L'}, inplace=True)
                    df['W'] = pd.to_numeric(df['W'], errors='coerce')
                    df['L'] = pd.to_numeric(df['L'], errors='coerce')
                    df['WinPct'] = df['W'] / (df['W'] + df['L'])
                    df = _normalize_team(df, 'Team').dropna()
                    if not df.empty:
                        return df[['Team','WinPct']]
        except Exception as e:
            print(f"⚠️ {pull.__name__} WinPct failed for {year}: {e}")

    # --- Final fallback ---
    print("⚠️ Falling back to neutral WinPct=0.500 for all teams.")
    return pd.DataFrame({'Team': list(TEAM_LEAGUE_2025.keys()),
                         'WinPct': [0.5]*len(TEAM_LEAGUE_2025)})

# -----------------------------
# Feature builder
# -----------------------------
def build_current_features(year: int) -> pd.DataFrame:
    print(f"Downloading FanGraphs leaderboards for {year} ...")

    # Use safe wrappers with retries and soft-failure
    bat = safe_batting_stats(year)   # Season, Name, Team, IDfg, OPS, OBP, SLG, WAR, wRC+
    pit = safe_pitching_stats(year)  # Season, Name, Team, IDfg, IP, SO, BB, ER, HR, SV, W, L, WAR, FIP, K%, BB%, ERA-

    # If blocks persist, create empty shells with required columns so pipeline doesn't die
    if bat.empty:
        bat = pd.DataFrame(columns=["Name","Team","G","AB","H","HR","RBI","BB","SO","SB","CS","R","2B","3B","HBP","SF","WAR","wRC+","OPS","OBP","SLG"])
    if pit.empty:
        pit = pd.DataFrame(columns=["Name","Team","IP","G","GS","SO","BB","ER","HR","SV","W","L","H","WAR","FIP","K%","BB%","ERA-","xFIP"])

    # Normalize teams
    for df in (bat, pit):
        if 'Team' in df.columns:
            df['Team'] = df['Team'].map(lambda x: TEAM_ABBR_CANON.get(str(x).strip(), str(x).strip()))

    # Derive league from team code
    bat['lgID'] = bat['Team'].map(TEAM_LEAGUE_2025)
    pit['lgID'] = pit['Team'].map(TEAM_LEAGUE_2025)

    # --- Map/rename core batting columns to your training names ---
    batting_map = {
        'G':'G_bat','AB':'AB','H':'H','HR':'HR','RBI':'RBI','BB':'BB','SO':'SO',
        'SB':'SB','CS':'CS','R':'R','2B':'2B','3B':'3B','HBP':'HBP','SF':'SF'
    }
    for src in batting_map.keys():
        if src not in bat.columns:
            bat[src] = 0
    bat.rename(columns=batting_map, inplace=True)

    # FanGraphs advanced batting
    if 'WAR' in bat.columns:   bat['bat_WAR_fg']   = pd.to_numeric(bat['WAR'], errors='coerce')
    if 'wRC+' in bat.columns:  bat['bat_wRC_plus'] = pd.to_numeric(bat['wRC+'], errors='coerce')
    if 'OPS' in bat.columns:   bat['bat_OPS']      = pd.to_numeric(bat['OPS'], errors='coerce')
    if 'OBP' in bat.columns:   bat['bat_OBP']      = pd.to_numeric(bat['OBP'], errors='coerce')
    if 'SLG' in bat.columns:   bat['bat_SLG']      = pd.to_numeric(bat['SLG'], errors='coerce')

    # --- Map/rename core pitching columns to your training names ---
    if 'IP' not in pit.columns:
        pit['IP'] = 0.0
    pit['IP'] = pd.to_numeric(pit['IP'], errors='coerce').fillna(0.0)
    pit['IPouts'] = pit['IP'].apply(baseball_ip_to_outs)

    pitching_map = {
        'G':'G_pit','GS':'GS','SO':'SO_pit','BB':'BB_pit','ER':'ER','HR':'HR_pit',
        'SV':'SV','W':'W_pit','L':'L_pit','H':'H_pit'
    }
    for src in pitching_map.keys():
        if src not in pit.columns:
            pit[src] = 0
    pit.rename(columns=pitching_map, inplace=True)

    # FanGraphs advanced pitching
    if 'WAR' in pit.columns:   pit['pit_WAR_fg']   = pd.to_numeric(pit['WAR'], errors='coerce')
    if 'FIP' in pit.columns:   pit['pit_FIP']      = pd.to_numeric(pit['FIP'], errors='coerce')
    if 'K%' in pit.columns:    pit['pit_Kpct']     = pd.to_numeric(pit['K%'], errors='coerce')
    if 'BB%' in pit.columns:   pit['pit_BBpct']    = pd.to_numeric(pit['BB%'], errors='coerce')
    if 'ERA-' in pit.columns:  pit['pit_ERA_minus']= pd.to_numeric(pit['ERA-'], errors='coerce')
    if 'xFIP' in pit.columns:  pit['pit_xFIP']     = pd.to_numeric(pit['xFIP'], errors='coerce')

    # --- Reduce to necessary columns & combine batting/pitching rows ---
    bat_keep = ['Name','Team','lgID','G_bat','AB','H','HR','RBI','BB','SO','SB','CS','R','2B','3B','HBP','SF',
                'bat_WAR_fg','bat_wRC_plus','bat_OPS','bat_OBP','bat_SLG']
    bat_keep = [c for c in bat_keep if c in bat.columns]

    pit_keep = ['Name','Team','lgID','IP','IPouts','G_pit','GS','SO_pit','BB_pit','ER','HR_pit','SV','W_pit','L_pit','H_pit',
                'pit_WAR_fg','pit_FIP','pit_Kpct','pit_BBpct','pit_ERA_minus','pit_xFIP']
    pit_keep = [c for c in pit_keep if c in pit.columns]

    bat_cur = bat[bat_keep].copy()
    pit_cur = pit[pit_keep].copy()

    # Outer-merge on (Name, Team, lgID)
    cur = pd.merge(bat_cur, pit_cur, on=['Name','Team','lgID'], how='outer')

    # Attach team WinPct (robust)
    wp = compute_team_winpct(year)  # ['Team','WinPct']
    cur = cur.merge(wp, on='Team', how='left')
    cur['WinPct'] = pd.to_numeric(cur['WinPct'], errors='coerce').fillna(0.5)

    # Fielding placeholders (your model expects them)
    ensure_columns(cur, ['G_fld','PO','A','E','DP','FieldPct'])

    # Fill numeric NaNs
    num_cols = cur.select_dtypes(include=[np.number]).columns
    cur[num_cols] = cur[num_cols].replace([np.inf,-np.inf], np.nan).fillna(0)

    # If lgID missing for odd team codes, set to None so groupby doesn't break
    if 'lgID' not in cur.columns:
        cur['lgID'] = None

    return cur

# -----------------------------
# Model utilities
# -----------------------------
def load_model_and_features(task_dir: Path):
    model_path_lr = task_dir / "model_logreg.joblib"
    model_path_rf = task_dir / "model_randomforest.joblib"
    if model_path_lr.exists():
        model = joblib.load(model_path_lr)
    else:
        model = joblib.load(model_path_rf)
    feat_cols = joblib.load(task_dir / "feature_columns.joblib")
    return model, feat_cols

def score_and_rank(df_cur: pd.DataFrame, model, feature_cols: List[str], prob_col: str) -> pd.DataFrame:
    X = df_cur.copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols].copy()
    X = X.replace([np.inf,-np.inf], np.nan).fillna(0)

    probs = model.predict_proba(X)[:,1]
    out = df_cur.copy()
    out[prob_col] = probs
    return out

def top5_per_league(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    keep = ['lgID','Team','Name',prob_col,'bat_WAR_fg','bat_wRC_plus','bat_OPS','bat_OBP','bat_SLG',
            'RBI','HR','AB','WinPct','pit_WAR_fg','pit_FIP','pit_Kpct','IPouts','SV','SO_pit','ER']
    keep = [c for c in keep if c in df.columns]
    view = df[keep].copy()
    if 'lgID' not in view.columns:
        view['lgID'] = None
    # rank within each league
    return (view.groupby('lgID', group_keys=False)
                .apply(lambda g: g.sort_values(prob_col, ascending=False).head(5))
                .reset_index(drop=True))

# -----------------------------
# Public API (for GitHub Action)
# -----------------------------
def main(YEAR: int, outdir: Path, timestamp: Optional[str] = None) -> Tuple[str, str]:
    # 1) Build current-season features (robust to 403s)
    current = build_current_features(YEAR)

    # 2) Load models & their feature lists
    mvp_model, mvp_feats = load_model_and_features(Path("models/MVP_top5"))
    cy_model,  cy_feats  = load_model_and_features(Path("models/CY_top5"))

    # 3) Score
    scored_mvp = score_and_rank(current, mvp_model, mvp_feats, prob_col="MVP_prob")
    scored_cy  = score_and_rank(current, cy_model,  cy_feats,  prob_col="CY_prob")

    # 4) Top 5 per league
    top5_mvp = top5_per_league(scored_mvp, "MVP_prob")
    top5_cy  = top5_per_league(scored_cy,  "CY_prob")

    # 5) Save & return paths
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    mvp_csv = outdir / f"top5_mvp_{YEAR}_{ts}.csv"
    cy_csv  = outdir / f"top5_cy_{YEAR}_{ts}.csv"
    top5_mvp.to_csv(mvp_csv, index=False)
    top5_cy.to_csv(cy_csv, index=False)

    print("\n=== Top 5 MVP per league ===")
    print(top5_mvp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Top 5 Cy Young per league ===")
    print(top5_cy.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n✅ Saved CSVs to {outdir.resolve()}")
    return str(mvp_csv), str(cy_csv)

# -----------------------------
# CLI entry (still supported)
# -----------------------------
if __name__ == "__main__":
    # When called directly, behave like your original script
    outdir = Path("predictions") / f"{CURRENT_YEAR}"
    outdir.mkdir(parents=True, exist_ok=True)
    main(CURRENT_YEAR, outdir)
