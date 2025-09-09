# predict_current_awards.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List

from pybaseball import batting_stats, pitching_stats, standings, team_batting, team_pitching

CURRENT_YEAR = 2025

# -----------------------------
# Helpers
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
# Baseball-Reference/pybaseball sometimes use different 3-letter codes for Washington & Tampa Bay, etc.
TEAM_ABBR_CANON = {
    "WSH":"WSN", "TBR":"TBR", "TBD":"TBR", "KCA":"KCR", "ANA":"LAA",
    "CHC":"CHC", "CHW":"CHW", "NYY":"NYY", "NYM":"NYM", "SF":"SFG", "SD":"SDP",
    "TB":"TBR", "KC":"KCR", "WSN":"WSN", "LAA":"LAA", "LAD":"LAD",
    "HOU":"HOU","SEA":"SEA","OAK":"OAK","TEX":"TEX",
    "ATL":"ATL","PHI":"PHI","MIA":"MIA","BAL":"BAL","BOS":"BOS","CLE":"CLE","DET":"DET","MIN":"MIN","TOR":"TOR",
    "STL":"STL","CIN":"CIN","MIL":"MIL","PIT":"PIT","ARI":"ARI","COL":"COL","SFG":"SFG","SDP":"SDP"
}

def baseball_ip_to_outs(ip: float) -> int:
    """Convert baseball IP (e.g., 123.2) to outs (integer)."""
    if pd.isna(ip):
        return 0
    whole = int(np.floor(ip))
    frac = ip - whole
    # 0.0 -> 0, 0.1 -> 1 out, 0.2 -> 2 outs
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

def compute_team_winpct(year: int) -> pd.DataFrame:
    """
    Return DataFrame with columns ['Team','WinPct'] for the given year.
    Tries pybaseball.standings() first (handles list/df), then FG team leaderboards.
    Falls back to 0.500 if everything fails.
    """
    def _normalize_team(df, col):
        df[col] = df[col].map(lambda x: TEAM_ABBR_CANON.get(str(x).strip(), str(x).strip()))
        return df

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



def build_current_features(year: int) -> pd.DataFrame:
    print(f"Downloading FanGraphs leaderboards for {year} ...")
    bat = batting_stats(year)   # includes Season, Name, Team, IDfg, OPS, OBP, SLG, WAR, wRC+
    pit = pitching_stats(year)  # includes Season, Name, Team, IDfg, IP, SO, BB, ER, HR, SV, W, L, WAR, FIP, K%, BB%, ERA-

    # Minimal identity / join keys
    for df in (bat, pit):
        if 'Team' in df.columns:
            df['Team'] = df['Team'].map(lambda x: TEAM_ABBR_CANON.get(str(x).strip(), str(x).strip()))

    # Derive league from team code
    bat['lgID'] = bat['Team'].map(TEAM_LEAGUE_2025)
    pit['lgID'] = pit['Team'].map(TEAM_LEAGUE_2025)

    # --- Map/rename core batting columns to your training names ---
    # Ensure presence then rename
    batting_map = {
        'G':'G_bat','AB':'AB','H':'H','HR':'HR','RBI':'RBI','BB':'BB','SO':'SO',
        'SB':'SB','CS':'CS','R':'R','2B':'2B','3B':'3B','HBP':'HBP','SF':'SF'
    }
    for src, dst in batting_map.items():
        if src not in bat.columns:
            bat[src] = 0
    bat.rename(columns=batting_map, inplace=True)

    # FanGraphs advanced batting
    adv_bat = {}
    if 'WAR' in bat.columns:   adv_bat['bat_WAR_fg']   = bat['WAR']
    if 'wRC+' in bat.columns:  adv_bat['bat_wRC_plus'] = bat['wRC+']
    if 'OPS' in bat.columns:   adv_bat['bat_OPS']      = bat['OPS']
    if 'OBP' in bat.columns:   adv_bat['bat_OBP']      = bat['OBP']
    if 'SLG' in bat.columns:   adv_bat['bat_SLG']      = bat['SLG']
    for k,v in adv_bat.items(): bat[k] = v

    # --- Map/rename core pitching columns to your training names ---
    # Create IPouts from IP
    if 'IP' not in pit.columns: pit['IP'] = 0.0
    pit['IPouts'] = pit['IP'].apply(baseball_ip_to_outs)

    pitching_map = {
        'G':'G_pit','GS':'GS','SO':'SO_pit','BB':'BB_pit','ER':'ER','HR':'HR_pit',
        'SV':'SV','W':'W_pit','L':'L_pit','H':'H_pit'
    }
    for src, dst in pitching_map.items():
        if src not in pit.columns:
            pit[src] = 0
    pit.rename(columns=pitching_map, inplace=True)

    # FanGraphs advanced pitching
    if 'WAR' in pit.columns:   pit['pit_WAR_fg']  = pit['WAR']
    if 'FIP' in pit.columns:   pit['pit_FIP']     = pit['FIP']
    if 'K%' in pit.columns:    pit['pit_Kpct']    = pit['K%']
    if 'BB%' in pit.columns:   pit['pit_BBpct']   = pit['BB%']
    if 'ERA-' in pit.columns:  pit['pit_ERA_minus']= pit['ERA-']
    if 'xFIP' in pit.columns:  pit['pit_xFIP']    = pit['xFIP']

    # --- Reduce to necessary columns & combine batting/pitching rows ---
    bat_keep = ['Name','Team','lgID'] + list(batting_map.values()) + list(adv_bat.keys())
    pit_keep = ['Name','Team','lgID','IPouts'] + list(pitching_map.values()) + \
               [c for c in ['pit_WAR_fg','pit_FIP','pit_Kpct','pit_BBpct','pit_ERA_minus','pit_xFIP'] if c in pit.columns]

    bat_cur = bat[bat_keep].copy()
    pit_cur = pit[pit_keep].copy()

    # Many players appear only as batters or only as pitchers.
    # We'll outer-merge on (Name, Team, lgID) — good enough for current-season scoring.
    cur = pd.merge(bat_cur, pit_cur, on=['Name','Team','lgID'], how='outer')

    # Attach team WinPct
    wp = compute_team_winpct(year)  # ['Team','WinPct']
    cur = cur.merge(wp, on='Team', how='left')
    cur['WinPct'] = cur['WinPct'].fillna(0.5)  # fallback if any team not matched

    # Fielding features absent in FG pull — set to 0
    ensure_columns(cur, ['G_fld','PO','A','E','DP','FieldPct'])

    # Fill missing numeric with 0
    num_cols = cur.select_dtypes(include=[np.number]).columns
    cur[num_cols] = cur[num_cols].replace([np.inf,-np.inf], np.nan).fillna(0)

    return cur

def load_model_and_features(task_dir: Path):
    model = joblib.load(task_dir / "model_logreg.joblib") if (task_dir / "model_logreg.joblib").exists() \
            else joblib.load(task_dir / "model_randomforest.joblib")
    feat_cols = joblib.load(task_dir / "feature_columns.joblib")
    return model, feat_cols

def score_and_rank(df_cur: pd.DataFrame, model, feature_cols: List[str], prob_col: str) -> pd.DataFrame:
    # Ensure all required feature columns exist; add missing as 0
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
    # Keep useful columns for display
    keep = ['lgID','Team','Name',prob_col,'bat_WAR_fg','bat_wRC_plus','bat_OPS','bat_OBP','bat_SLG',
            'RBI','HR','AB','WinPct','pit_WAR_fg','pit_FIP','pit_Kpct','IPouts','SV','SO_pit','ER']
    keep = [c for c in keep if c in df.columns]
    view = df[keep].copy()
    # rank within each league
    return (view.groupby('lgID', group_keys=False)
                .apply(lambda g: g.sort_values(prob_col, ascending=False).head(5))
                .reset_index(drop=True))

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # 1) Build current-season features
    current = build_current_features(CURRENT_YEAR)

    # 2) Load models & their feature lists
    mvp_model, mvp_feats = load_model_and_features(Path("models/MVP_top5"))
    cy_model,  cy_feats  = load_model_and_features(Path("models/CY_top5"))

    # 3) Score
    scored_mvp = score_and_rank(current, mvp_model, mvp_feats, prob_col="MVP_prob")
    scored_cy  = score_and_rank(current, cy_model,  cy_feats,  prob_col="CY_prob")

    # 4) Top 5 per league
    top5_mvp = top5_per_league(scored_mvp, "MVP_prob")
    top5_cy  = top5_per_league(scored_cy,  "CY_prob")

    # 5) Save & print
    outdir = Path("predictions") / f"{CURRENT_YEAR}"
    outdir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    top5_mvp.to_csv(outdir / f"top5_mvp_{CURRENT_YEAR}_{ts}.csv", index=False)
    top5_cy.to_csv(outdir / f"top5_cy_{CURRENT_YEAR}_{ts}.csv", index=False)

    print("\n=== Top 5 MVP per league ===")
    print(top5_mvp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Top 5 Cy Young per league ===")
    print(top5_cy.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n✅ Saved CSVs to {outdir.resolve()}")
