# file: train_awards_baseline.py
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

RANDOM_STATE = 42

# ---------------------------
# 1) Load and prepare data
# ---------------------------
DATA_PATH = Path("player_season_features_with_fg_1980_present.csv")  # from your merge step
df = pd.read_csv(DATA_PATH)

# Basic sanity: restrict to 1980+ and drop rows missing year/league
df = df[(df["yearID"] >= 1980) & df["lgID"].isin(["AL", "NL"])].copy()

# Labels (created in your merge step)
LABELS = {
    "MVP": "is_top5_MVP",
    "CY":  "is_top5_CY",
}

# Non-feature columns to exclude if present
EXCLUDE = {
    "id_like": ["playerID", "teamID", "lgID", "yearID"],
    "labels": list(LABELS.values()),
    # raw awards columns (won't exist for current season at prediction time)
    "awards": [
        "pointsWon_Most_Valuable_Player","pointsMax_Most_Valuable_Player",
        "voteShare_Most_Valuable_Player","votesFirst_Most_Valuable_Player",
        "pointsWon_Cy_Young_Award","pointsMax_Cy_Young_Award",
        "voteShare_Cy_Young_Award","votesFirst_Cy_Young_Award",
    ],
}

EXCLUDE_COLS = set(sum(EXCLUDE.values(), []))
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype != "O"]

# Fill NaNs in features
df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

# Time-based split: train on <= 2019, test on 2017–2023
train_mask = df["yearID"] <= 2019
test_mask  = (df["yearID"] >= 2020) & (df["yearID"] <= 2023)

def recall_at_5_by_year_league(df_eval: pd.DataFrame, proba_col: str, label_col: str):
    """
    For each (yearID, lgID), take top-5 predicted probs and compute
    how many of the true top-5 were captured (Recall@5).
    Returns overall mean and a breakdown table.
    """
    rows = []
    for (year, lg), grp in df_eval.groupby(["yearID", "lgID"]):
        # rank predictions
        grp = grp.sort_values(proba_col, ascending=False)
        pred_top5_idx = set(grp.head(5).index)

        # true top-5 (label=1)
        true_top_idx = set(grp.index[grp[label_col] == 1])

        if len(true_top_idx) == 0:
            continue  # no labeled positives (shouldn't happen)

        hits = len(pred_top5_idx & true_top_idx)
        rec5 = hits / min(5, len(true_top_idx))
        rows.append({"yearID": year, "lgID": lg, "hits": hits, "recall_at_5": rec5})

    out = pd.DataFrame(rows)
    return (out["recall_at_5"].mean() if not out.empty else np.nan), out.sort_values(["yearID","lgID"])

def fit_and_eval(task_name: str, label_col: str):
    print(f"\n=== {task_name}: Training & Evaluation ===")
    y = df[label_col].astype(int)
    X = df[FEATURE_COLS]

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    meta_test = df.loc[test_mask, ["yearID","lgID","playerID"]].copy()

    # Two baselines: Logistic Regression & Random Forest
    logit = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # sparse-safe; with_mean=False in case of all-zero features
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=None
        ))
    ])

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE
    )

    # Fit both
    logit.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predict probabilities
    p_logit = logit.predict_proba(X_test)[:,1]
    p_rf    = rf.predict_proba(X_test)[:,1]

    # Global metrics
    def global_metrics(p, name):
        auc = roc_auc_score(y_test, p)
        y_hat = (p >= 0.5).astype(int)
        f1  = f1_score(y_test, y_hat)
        prec = precision_score(y_test, y_hat, zero_division=0)
        rec  = recall_score(y_test, y_hat)
        print(f"{name} :: AUC={auc:.3f}  F1={f1:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")

    print("\n-- Global metrics on 2017–2023 test --")
    global_metrics(p_logit, "LogReg")
    global_metrics(p_rf,    "RandForest")

    # Recall@5 per (year, league)
    eval_df = meta_test.copy()
    eval_df["y_true"] = y_test.values
    eval_df["p_logit"] = p_logit
    eval_df["p_rf"]    = p_rf

    r5_logit_mean, r5_logit_tbl = recall_at_5_by_year_league(
        eval_df.rename(columns={"y_true": label_col}), "p_logit", label_col
    )
    r5_rf_mean, r5_rf_tbl = recall_at_5_by_year_league(
        eval_df.rename(columns={"y_true": label_col}), "p_rf", label_col
    )

    print("\n-- Recall@5 (mean over 2017–2023 per league) --")
    print(f"LogReg Recall@5:  {r5_logit_mean:.3f}")
    print(f"RF     Recall@5:  {r5_rf_mean:.3f}")

    # Save per-year tables for review
    out_dir = Path("models") / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    r5_logit_tbl.to_csv(out_dir / "recall_at5_logreg.csv", index=False)
    r5_rf_tbl.to_csv(out_dir / "recall_at5_randomforest.csv", index=False)

    # Save models
    joblib.dump(logit, out_dir / "model_logreg.joblib")
    joblib.dump(rf,    out_dir / "model_randomforest.joblib")
    joblib.dump(FEATURE_COLS, out_dir / "feature_columns.joblib")

    print(f"\nSaved models and metrics to {out_dir.resolve()}")

    # Feature importances / coefficients (top 20)
    try:
        coefs = pd.Series(logit.named_steps["clf"].coef_[0], index=FEATURE_COLS).sort_values(key=np.abs, ascending=False)
        coefs.head(20).to_csv(out_dir / "logreg_top_coeffs.csv")
    except Exception:
        pass

    try:
        importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        importances.head(20).to_csv(out_dir / "rf_top_importances.csv")
    except Exception:
        pass

# ---------------------------
# 2) Train/Eval both tasks
# ---------------------------
fit_and_eval("MVP_top5", LABELS["MVP"])
fit_and_eval("CY_top5",  LABELS["CY"])

print("\n✅ Done.")
