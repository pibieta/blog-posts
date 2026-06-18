"""
Cross-target CMI pipeline on Lending Club loan-level data.

Computes the score-vs-target diagnostic triad that post 2 of the CMI series
ultimately reports:

    AUC(S, Y^{(1)}), AUC(S, Y^{(2)})                  -- discrimination
    I(S; Y^{(1)})                                      -- self-information
    I(S; Y^{(2)})                                      -- marginal cross-target
    I(S; Y^{(2)} | Y^{(1)})                            -- conditional cross-target
    I(S; Y^{(2)} | Y^{(1)}) / H(Y^{(2)} | Y^{(1)})     -- normalized portability

DESIGN CHOICES
--------------
* Cohort
    36-month term loans whose status is terminal (Fully Paid, Charged Off,
    Default, plus the "Does not meet credit policy" variants of those).

* Targets
    Y^{(1)} = "loan accrued at least one late-payment fee"  (total_rec_late_fee > 0)
    Y^{(2)} = "loan ended in loss"                          (Charged Off / Default)

* Score
    S^{(1)} : application-time features -> Y^{(1)}, fit with histogram GBDT,
    out-of-fold predictions over K=5 stratified folds.

* CMI estimation
    Log-loss gain identity (post 1) with cross-fitted auxiliaries:
        * self / marginal CMI :   isotonic calibration of S -> Y
        * conditional CMI :       histogram GBDT on (S, Y^{(1)}) -> Y^{(2)}

* Leakage hygiene
    Only application-time features used.  Post-origination fields
    (total_pymnt, recoveries, last_pymnt_*, last_fico_*, out_prncp, etc.)
    are explicitly excluded.  In particular total_rec_late_fee IS the
    target Y^{(1)} and is never used as a feature.

USAGE
-----
    python cross_target_pipeline.py [--sample-size N] [--seed SEED]
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

HERE = Path(__file__).parent
DATA = HERE / "data" / "accepted_2007_to_2018Q4.csv"
COHORT_CACHE = HERE / "data" / "cohort_36mo_terminal.pkl"
RESULTS_JSON = HERE / "cross_target_results.json"

TERMINAL_PAID = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
TERMINAL_LOSS = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
TERMINAL = TERMINAL_PAID | TERMINAL_LOSS

# Application-time numerical features (populated at loan origination).
NUM_FEATURES = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc",
    "dti", "fico_range_low", "fico_range_high",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc",
    "mths_since_last_delinq", "mths_since_last_record",
]
# Application-time categorical features.
CAT_FEATURES = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "application_type", "term",
]
# Strings parsed into derived numerics.
EXTRA = ["emp_length", "earliest_cr_line"]

LOAN_STATUS = "loan_status"
LATE_FEE = "total_rec_late_fee"

USED_COLS = NUM_FEATURES + CAT_FEATURES + EXTRA + [LOAN_STATUS, LATE_FEE]


# ---------------------------------------------------------------------------
# Feature parsing helpers

_EMP_NUM = re.compile(r"(\d+)")


def parse_emp_length(s):
    """'< 1 year' -> 0.5, '10+ years' -> 10, NaN -> NaN."""
    if pd.isna(s):
        return np.nan
    s = str(s)
    if s == "< 1 year":
        return 0.5
    m = _EMP_NUM.search(s)
    return float(m.group(1)) if m else np.nan


def parse_credit_line(s, reference="2018-12-01"):
    """Months between earliest_cr_line and a fixed reference date."""
    if pd.isna(s):
        return np.nan
    try:
        d = pd.to_datetime(s, format="%b-%Y")
        return (pd.Timestamp(reference) - d).days / 30.4375
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Stage 1 — cohort load + subsample

def load_cohort(sample_size: int, seed: int) -> pd.DataFrame:
    if COHORT_CACHE.exists():
        print(f"[load] reading cached cohort {COHORT_CACHE.name}")
        df = pd.read_pickle(COHORT_CACHE)
    else:
        print(f"[load] streaming {DATA.name} (no cache; first run)")
        parts = []
        for chunk in pd.read_csv(DATA, usecols=USED_COLS,
                                 chunksize=200_000, low_memory=False):
            mask = (chunk["term"].astype(str).str.contains("36", na=False)
                    & chunk[LOAN_STATUS].isin(TERMINAL))
            parts.append(chunk[mask])
        df = pd.concat(parts, ignore_index=True)
        del parts
        gc.collect()
        df.to_pickle(COHORT_CACHE)
        print(f"[load] saved cache {COHORT_CACHE.name}")
    print(f"[load] cohort rows = {len(df):,}")
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        print(f"[load] subsampled to {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Stage 2 — features + targets

def build_xy(df: pd.DataFrame):
    print("[features] parsing + encoding")
    late_fee = pd.to_numeric(df[LATE_FEE], errors="coerce").fillna(0.0)
    Y1 = (late_fee > 0).astype(np.int8).values
    Y2 = df[LOAN_STATUS].isin(TERMINAL_LOSS).astype(np.int8).values

    df = df.copy()
    df["emp_length_yrs"] = df["emp_length"].apply(parse_emp_length)
    df["mths_since_earliest_cr"] = df["earliest_cr_line"].apply(parse_credit_line)

    num_cols = NUM_FEATURES + ["emp_length_yrs", "mths_since_earliest_cr"]
    X_num = df[num_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = pd.DataFrame(
        enc.fit_transform(df[CAT_FEATURES].astype(str)),
        columns=CAT_FEATURES,
    ).astype(np.float32)

    X = pd.concat([X_num.reset_index(drop=True),
                   X_cat.reset_index(drop=True)], axis=1)
    cat_idx = list(range(len(num_cols), len(num_cols) + len(CAT_FEATURES)))

    print(f"[features] X shape = {X.shape}  "
          f"({len(num_cols)} numeric, {len(CAT_FEATURES)} categorical)")
    print(f"[features] Y(1) positive rate = {Y1.mean():.4%}")
    print(f"[features] Y(2) positive rate = {Y2.mean():.4%}")
    return X.values, Y1, Y2, cat_idx


# ---------------------------------------------------------------------------
# Stage 3 — out-of-fold score S^{(1)}

def oof_score(X, Y1, cat_idx, K=5, seed=0):
    print(f"[score] training out-of-fold S^(1) with K={K} folds")
    S = np.zeros(len(X), dtype=np.float64)
    kf = StratifiedKFold(K, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(kf.split(X, Y1), start=1):
        t0 = time.time()
        m = HistGradientBoostingClassifier(
            max_iter=400, max_depth=6, learning_rate=0.05,
            categorical_features=cat_idx, random_state=seed,
        )
        m.fit(X[tr], Y1[tr])
        S[te] = m.predict_proba(X[te])[:, 1]
        print(f"[score]  fold {fold}/{K} done in {time.time()-t0:.1f}s")
    return np.clip(S, 1e-12, 1 - 1e-12)


# ---------------------------------------------------------------------------
# Stage 4 — CMI estimators

def H_b(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def _loss(y, p):
    return -y * np.log(p) - (1 - y) * np.log(1 - p)


def cf_marginal_cmi(S, Y, K=5, seed=0):
    """Cross-fitted I(S; Y) via the log-loss gain identity."""
    fold_means = []
    for tr, te in StratifiedKFold(K, shuffle=True,
                                  random_state=seed).split(S, Y):
        rate = float(np.clip(Y[tr].mean(), 1e-12, 1 - 1e-12))
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(S[tr], Y[tr])
        p0 = np.full(len(te), rate)
        p1 = np.clip(cal.predict(S[te]), 1e-12, 1 - 1e-12)
        fold_means.append((_loss(Y[te], p0) - _loss(Y[te], p1)).mean())
    fm = np.asarray(fold_means)
    return fm.mean(), fm.std(ddof=1) / np.sqrt(K)


def cf_conditional_cmi(S, Y1, Y2, K=5, seed=0):
    """Cross-fitted I(S; Y2 | Y1).

    Baseline auxiliary: empirical P(Y2 | Y1) (2-cell lookup).
    Full auxiliary: HistGBDT on (S, Y1) -> Y2.
    """
    fold_means = []
    for tr, te in StratifiedKFold(K, shuffle=True,
                                  random_state=seed).split(S, Y2):
        # Baseline rates
        m1 = (Y1[tr] == 1)
        m0 = (Y1[tr] == 0)
        rate1 = float(np.clip(Y2[tr][m1].mean() if m1.any() else 0.5,
                              1e-12, 1 - 1e-12))
        rate0 = float(np.clip(Y2[tr][m0].mean() if m0.any() else 0.5,
                              1e-12, 1 - 1e-12))
        p0_te = np.where(Y1[te] == 1, rate1, rate0)

        # Full auxiliary
        Z_tr = np.column_stack([S[tr], Y1[tr]])
        Z_te = np.column_stack([S[te], Y1[te]])
        m = HistGradientBoostingClassifier(
            max_iter=400, max_depth=5, learning_rate=0.05,
            random_state=seed,
        )
        m.fit(Z_tr, Y2[tr])
        p1_te = np.clip(m.predict_proba(Z_te)[:, 1], 1e-12, 1 - 1e-12)

        fold_means.append((_loss(Y2[te], p0_te) - _loss(Y2[te], p1_te)).mean())
    fm = np.asarray(fold_means)
    return fm.mean(), fm.std(ddof=1) / np.sqrt(K)


def conditional_entropy_Y2_given_Y1(Y1, Y2):
    """H(Y2 | Y1) computed on the full sample."""
    p_y1_1 = float(Y1.mean())
    p_y1_0 = 1.0 - p_y1_1
    p_y2_given_y1_1 = float(Y2[Y1 == 1].mean()) if (Y1 == 1).any() else 0.0
    p_y2_given_y1_0 = float(Y2[Y1 == 0].mean()) if (Y1 == 0).any() else 0.0
    return (p_y1_1 * H_b(np.array(p_y2_given_y1_1))
            + p_y1_0 * H_b(np.array(p_y2_given_y1_0)))


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=200_000,
                    help="random subsample after cohort filtering (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_cohort(args.sample_size, args.seed)
    X, Y1, Y2, cat_idx = build_xy(df)
    del df
    gc.collect()

    # --- S^{(1)}: trained on Y^{(1)} (the cross-target score) ---
    print("\n[score] S^(1) -- trained on Y^(1)")
    S1 = oof_score(X, Y1, cat_idx, K=5, seed=args.seed)
    auc_S1_Y1 = roc_auc_score(Y1, S1)
    auc_S1_Y2 = roc_auc_score(Y2, S1)
    print(f"  AUC(S^(1), Y(1)) = {auc_S1_Y1:.4f}")
    print(f"  AUC(S^(1), Y(2)) = {auc_S1_Y2:.4f}")

    print("\n[CMI] S^(1) triad")
    si, si_se = cf_marginal_cmi(S1, Y1, seed=args.seed)
    print(f"  I(S^(1); Y(1))          = {si:>9.4f}  +/- {si_se:.4f} nats")
    mc, mc_se = cf_marginal_cmi(S1, Y2, seed=args.seed)
    print(f"  I(S^(1); Y(2))          = {mc:>9.4f}  +/- {mc_se:.4f} nats")
    cc, cc_se = cf_conditional_cmi(S1, Y1, Y2, seed=args.seed)
    print(f"  I(S^(1); Y(2) | Y(1))   = {cc:>9.4f}  +/- {cc_se:.4f} nats")

    # --- S^{(2)}: same architecture, trained directly on Y^{(2)} (baseline) ---
    print("\n[score] S^(2) -- baseline, same architecture trained on Y^(2)")
    S2 = oof_score(X, Y2, cat_idx, K=5, seed=args.seed)
    auc_S2_Y2 = roc_auc_score(Y2, S2)
    print(f"  AUC(S^(2), Y(2)) = {auc_S2_Y2:.4f}")
    cc2, cc2_se = cf_conditional_cmi(S2, Y1, Y2, seed=args.seed)
    print(f"  I(S^(2); Y(2) | Y(1))   = {cc2:>9.4f}  +/- {cc2_se:.4f} nats")

    # --- Normalization and the retraining-gain readout ---
    h_y2_y1 = conditional_entropy_Y2_given_Y1(Y1, Y2)
    ratio_S1 = cc / h_y2_y1 if h_y2_y1 > 0 else float("nan")
    ratio_S2 = cc2 / h_y2_y1 if h_y2_y1 > 0 else float("nan")
    portability = cc / cc2 if cc2 > 0 else float("nan")

    print(f"\n[norm] H(Y(2) | Y(1)) = {h_y2_y1:.4f} nats")
    print(f"[norm] S^(1) resolves {ratio_S1:.2%} of remaining Y(2)-uncertainty | Y(1)")
    print(f"[norm] S^(2) resolves {ratio_S2:.2%}  (same-architecture baseline)")
    print(f"[norm] portability ratio  I(S^(1);Y(2)|Y(1)) / I(S^(2);Y(2)|Y(1))"
          f"  = {portability:.2%}")
    print(f"        (fraction of retrainable conditional information"
          f" that the cross-target score captures)")

    results = {
        "config": {
            "sample_size": int(len(Y1)),
            "seed": int(args.seed),
            "n_features": int(X.shape[1]),
            "cohort": "36-month term, terminal status",
            "Y1_def": "total_rec_late_fee > 0",
            "Y2_def": "loan_status in {Charged Off, Default, ...}",
        },
        "base_rates": {
            "p_Y1": float(Y1.mean()),
            "p_Y2": float(Y2.mean()),
        },
        "auc": {
            "S1_vs_Y1": float(auc_S1_Y1),
            "S1_vs_Y2": float(auc_S1_Y2),
            "S2_vs_Y2": float(auc_S2_Y2),
        },
        "cmi_nats_S1": {
            "I_S_Y1":       {"mean": float(si),  "se": float(si_se)},
            "I_S_Y2":       {"mean": float(mc),  "se": float(mc_se)},
            "I_S_Y2_g_Y1":  {"mean": float(cc),  "se": float(cc_se)},
        },
        "cmi_nats_S2": {
            "I_S_Y2_g_Y1":  {"mean": float(cc2), "se": float(cc2_se)},
        },
        "H_Y2_given_Y1": float(h_y2_y1),
        "normalized_ratios": {
            "S1_over_H": float(ratio_S1),
            "S2_over_H": float(ratio_S2),
            "portability_S1_over_S2": float(portability),
        },
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n[save] {RESULTS_JSON.name}")


if __name__ == "__main__":
    main()
