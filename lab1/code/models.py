#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py

Example:
  python code/model.py --clean_csv data/cleaned_data.csv --out_dir output
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance


# ============================================================
# Helpers
# ============================================================

def to01(s: pd.Series) -> pd.Series:
    """Robust mapping of many boolean-like representations to {0,1,NaN}."""
    if getattr(s.dtype, "name", "") == "boolean":
        return s.astype("float")
    x = s.copy()
    x = x.astype(str).str.strip().str.lower()
    x = x.replace({
        "true": "1", "false": "0",
        "t": "1", "f": "0",
        "yes": "1", "no": "0",
    })
    out = pd.to_numeric(x, errors="coerce")
    out = out.where(out.isin([0, 1]), np.nan)
    return out


def _as_bool(s: pd.Series, default: bool = False) -> pd.Series:
    """Coerce to pandas BooleanDtype, fill NaN with default."""
    return s.astype("boolean").fillna(default)


def _is_yes_int(s: pd.Series) -> pd.Series:
    """int-coded yes/no: 1->True, 0->False, others->NA; returns BooleanDtype with NA."""
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=x.index, dtype="boolean")
    out = out.mask(x == 0, False)
    out = out.mask(x == 1, True)
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = (z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / den
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def choose_threshold_sensitivity(y_true: np.ndarray, p: np.ndarray, target_sens: float = 0.99) -> float:
    """Pick the largest threshold that still achieves >= target_sens (conservative)."""
    p = np.asarray(p, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    ths = np.unique(p[~np.isnan(p)])
    ths = np.sort(ths)[::-1]
    best = float(ths[-1]) if len(ths) else 0.5

    for th in ths:
        pred = (p >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        if sens >= target_sens:
            best = float(th)
            break
    return best


def calibration_bins(p: np.ndarray, y_true: np.ndarray, label: str, q: int = 10) -> pd.DataFrame:
    """Robust calibration binning (decile-like) handling repeated probabilities."""
    cal = pd.DataFrame({"p": p, "y": y_true}).dropna()

    nunique = cal["p"].nunique()
    q_eff = int(min(q, nunique)) if nunique > 1 else 1

    if q_eff == 1:
        cal["bin"] = "B1"
    else:
        cal["bin"] = pd.qcut(cal["p"], q_eff, duplicates="drop")
        cats = cal["bin"].cat.categories
        mapping = {cats[i]: f"B{i+1}" for i in range(len(cats))}
        cal["bin"] = cal["bin"].map(mapping).astype("object")

    tab = cal.groupby("bin", observed=False).agg(
        n=("y", "count"),
        k=("y", "sum"),
        obs=("y", "mean"),
        pred=("p", "mean"),
    ).reset_index()

    los, his = [], []
    for _, r in tab.iterrows():
        lo, hi = wilson_ci(int(r["k"]), int(r["n"]))
        los.append(lo)
        his.append(hi)
    tab["lo"], tab["hi"] = los, his
    tab["model"] = label
    return tab


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns: {miss}")


def perturb_age_missingness(
    X: pd.DataFrame,
    *,
    age_col: str = "age_years",
    cols_to_perturb=None,
    p_under2: float = 0.25,
    p_2to5: float = 0.12,
    p_5plus: float = 0.05,
    random_state: int = 7,
) -> pd.DataFrame:
    """
    Inject extra missingness into selected columns, with higher rates at younger ages.
    This simulates age-dependent measurement/documentation constraints.
    """
    rng = np.random.default_rng(random_state)
    Xp = X.copy()

    if cols_to_perturb is None:
        cols_to_perturb = list(X.columns)

    age = pd.to_numeric(Xp[age_col], errors="coerce")
    g_under2 = age < 2
    g_2to5 = (age >= 2) & (age < 5)
    g_5plus = age >= 5

    p = np.zeros(len(Xp), dtype=float)
    p[np.asarray(g_under2.fillna(False))] = p_under2
    p[np.asarray(g_2to5.fillna(False))] = p_2to5
    p[np.asarray(g_5plus.fillna(False))] = p_5plus

    for c in cols_to_perturb:
        if c not in Xp.columns:
            continue
        mask = rng.random(len(Xp)) < p
        Xp.loc[mask, c] = np.nan

    return Xp


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman via ranks (no scipy)."""
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    ra = ra - np.nanmean(ra)
    rb = rb - np.nanmean(rb)
    denom = np.sqrt(np.nansum(ra**2) * np.nansum(rb**2))
    return float(np.nansum(ra * rb) / denom) if denom > 0 else np.nan


def flip_rate(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean(a != b))


# ============================================================
# PECARN-like CDR (binary CT recommendation)
# ============================================================

def pecarn_cdr_recommend_ct(df: pd.DataFrame, *, cdr_mode: str = "conservative") -> pd.Series:
    """
    Binary CT recommendation from PECARN-like rule.

    """
    under2 = df.get("age_under2", pd.Series(False, index=df.index))
    under2 = under2.astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)

    ams = _as_bool(df.get("altered_mental_status__b", pd.Series(pd.NA, index=df.index)), default=False)
    vomit = _as_bool(df.get("vomiting_anytime_after_injury__b", pd.Series(pd.NA, index=df.index)), default=False)
    basilar = _as_bool(df.get("basilar_skull_fracture_signs__b", pd.Series(pd.NA, index=df.index)), default=False)
    hema = _as_bool(df.get("scalp_hematoma_or_swelling__b", pd.Series(pd.NA, index=df.index)), default=False)

    acting_normal = df.get("parent_reports_acting_normal__b", pd.Series(pd.NA, index=df.index)).astype("boolean")
    acting_normal_missing = acting_normal.isna()
    acting_normal = acting_normal.fillna(False)

    gcs = pd.to_numeric(df.get("gcs_total", pd.Series(np.nan, index=df.index)), errors="coerce")
    gcs_lt15 = (gcs < 15)

    loc_yes = _is_yes_int(df.get("loss_of_consciousness_history", pd.Series(np.nan, index=df.index))).fillna(False)
    psf_yes = _is_yes_int(df.get("palpable_skull_fracture", pd.Series(np.nan, index=df.index))).fillna(False)

    mech3 = df.get("mechanism_severity_3", pd.Series(np.nan, index=df.index)).astype("object")
    severe_mech = (mech3 == "high")

    # VERIFY with codebook
    headache_sev = pd.to_numeric(df.get("headache_severity", pd.Series(np.nan, index=df.index)), errors="coerce")
    severe_headache = headache_sev.isin([3, 4, 5])
    severe_headache_missing = headache_sev.isna()

    # VERIFY with codebook
    hema_loc = pd.to_numeric(df.get("scalp_hematoma_location", pd.Series(np.nan, index=df.index)), errors="coerce")
    frontal_code = 1
    nonfrontal_hema = hema & (~hema_loc.isin([frontal_code]))
    hema_loc_missing = hema & hema_loc.isna()

    high_u2 = under2 & (gcs_lt15 | ams | psf_yes)
    high_o2 = (~under2) & (gcs_lt15 | ams | basilar)

    inter_u2 = under2 & (~high_u2) & (
        nonfrontal_hema | hema_loc_missing | loc_yes | severe_mech | (~acting_normal) | acting_normal_missing
    )
    inter_o2 = (~under2) & (~high_o2) & (
        loc_yes | vomit | severe_headache | severe_mech | severe_headache_missing
    )

    if cdr_mode == "strict":
        rec = high_u2 | high_o2
    else:
        rec = high_u2 | high_o2 | inter_u2 | inter_o2

    return rec.astype(int)


def save_model_outputs(
    out_dir: Path,
    *,
    y_test: np.ndarray,
    p_logit_test: np.ndarray,
    p_gb_test: np.ndarray,
    pred_logit: np.ndarray,
    pred_gb: np.ndarray,
    pred_cdr: np.ndarray,
    th_logit: float,
    th_gb: float,
    target_sens: float,
    tab_l: pd.DataFrame,
    tab_g: pd.DataFrame,
    cdr_tab: pd.DataFrame,
    
    p_logit_pert_test: Optional[np.ndarray] = None,
    p_gb_pert_test: Optional[np.ndarray] = None,
    pred_logit_orig: Optional[np.ndarray] = None,
    pred_logit_pert: Optional[np.ndarray] = None,
    pred_gb_orig: Optional[np.ndarray] = None,
    pred_gb_pert: Optional[np.ndarray] = None,
    pred_cdr_orig: Optional[np.ndarray] = None,
    pred_cdr_pert: Optional[np.ndarray] = None,
    logit_pipe: Optional[Pipeline] = None,
    meta: Optional[dict] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tab_l.to_csv(out_dir / "logit_calibration.csv", index=False)
    tab_g.to_csv(out_dir / "gb_calibration.csv", index=False)
    cdr_tab.to_csv(out_dir / "cdr_calibration.csv", index=False)

    payload: dict[str, object] = dict(
        y_test=np.asarray(y_test, dtype=int),
        p_logit_test=np.asarray(p_logit_test, dtype=float),
        p_gb_test=np.asarray(p_gb_test, dtype=float),
        pred_logit=np.asarray(pred_logit, dtype=int),
        pred_gb=np.asarray(pred_gb, dtype=int),
        pred_cdr=np.asarray(pred_cdr, dtype=int),
        th_logit=float(th_logit),
        th_gb=float(th_gb),
        target_sens=float(target_sens),
    )

    def _pack(tab: pd.DataFrame, prefix: str):
        keep = ["bin", "n", "k", "obs", "pred", "lo", "hi"]
        t = tab[keep].copy()
        payload[f"{prefix}_cal_bin"] = t["bin"].astype(str).to_numpy()
        payload[f"{prefix}_cal_n"] = t["n"].to_numpy(dtype=int)
        payload[f"{prefix}_cal_k"] = t["k"].to_numpy(dtype=int)
        payload[f"{prefix}_cal_obs"] = t["obs"].to_numpy(dtype=float)
        payload[f"{prefix}_cal_pred"] = t["pred"].to_numpy(dtype=float)
        payload[f"{prefix}_cal_lo"] = t["lo"].to_numpy(dtype=float)
        payload[f"{prefix}_cal_hi"] = t["hi"].to_numpy(dtype=float)

    _pack(tab_l, "logit")
    _pack(tab_g, "gb")
    _pack(cdr_tab, "cdr")

    
    def _maybe(name: str, arr, dtype):
        if arr is not None:
            payload[name] = np.asarray(arr, dtype=dtype)

    _maybe("p_logit_pert_test", p_logit_pert_test, float)
    _maybe("p_gb_pert_test", p_gb_pert_test, float)
    _maybe("pred_logit_orig", pred_logit_orig, int)
    _maybe("pred_logit_pert", pred_logit_pert, int)
    _maybe("pred_gb_orig", pred_gb_orig, int)
    _maybe("pred_gb_pert", pred_gb_pert, int)
    _maybe("pred_cdr_orig", pred_cdr_orig, int)
    _maybe("pred_cdr_pert", pred_cdr_pert, int)

   
    if logit_pipe is not None:
        try:
            prep = logit_pipe.named_steps["prep"]
            model = logit_pipe.named_steps["model"]
            feat_names = prep.get_feature_names_out().astype(str)
            coef = np.asarray(model.coef_).reshape(-1).astype(float)
            intercept = np.asarray(model.intercept_).reshape(-1).astype(float)
            payload["logit_feature_names"] = feat_names
            payload["logit_coef"] = coef
            payload["logit_intercept"] = intercept
        except Exception as e:
            print("[warn] could not export logit feature names/coef:", repr(e))

    npz_path = out_dir / "model_outputs.npz"
    np.savez_compressed(npz_path, **payload)

    if meta is not None:
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return npz_path


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_csv", type=str, required=True, help="Path to cleaned_data.csv")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for artifacts")
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--target_sens", type=float, default=0.99)
    ap.add_argument("--cdr_mode", type=str, default="conservative", choices=["conservative", "strict"])
    args = ap.parse_args()

    clean_csv = Path(args.clean_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(clean_csv, low_memory=False)

    YCOL = "clinically_important_tbi__recalc"
    features_num = ["age_years"]
    features_cat = ["mechanism_severity_3"]
    features_bin = [
        "altered_mental_status__b",
        "vomiting_anytime_after_injury__b",
        "basilar_skull_fracture_signs__b",
        "scalp_hematoma_or_swelling__b",
        "parent_reports_acting_normal__b",
    ]
    use_cols = features_num + features_cat + features_bin
    _ensure_cols(df, [YCOL] + use_cols)

    y_raw = to01(df[YCOL])
    mask_y = y_raw.notna()

    X = df.loc[mask_y, use_cols].copy()
    y = y_raw.loc[mask_y].astype(int).to_numpy()

    # dtype normalization
    X["age_years"] = pd.to_numeric(X["age_years"], errors="coerce")
    X["mechanism_severity_3"] = X["mechanism_severity_3"].astype("object")
    for c in features_bin:
        X[c] = X[c].astype("float")

    cdr_pred_all = pecarn_cdr_recommend_ct(df.loc[mask_y], cdr_mode=args.cdr_mode).to_numpy()

    X_train, X_test, y_train, y_test, cdr_train, cdr_test = train_test_split(
        X, y, cdr_pred_all,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ]), features_num),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), features_cat),
            ("bin", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ]), features_bin),
        ],
        remainder="drop"
    )

    # Logistic
    logit = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced"
    )
    logit_pipe = Pipeline(steps=[("prep", preprocess), ("model", logit)])
    logit_pipe.fit(X_train, y_train)

    p_logit_train = logit_pipe.predict_proba(X_train)[:, 1]
    p_logit_test = logit_pipe.predict_proba(X_test)[:, 1]

    # Boosting + calibration
    gb = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=400,
        random_state=args.seed
    )
    gb_pipe = Pipeline(steps=[("prep", preprocess), ("model", gb)])
    gb_cal = CalibratedClassifierCV(gb_pipe, method="isotonic", cv=3)
    gb_cal.fit(X_train, y_train)

    p_gb_train = gb_cal.predict_proba(X_train)[:, 1]
    p_gb_test = gb_cal.predict_proba(X_test)[:, 1]

    # Thresholds at target sensitivity on TRAIN
    th_logit = choose_threshold_sensitivity(y_train, p_logit_train, target_sens=args.target_sens)
    th_gb = choose_threshold_sensitivity(y_train, p_gb_train, target_sens=args.target_sens)

    pred_logit = (p_logit_test >= th_logit).astype(int)
    pred_gb = (p_gb_test >= th_gb).astype(int)
    pred_cdr = cdr_test.astype(int)

    # Metrics (test)
    auc_l = roc_auc_score(y_test, p_logit_test)
    auc_g = roc_auc_score(y_test, p_gb_test)
    ap_l = average_precision_score(y_test, p_logit_test)
    ap_g = average_precision_score(y_test, p_gb_test)

    def _sens_spec_ppv_npv(ytrue, yhat):
        tn, fp, fn, tp = confusion_matrix(ytrue, yhat).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                    sens=float(sens), spec=float(spec), ppv=float(ppv), npv=float(npv))

    met = {
        "n_total_modeled": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "target_sens": float(args.target_sens),
        "threshold_logit": float(th_logit),
        "threshold_gb": float(th_gb),
        "auroc_logit": float(auc_l),
        "auroc_gb": float(auc_g),
        "ap_logit": float(ap_l),
        "ap_gb": float(ap_g),
        "operating_point": {
            "cdr": _sens_spec_ppv_npv(y_test, pred_cdr),
            "logit": _sens_spec_ppv_npv(y_test, pred_logit),
            "gb": _sens_spec_ppv_npv(y_test, pred_gb),
        }
    }

    # Confusion matrices CSV
    cm_rows = []
    for name, yhat in [("cdr", pred_cdr), ("logit", pred_logit), ("gb", pred_gb)]:
        tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()
        cm_rows.append({"model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    pd.DataFrame(cm_rows).to_csv(out_dir / "confusion_matrices.csv", index=False)

    # Calibration (test)
    tab_l = calibration_bins(p_logit_test, y_test, "Logistic", q=10)
    tab_g = calibration_bins(p_gb_test, y_test, "Boosting (calibrated)", q=10)

    cdr_df = pd.DataFrame({"bin": np.where(pred_cdr == 1, "CDR=1", "CDR=0"), "y": y_test})
    cdr_tab = cdr_df.groupby("bin").agg(n=("y", "count"), k=("y", "sum"), obs=("y", "mean")).reset_index()
    cdr_tab["pred"] = cdr_tab["bin"].map({"CDR=0": 0.0, "CDR=1": 1.0})
    los, his = [], []
    for _, r in cdr_tab.iterrows():
        lo, hi = wilson_ci(int(r["k"]), int(r["n"]))
        los.append(lo)
        his.append(hi)
    cdr_tab["lo"], cdr_tab["hi"] = los, his
    cdr_tab["model"] = "PECARN CDR"
    cdr_tab = cdr_tab[["bin", "n", "k", "obs", "pred", "lo", "hi", "model"]].copy()

    # Permutation importance on TEST (raw X_test columns)
    pi = permutation_importance(
        gb_cal, X_test, y_test,
        n_repeats=15,
        random_state=args.seed,
        scoring="average_precision",
        n_jobs=-1
    )
    imp = pd.DataFrame({
        "feature": list(X_test.columns),
        "mean": pi.importances_mean,
        "std":  pi.importances_std,
    }).sort_values("mean", ascending=False)
    imp.to_csv(out_dir / "perm_importance.csv", index=False)

    # ============================================================
    # Stability under perturbation (for figm4 / figm5)
    # ============================================================
    cols_perturb = [c for c in list(X_test.columns) if c in [
        "altered_mental_status__b",
        "vomiting_anytime_after_injury__b",
        "basilar_skull_fracture_signs__b",
        "scalp_hematoma_or_swelling__b",
        "parent_reports_acting_normal__b",
    ]]

    X_test_pert = perturb_age_missingness(
        X_test,
        age_col="age_years",
        cols_to_perturb=cols_perturb,
        p_under2=0.25, p_2to5=0.12, p_5plus=0.05,
        random_state=args.seed,
    )

    p_logit_pert = logit_pipe.predict_proba(X_test_pert)[:, 1]
    p_gb_pert = gb_cal.predict_proba(X_test_pert)[:, 1]

    pred_logit_orig = (p_logit_test >= th_logit).astype(int)
    pred_logit_pert = (p_logit_pert >= th_logit).astype(int)
    pred_gb_orig = (p_gb_test >= th_gb).astype(int)
    pred_gb_pert = (p_gb_pert >= th_gb).astype(int)

    # CDR recompute on aligned df rows
    df_test_rows = df.loc[X_test.index].copy()
    df_test_pert = df_test_rows.copy()
    for c in cols_perturb:
        if c in df_test_pert.columns:
            df_test_pert[c] = X_test_pert[c]

    pred_cdr_orig = pecarn_cdr_recommend_ct(df_test_rows, cdr_mode=args.cdr_mode).to_numpy().astype(int)
    pred_cdr_pert = pecarn_cdr_recommend_ct(df_test_pert, cdr_mode=args.cdr_mode).to_numpy().astype(int)

    stability_metrics = pd.DataFrame([
        {
            "model": "PECARN CDR",
            "recommend_flip_rate": flip_rate(pred_cdr_orig, pred_cdr_pert),
            "rank_corr": np.nan,
            "mean_delta_p": np.nan,
        },
        {
            "model": "Logistic",
            "recommend_flip_rate": flip_rate(pred_logit_orig, pred_logit_pert),
            "rank_corr": spearman_corr(p_logit_test, p_logit_pert),
            "mean_delta_p": float(np.mean(p_logit_pert - p_logit_test)),
        },
        {
            "model": "Boosting (calibrated)",
            "recommend_flip_rate": flip_rate(pred_gb_orig, pred_gb_pert),
            "rank_corr": spearman_corr(p_gb_test, p_gb_pert),
            "mean_delta_p": float(np.mean(p_gb_pert - p_gb_test)),
        },
    ])
    stability_metrics.to_csv(out_dir / "stability_metrics.csv", index=False)

    stability_rates = pd.DataFrame([
        {"model": "PECARN CDR", "setting": "Original",  "ct_recommend_rate": float(np.mean(pred_cdr_orig))},
        {"model": "PECARN CDR", "setting": "Perturbed", "ct_recommend_rate": float(np.mean(pred_cdr_pert))},
        {"model": "Logistic",   "setting": "Original",  "ct_recommend_rate": float(np.mean(pred_logit_orig))},
        {"model": "Logistic",   "setting": "Perturbed", "ct_recommend_rate": float(np.mean(pred_logit_pert))},
        {"model": "Boosting (calibrated)", "setting": "Original",  "ct_recommend_rate": float(np.mean(pred_gb_orig))},
        {"model": "Boosting (calibrated)", "setting": "Perturbed", "ct_recommend_rate": float(np.mean(pred_gb_pert))},
    ])
    stability_rates.to_csv(out_dir / "stability_rates.csv", index=False)

    # ---- final save ----
    npz_path = save_model_outputs(
        out_dir,
        y_test=y_test,
        p_logit_test=p_logit_test,
        p_gb_test=p_gb_test,
        pred_logit=pred_logit,
        pred_gb=pred_gb,
        pred_cdr=pred_cdr,
        th_logit=th_logit,
        th_gb=th_gb,
        target_sens=args.target_sens,
        tab_l=tab_l[["bin", "n", "k", "obs", "pred", "lo", "hi", "model"]],
        tab_g=tab_g[["bin", "n", "k", "obs", "pred", "lo", "hi", "model"]],
        cdr_tab=cdr_tab,
        # stability outputs (for figm4/figm5)
        p_logit_pert_test=p_logit_pert,
        p_gb_pert_test=p_gb_pert,
        pred_logit_orig=pred_logit_orig,
        pred_logit_pert=pred_logit_pert,
        pred_gb_orig=pred_gb_orig,
        pred_gb_pert=pred_gb_pert,
        pred_cdr_orig=pred_cdr_orig,
        pred_cdr_pert=pred_cdr_pert,
        logit_pipe=logit_pipe,
        meta=met,
    )

    print("[model] done")
    print("  clean_csv :", clean_csv.resolve())
    print("  out_dir   :", out_dir.resolve())
    print("  npz       :", npz_path.resolve())
    print("  metrics   :", (out_dir / "metrics.json").resolve())


if __name__ == "__main__":
    main()