# models.py
"""
Three models for predicting clinically-important TBI (PosIntFinal):
  1. PECARN Clinical Decision Rule (Kuppermann 2009)
  2. Logistic Regression
  3. Random Forest
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from clean import clean_data
import os

#os.chdir(os.path.dirname(os.path.abspath(__file__)))


from rules import STRUCTURAL_DETAIL_COLS


# MODEL 1: PECARN Clinical Decision Rule (Kuppermann et al. 2009)


def apply_cdr(df: pd.DataFrame) -> pd.Series:
    """
    Reconstruct Kuppermann's Decision rule as a binary classifier.
    Returns 1 (not very low risk: CT or observation may be warranted)
    or 0 (very low risk).
    """
    predictions = pd.Series(np.nan, index=df.index)

    # Under 2 years
    # Predictors: AMS, non-frontal scalp hematoma, LOC >= 5s,
    #             severe mechanism, palpable skull fx, not acting normally
    mask_under2 = df["AgeTwoPlus"].eq(1)
    if mask_under2.any():
        d = df.loc[mask_under2]
        pred = (
            d["AMS"].eq(1)
            | (d["Hema"].eq(1) & d["HemaLoc"].isin([2, 3]))
            | (d["LOCSeparate"].eq(1) & d["LocLen"].isin([2, 3, 4]))
            | d["High_impact_InjSev"].eq(3)
            | d["SFxPalp"].isin([1, 2])
            | d["ActNorm"].eq(0)
        )
        predictions.loc[mask_under2] = pred.astype(int)

    # 2 years and older
    # Predictors: AMS, any LOC, vomiting,
    #             severe mechanism, basilar skull fx signs, severe headache
    mask_over2 = df["AgeTwoPlus"].eq(2)
    if mask_over2.any():
        d = df.loc[mask_over2]
        pred = (
            d["AMS"].eq(1)
            | d["LOCSeparate"].isin([1, 2])
            | d["Vomit"].eq(1)
            | d["High_impact_InjSev"].eq(3)
            | d["SFxBas"].eq(1)
            | (d["HA_verb"].eq(1) & d["HASeverity"].eq(3))
        )
        predictions.loc[mask_over2] = pred.astype(int)

    return predictions.astype(int)


# Feature preparation — for logistic regression & random forest
#
# Sentinel "not applicable" codes (92, 91) are replaced with NaN,
# then filled with 0s so sklearn doesn't treat them as meaningful numeric values.

# Original PECARN CDR features (Kuppermann 2009)
FEATURES_PECARN = [
    "High_impact_InjSev",
    "LOCSeparate",
    "Vomit",
    "AMS",
    "SFxBas",
    "SFxPalp",
    "Hema",
    "HA_verb",
]

# EDA-informed features — drops high CT-rate/low-yield symptoms,
# adds ones where the gap suggests better signal
FEATURES_EDA = [
    "SFxBas",       # highest yield both age groups
    "AMS",          # high yield, strong signal both groups
    "Seiz",         # high yield relative to CT rate
    "NeuroD",       # consistent yield both groups
    "SFxPalp",      # very high yield especially under-2
    "LOCSeparate",  # decent yield both groups
    "Amnesia_verb", # over-2 signal, moderate yield
]

# Amnesia_verb & HA_verb → 0 for non-verbal patients (91 = pre-verbal sentinel)
NA_SENTINELS = {"HASeverity": 92, "HA_verb": 91, "Amnesia_verb": 91, "SeizLen": 92}


def prepare_features(df: pd.DataFrame, feature_set: str = "pecarn") -> pd.DataFrame:
    """Select and encode features, replacing sentinel codes and filling NaNs with 0."""
    features = FEATURES_PECARN if feature_set == "pecarn" else FEATURES_EDA
    X = df[features].copy()

    # convert sentinel codes → NaN
    for col, code in NA_SENTINELS.items():
        if col in X.columns:
            X[col] = X[col].replace(code, np.nan)

    # structural NaNs mean "not applicable" → 0
    structural_in_features = [c for c in STRUCTURAL_DETAIL_COLS if c in X.columns]
    if structural_in_features:
        X[structural_in_features] = X[structural_in_features].fillna(0)

    # remaining NaNs: unobserved/not reported → 0
    X = X.fillna(0)

    # convert nullable Int64 → float64 for sklearn
    X = X.astype(float)

    nan_cols = X.columns[X.isna().any()].tolist()
    assert not nan_cols, f"prepare_features: NaNs remain in {nan_cols}"

    return X


# MODEL 2: Logistic Regression
#
# StandardScaler: required because LR coefficients are scale-sensitive.
# L2 penalty: shrinks large coefficients to reduce overfitting.
# class_weight='balanced': up-weights the rare ciTBI class (~1% of data).


def build_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
) -> Pipeline:
    """Fit a scaled, L2-penalized logistic regression. Returns fitted Pipeline."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                penalty="l2",
                C=C,
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# MODEL 3: Random Forest


def pick_threshold_for_target_sensitivity(
    y_true, y_prob, target=0.85, min_specificity=0.10
):
    """Scan thresholds high-to-low; return first t meeting
    sensitivity & specificity targets."""
    thresholds = np.linspace(1, 0, 501)
    best_t = None
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if sens >= target and spec >= min_specificity:
            best_t = t
            break

    if best_t is None:
        print(
            f"Warning: target sensitivity {target} with min specificity "
            f"{min_specificity} unreachable — defaulting to 0.5"
        )
        return 0.5

    return best_t


def build_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 8,
    min_samples_leaf: int = 20,
    threshold: float = 0.10,
) -> tuple[RandomForestClassifier, float]:
    """
    Fit a Random Forest classifier.

    Returns (fitted model, threshold) — threshold is passed through
    so evaluate_model can apply it consistently at predict time.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf, threshold


def get_feature_importances(
    rf: RandomForestClassifier, feature_names: list
) -> pd.DataFrame:
    """Return a sorted DataFrame of RF feature importances (mean Gini decrease)."""
    return (
        pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# Evaluation


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_name: str = "Model",
    ct_done: pd.Series | None = None,
) -> dict:
    """
    Print and return key clinical metrics: sensitivity, specificity, NPV, PPV,
    ROC AUC (if probabilities provided), and CT rate.

    Implied CT rate = fraction of patients flagged (predicted=1). Core clinical
    tradeoff: sensitivity near 1.0 (miss no ciTBI) vs. low CT rate (less radiation).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)

    metrics = {
        "model": model_name,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "ct_rate": float(y_pred.sum() / n),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    if ct_done is not None:
        metrics["ct_rate_observed"] = float(ct_done.mean())

    print(f"\n{'=' * 50}\n  {model_name}\n{'=' * 50}")
    print(f"  Sensitivity (ciTBI caught):     {metrics['sensitivity']:.3f}")
    print(f"  Specificity (no ciTBI cleared): {metrics['specificity']:.3f}")
    print(f"  NPV:                            {metrics['npv']:.3f}")
    print(f"  PPV:                            {metrics['ppv']:.3f}")
    flagged = f"{int(y_pred.sum())}/{n}"
    print(f"  Implied CT rate:                {metrics['ct_rate']:.3f}  ({flagged})")
    if "ct_rate_observed" in metrics:
        print(f"  Observed CT rate (actual):      {metrics['ct_rate_observed']:.3f}")
    if "roc_auc" in metrics:
        print(f"  ROC AUC:                        {metrics['roc_auc']:.3f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return metrics


# Run all three models


def run_all_models(
    df_clean: pd.DataFrame,
    model_name: str,
    outcome_col: str = "PosIntFinal",
    test_size: float = 0.2,
    feature_set: str = "pecarn",  # "pecarn" or "eda"
    age_group: str = "all",       # "all", "under2", "2plus"
) -> dict:
    """Train and evaluate a model on the cleaned PECARN dataset."""
    # convert nullable Int64 → float64 for sklearn compatibility
    df_clean = df_clean.copy()
    df_clean = df_clean.astype({
        col: float for col in df_clean.select_dtypes("Int64").columns
    })

    # age filter
    if age_group == "under2":
        df_clean = df_clean[df_clean["AgeinYears"] < 2]
    elif age_group == "2plus":
        df_clean = df_clean[df_clean["AgeinYears"] >= 2]

    # three-way split: 60% train / 20% val / 20% test
    df_train, df_test = train_test_split(
        df_clean,
        test_size=test_size,
        stratify=df_clean[outcome_col],
        random_state=42,
    )
    df_tr, df_val = train_test_split(
        df_train,
        test_size=0.25,  # 0.25 of 0.8 = 0.20 overall
        stratify=df_train[outcome_col],
        random_state=42,
    )

    X_tr = prepare_features(df_tr, feature_set=feature_set)
    X_val = prepare_features(df_val, feature_set=feature_set)
    X_test = prepare_features(df_test, feature_set=feature_set)

    y_tr = df_tr[outcome_col]
    y_val = df_val[outcome_col]
    y_test = df_test[outcome_col]

    ct_done_test = df_test["CTDone"] if "CTDone" in df_test.columns else None

    if model_name == "cdr":
        label = f"PECARN CDR ({age_group})"
        return evaluate_model(
            y_test, apply_cdr(df_test), model_name=label, ct_done=ct_done_test
        )

    elif model_name == "lr":
        model = build_logistic_regression(X_tr, y_tr)
        label = f"Logistic Regression ({feature_set}, {age_group})"
        return evaluate_model(
            y_test,
            model.predict(X_test),
            model.predict_proba(X_test)[:, 1],
            model_name=label,
            ct_done=ct_done_test,
        )

    elif model_name == "rf":
        model, _ = build_random_forest(X_tr, y_tr)

        # tune threshold on val set to hit target sensitivity
        y_val_prob = model.predict_proba(X_val)[:, 1]
        best_threshold = pick_threshold_for_target_sensitivity(
            y_val, y_val_prob, target=0.95
        )

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= best_threshold).astype(int)
        label = (
            f"Random Forest ({feature_set}, {age_group}, tuned t={best_threshold:.3f})"
        )
        return evaluate_model(
            y_test, y_pred, y_prob, model_name=label, ct_done=ct_done_test
        )

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: 'cdr', 'lr', 'rf'"
        )
    

if __name__ == "__main__":

    df_raw = pd.read_csv("../data/TBI PUD 10-08-2013.csv")
    df_model = clean_data(df_raw, mode="model", run_validation=False)

    for model_name in ["cdr", "lr", "rf"]:
        for features in ["pecarn", "eda"]:
            run_all_models(df_model, model_name=model_name, feature_set=features)