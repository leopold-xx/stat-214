import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


def pecarn_binary_predict(df):
    """Return binary PECARN predictions (1=rule positive, 0=very low risk)."""

    is_ge2 = df["age_two_plus"] == 2
    is_lt2 = df["age_two_plus"] == 1

    # Altered mental status (includes GCS < 15)
    ams = (df["gcs_total"] < 15)
    if "ams" in df.columns:
        ams = ams | (df["ams"] == 1)

    # Severe mechanism
    severe_mech = df["injury_severity_high_impact"] == 3

    # <2 years predictors
    palpable = df["palpable_skull_fracture"].isin([1, 2])
    nonfrontal_hema = df["scalp_hematoma_location"].isin([2, 3])
    loc_ge_5s = df["loc_duration"].isin([2, 3, 4])
    not_acting_normal = df["acting_normal"] == 0

    # ≥2 years predictors
    basilar = df["basilar_skull_fracture_signs"] == 1
    loc_any = df["loss_of_consciousness"].isin([1, 2])
    vomiting = df["vomiting"] == 1
    severe_headache = df["headache_severity"] == 3

    # Apply age-specific rule
    positive = (
        # <2 years
        (is_lt2 & (ams | palpable | nonfrontal_hema | loc_ge_5s | severe_mech | not_acting_normal))
        |
        # ≥2 years
        (is_ge2 & (ams | basilar | loc_any | vomiting | severe_mech | severe_headache))
    )

    return positive.astype(int)


def fit_logistic(X_train, y_train):
    """Fit logistic regression (balanced class weight)."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
    )
    model.fit(X_train, y_train)
    return model


def predict_logistic(model, X_test, threshold=0.005):
    """Return binary predictions from logistic model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred

def fit_hgb(X_train, y_train):
    """Fit histogram gradient boosting classifier."""
    model = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        random_state=0,
    )
    model.fit(X_train, y_train)
    return model


def predict_hgb(model, X_test, threshold=0.5):
    """Return binary predictions for HGB."""
    y_prob = model.predict_proba(X_test)[:, 1]
    return (y_prob >= threshold).astype(int)


def evaluate_model(y_true, y_pred):
    """Evaluate binary predictions (0/1)."""

    y_true = np.asarray(pd.Series(y_true))
    y_pred = np.asarray(pd.Series(y_pred))

    # Drop missing pairs (rare, but safe)
    mask = (~pd.isna(y_true)) & (~pd.isna(y_pred))
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)

    # Sanity check
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths after masking.")

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = int(len(y_true))

    def safe_div(a, b):
        return a / b if b != 0 else np.nan

    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "ppv": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "accuracy": safe_div(tp + tn, n),
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
    }


def make_folds(n, n_splits=10, seed=0):
    """Return a list of (train_idx, test_idx) for K-fold CV."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(n)
    return list(kf.split(idx))

def cv_rule_model(df, y_col, predict_fn, n_splits=10, seed=0):
    """10-fold CV evaluation for rule-based model (no training)."""
    folds = make_folds(len(df), n_splits=n_splits, seed=seed)

    fold_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        # Rule model ignores training data; we still evaluate on test fold only
        df_test = df.iloc[test_idx]
        y_true = df_test[y_col]
        y_pred = predict_fn(df_test)

        m = evaluate_model(y_true, y_pred)
        m["fold"] = fold_id
        fold_rows.append(m)

    res = pd.DataFrame(fold_rows)
    summary = res.drop(columns=["fold"]).mean(numeric_only=True).to_frame().T
    return res, summary


def cv_trainable_model(df, y_col, x_cols, fit_fn, predict_fn, n_splits=10, seed=0):
    """10-fold CV evaluation for trainable binary model."""
    folds = make_folds(len(df), n_splits=n_splits, seed=seed)

    fold_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train = train_df[x_cols]
        y_train = train_df[y_col]
        X_test = test_df[x_cols]
        y_test = test_df[y_col]

        model = fit_fn(X_train, y_train)
        y_pred = predict_fn(model, X_test)

        m = evaluate_model(y_test, y_pred)
        m["fold"] = fold_id
        fold_rows.append(m)

    res = pd.DataFrame(fold_rows)
    summary = res.drop(columns=["fold"]).mean(numeric_only=True).to_frame().T
    return res, summary