# clean.py
"""
Cleaning pipeline:

1) fix_gcs_sum_errors
2) na_overview
3) flag_informative_missing
4) convert_structural_codes_and_drop_outcomes

"""

import numpy as np
import pandas as pd

from validate import run_all_validations
from rules import STRUCTURAL_DETAIL_COLS


# Fix GCS sum errors
def fix_gcs_sum_errors(df, verbose=False):
    """
    Fix GCS scores where components don't sum to total.
    Recalculates GCSTotal from components and updates GCSGroup if needed.

    Returns:
        (df_fixed, correction_summary)
    """
    df_fixed = df.copy()

    required = ["GCSEye", "GCSVerbal", "GCSMotor", "GCSTotal"]
    if not all(c in df_fixed.columns for c in required):
        return df_fixed, {"n_corrected": 0, "corrected_indices": [], "skipped": True}

    df_fixed["_GCS_calculated"] = (
        df_fixed["GCSEye"] + df_fixed["GCSVerbal"] + df_fixed["GCSMotor"]
    )

    sum_mismatch = (
        (df_fixed["_GCS_calculated"] != df_fixed["GCSTotal"])
        & df_fixed[required].notna().all(axis=1)
    )

    n_corrections = int(sum_mismatch.sum())

    if n_corrections > 0:
        df_fixed.loc[sum_mismatch, "GCSTotal"] = df_fixed.loc[
            sum_mismatch, "_GCS_calculated"
        ]

        if "GCSGroup" in df_fixed.columns:
            df_fixed.loc[sum_mismatch, "GCSGroup"] = df_fixed.loc[
                sum_mismatch, "GCSTotal"
            ].apply(lambda x: 1 if (3 <= x <= 13) else 2)

    correction_summary = {
        "n_corrected": n_corrections,
        "corrected_indices": df_fixed.index[sum_mismatch].tolist(),
    }

    df_fixed = df_fixed.drop(columns=["_GCS_calculated"])
    if verbose:
        print(correction_summary)

    return df_fixed, correction_summary


# Missingness overview
def na_overview(df, outcome_col="PosIntFinal", threshold=0.02):
    results = []

    na_props = df.isna().mean()
    cols_to_check = na_props[na_props > threshold].index

    for col in cols_to_check:
        if col == outcome_col:
            continue

        ct = pd.crosstab(df[col].isna(), df[outcome_col], normalize="index")

        if True not in ct.index or False not in ct.index:
            continue
        if 1 not in ct.columns:
            continue

        p_missing = ct.loc[True, 1]
        p_observed = ct.loc[False, 1]

        risk_diff = p_missing - p_observed
        risk_ratio = (p_missing / p_observed) if p_observed > 0 else np.nan

        results.append({
            "Variable": col,
            "Missing_Prop": float(na_props[col]),
            "P(TBI | Missing)": float(p_missing),
            "P(TBI | Observed)": float(p_observed),
            "Risk_Difference": float(risk_diff),
            "Risk_Ratio": float(risk_ratio) if pd.notna(risk_ratio) else np.nan,
        })

    summary = pd.DataFrame(results)
    if summary.empty:
        return summary

    return summary.sort_values("Missing_Prop", ascending=False)


# Flag informative missing variables — no imputation
def flag_informative_missing(
    raw_df,
    summary_df,
    missing_threshold=0.02,
    rr_threshold=2,
    use_rr_filter=True,
    skip_cols=("PatNum", "PosIntFinal", "PosCT"),
):
    """
    Adds binary _missing flag columns for variables with informative missingness.
    Does NOT impute anything — original values are left as NaN.
    """
    df_out = raw_df.copy()

    if use_rr_filter:
        flag_df = summary_df[
            (summary_df["Missing_Prop"] > missing_threshold)
            & (summary_df["Risk_Ratio"] > rr_threshold)
        ]
    else:
        flag_df = summary_df[summary_df["Missing_Prop"] > missing_threshold]

    vars_to_flag = flag_df["Variable"].to_list()
    vars_to_flag = [
        c
        for c in vars_to_flag
        if c in df_out.columns
        and c not in skip_cols
        and c not in STRUCTURAL_DETAIL_COLS
    ]

    if vars_to_flag:
        flags = df_out[vars_to_flag].isna().astype(int)
        flags.columns = [f"{c}_missing" for c in vars_to_flag]
        df_out = pd.concat([df_out, flags], axis=1)

    return df_out, vars_to_flag


# Convert structural codes and drop missing outcomes
def convert_structural_codes_and_drop_outcomes(
    df_flagged,
    outcome_cols=None,
    max_allowed_missing=0.01,
    main_outcome="PosIntFinal",
    structural_code=92,
    verbose=False,
):
    """
    1. Drops rows missing the main outcome
    2. Converts structural code (92) -> NaN in structural detail columns
    3. Drops rows missing any outcome variable (if within allowed threshold)

    No imputation is performed anywhere.
    """
    if outcome_cols is None:
        outcome_cols = [
            "PosIntFinal",
            "DeathTBI",
            "HospHead",
            "Intub24Head",
            "Neurosurgery",
        ]

    df = df_flagged.copy()

    df = df.dropna(subset=[main_outcome])

    structural_cols = [col for col in STRUCTURAL_DETAIL_COLS if col in df.columns]
    if structural_cols:
        df[structural_cols] = df[structural_cols].replace(structural_code, np.nan)

    if verbose:
        n_nan = int(df.isna().sum().sum())
        print("Remaining NaNs after structural code conversion:", n_nan)

    total_rows = len(df)
    missing_counts = df[outcome_cols].isna().sum()
    total_missing = float(missing_counts.sum())
    missing_prop = total_missing / total_rows if total_rows > 0 else 0.0

    if verbose:
        print("Missing outcome counts:\n", missing_counts)
        print("Total missing outcome proportion:", missing_prop)

    if missing_prop > max_allowed_missing:
        raise ValueError(
            f"Missing outcome proportion ({missing_prop:.4f}) exceeds threshold."
        )

    df = df.dropna(subset=outcome_cols)

    if verbose:
        print("Final remaining NaNs:", int(df.isna().sum().sum()))

    return df


# Modeling subset filter
def apply_modeling_subset(df):
    """
    Restrict to minor head trauma cases (GCS 14-15),
    consistent with PECARN eligibility criteria.
    GCS 3-13 cases should be excluded for modeling.
    """
    df_out = df.copy()
    if "GCSGroup" in df_out.columns:
        df_out = df_out[df_out["GCSGroup"] == 2]
    elif "GCSTotal" in df_out.columns:
        df_out = df_out[df_out["GCSTotal"] >= 14]
    return df_out


def clean_data(
    df,
    mode="eda",
    run_validation=True,
    verbose=False,
    return_report=False,
    use_rr_filter=True,
    missing_threshold=0.02,
):
    """
    Main cleaning function.

    Args:
        df:               raw DataFrame
        mode:             "eda"   -> full cleaned dataset (all GCS groups)
                          "model" -> restrict to GCS 14-15 (PECARN minor head trauma)
        run_validation:   run validate checks before/after
        verbose:          print debug info
        return_report:    return (df_final, report) instead of just df_final

    Returns:
        Cleaned DataFrame, or (DataFrame, report) if return_report=True.
    """
    report = {}

    if run_validation:
        report["validation_before"] = run_all_validations(df)

    categorical_cols = [
        "Gender",
        "Ethnicity",
        "Race",
        "EmplType",
        "InjuryMech",
        "Vomit",
        "Seiz",
        "GCSEye",
        "GCSVerbal",
        "GCSMotor",
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Step 1: GCS fixes
    df1, gcs_fix_report = fix_gcs_sum_errors(df, verbose=verbose)
    report["gcs_fix"] = gcs_fix_report

    # Step 2: missingness analysis + flags (no imputation)
    summary = na_overview(df1, threshold=missing_threshold)
    report["missingness_summary"] = summary

    df2, flagged_vars = flag_informative_missing(
        df1,
        summary,
        missing_threshold=missing_threshold,
        use_rr_filter=use_rr_filter,
    )
    report["flagged_vars"] = flagged_vars

    # Step 3: structural code conversion + drop outcomes
    df_final = convert_structural_codes_and_drop_outcomes(df2, verbose=verbose)
    report["n_rows_final"] = len(df_final)
    report["n_cols_final"] = df_final.shape[1]

    # Step 4: restrict to modeling population if mode="model"
    if mode == "model":
        df_final = apply_modeling_subset(df_final)
        report["n_rows_after_subset"] = len(df_final)
        if verbose:
            print(f"[model mode] Rows after GCS 14-15 filter: {len(df_final)}")

    if run_validation:
        report["validation_after"] = run_all_validations(df_final)

    if return_report:
        return df_final, report

    return df_final