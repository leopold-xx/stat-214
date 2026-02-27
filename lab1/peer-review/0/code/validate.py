# validate.py
"""
Validation functions (non-mutating checks).

- validate_values: value-set checks using COLUMN_RULES + ALLOWED_VALUES
- validate_patnum: PatNum missing/duplicates
- validate_age: range checks + AgeInMonth/Years consistency summary
- validate_parent_child_logic: parent-child consistency counts
- validate_gcs_scores: GCS sum/group/range checks (returns dict + flagged cases)

You can run these directly in a notebook OR call them inside clean_data().
"""

import numpy as np
import pandas as pd
from scipy import stats

from rules import ALLOWED_VALUES, COLUMN_RULES, CONDITIONAL_GROUPS, PREFIX_RULES


def build_column_rules(df, base_rules=COLUMN_RULES, prefix_rules=PREFIX_RULES):
    """Extend base_rules with prefix-matched columns from df.

    Covers variable families like Finding*, Ind*, and CTSed* that aren't
    listed individually in COLUMN_RULES.
    """
    rules = dict(base_rules)
    for col in df.columns:
        for prefix, rule in prefix_rules.items():
            if col.startswith(prefix):
                rules[col] = rule
    return rules


# Value-set validation


def validate_values(df, column_rules, allowed_values):
    qc_rows = []

    for col, rule in column_rules.items():
        if col not in df.columns:
            qc_rows.append({
                "column": col,
                "rule": rule,
                "status": "COLUMN_MISSING_FROM_DF",
                "unexpected_value": None,
                "count": None,
            })
            continue

        if rule not in allowed_values:
            qc_rows.append({
                "column": col,
                "rule": rule,
                "status": "RULE_NOT_DEFINED",
                "unexpected_value": None,
                "count": None,
            })
            continue

        allowed = allowed_values[rule]
        observed = set(df[col].dropna().unique())
        unexpected = observed - allowed

        if len(unexpected) == 0:
            qc_rows.append({
                "column": col,
                "rule": rule,
                "status": "OK",
                "unexpected_value": None,
                "count": 0,
            })
        else:
            for val in sorted(unexpected):
                qc_rows.append({
                    "column": col,
                    "rule": rule,
                    "status": "UNEXPECTED_VALUE",
                    "unexpected_value": val,
                    "count": int((df[col] == val).sum()),
                })

    qc = pd.DataFrame(qc_rows)
    covered = set(column_rules.keys())
    not_validated = sorted([c for c in df.columns if c not in covered])

    return qc, not_validated


# PatNum validation


def validate_patnum(df, col="PatNum"):
    if col not in df.columns:
        return {"missing_column": True, "n_missing": None, "n_duplicates": None}

    return {
        "missing_column": False,
        "n_missing": int(df[col].isna().sum()),
        "n_duplicates": int(df[col].duplicated().sum()),
    }


# Age validation


def validate_age(df, months_col="AgeInMonth", years_col="AgeinYears", max_months=240):
    out = {
        "missing_months_col": months_col not in df.columns,
        "missing_years_col": years_col not in df.columns,
        "n_months_lt0": None,
        "n_months_gtmax": None,
        "abs_months_years_diff_desc": None,
    }

    if months_col in df.columns:
        out["n_months_lt0"] = int((df[months_col] < 0).sum())
        out["n_months_gtmax"] = int((df[months_col] > max_months).sum())

    if (months_col in df.columns) and (years_col in df.columns):
        mask = df[months_col].notna() & df[years_col].notna()
        if mask.any():
            absdiff = (df.loc[mask, months_col] / 12 - df.loc[mask, years_col]).abs()
            out["abs_months_years_diff_desc"] = absdiff.describe().to_dict()

    return out


# Parent-child validation


def validate_parent_child_logic(df, mapping, structural_code=92):
    results = []

    for parent, children in mapping.items():
        for child in children:
            if parent not in df.columns or child not in df.columns:
                continue

            # Type A: Parent=0 but child has a real (non-missing) value
            type_a = (
                (df[parent] == 0)
                & (df[child].notna())
                & (df[child] != structural_code)
            ).sum()

            # Type B: Parent=1 but child is structural missing code (92)
            type_b = ((df[parent] == 1) & (df[child] == structural_code)).sum()

            # Type C: Parent=1 but child is NaN missing
            type_c = ((df[parent] == 1) & (df[child].isna())).sum()

            # Type D: Parent=0 but child is NaN missing
            type_d = ((df[parent] == 0) & (df[child].isna())).sum()

            results.append({
                "Parent": parent,
                "Child": child,
                "TypeA_Parent0_ChildReal": int(type_a),
                "TypeB_Parent1_Child92": int(type_b),
                "TypeC_Parent1_ChildMissing": int(type_c),
                "TypeD_Parent0_ChildMissing": int(type_d),
            })

    return pd.DataFrame(results)


# GCS validation


def validate_gcs_scores(df):
    """
    Validate Glasgow Coma Scale (GCS) data for consistency.

    Checks:
    1. GCSTotal = GCSEye + GCSVerbal + GCSMotor
    2. GCSGroup matches GCSTotal (3-13 → Group 1, 14-15 → Group 2)
    3. Component scores are within valid ranges
    4. Total score is between 3-15

    Returns:
        dict with violation counts and a flagged_cases DataFrame.
    """
    results = {
        "type1_sum_mismatch": 0,
        "type2_group_mismatch": 0,
        "type3_eye_invalid": 0,
        "type3_verbal_invalid": 0,
        "type3_motor_invalid": 0,
        "type4_total_invalid": 0,
        "flagged_cases": None,
    }

    required = ["GCSEye", "GCSVerbal", "GCSMotor", "GCSTotal"]
    if not all(c in df.columns for c in required):
        results["missing_required_cols"] = [c for c in required if c not in df.columns]
        results["flagged_cases"] = pd.DataFrame()
        return results

    df_work = df.copy()
    df_work["_GCS_calculated"] = (
        df_work["GCSEye"] + df_work["GCSVerbal"] + df_work["GCSMotor"]
    )

    # Type 1: sum mismatch
    sum_mismatch = (
        (df_work["_GCS_calculated"] != df_work["GCSTotal"])
        & df_work[required].notna().all(axis=1)
    )
    results["type1_sum_mismatch"] = int(sum_mismatch.sum())

    # Type 2: group mismatch
    group_mismatch = pd.Series(False, index=df_work.index)
    if "GCSGroup" in df_work.columns:
        df_work["_GCS_expected_group"] = df_work["GCSTotal"].apply(
            lambda x: (
                1 if (pd.notna(x) and 3 <= x <= 13)
                else (2 if (pd.notna(x) and 14 <= x <= 15) else np.nan)
            )
        )
        group_mismatch = (
            (df_work["GCSGroup"] != df_work["_GCS_expected_group"])
            & df_work["GCSTotal"].notna()
            & df_work["GCSGroup"].notna()
        )
        results["type2_group_mismatch"] = int(group_mismatch.sum())

    # Type 3: component out-of-range
    eye_invalid = df_work["GCSEye"].notna() & (~df_work["GCSEye"].isin([1, 2, 3, 4]))
    verbal_invalid = df_work["GCSVerbal"].notna() & (
        ~df_work["GCSVerbal"].isin([1, 2, 3, 4, 5])
    )
    motor_invalid = df_work["GCSMotor"].notna() & (
        ~df_work["GCSMotor"].isin([1, 2, 3, 4, 5, 6])
    )
    results["type3_eye_invalid"] = int(eye_invalid.sum())
    results["type3_verbal_invalid"] = int(verbal_invalid.sum())
    results["type3_motor_invalid"] = int(motor_invalid.sum())

    # Type 4: total out-of-range
    total_invalid = df_work["GCSTotal"].notna() & (
        (df_work["GCSTotal"] < 3) | (df_work["GCSTotal"] > 15)
    )
    results["type4_total_invalid"] = int(total_invalid.sum())

    any_violation = (
        sum_mismatch | group_mismatch | eye_invalid | verbal_invalid
        | motor_invalid | total_invalid
    )

    keep_cols = ["GCSEye", "GCSVerbal", "GCSMotor", "GCSTotal", "_GCS_calculated"]
    if "GCSGroup" in df_work.columns:
        keep_cols += ["GCSGroup"]
    if "_GCS_expected_group" in df_work.columns:
        keep_cols += ["_GCS_expected_group"]

    results["flagged_cases"] = df_work.loc[any_violation, keep_cols].copy()
    return results


def validate_data_types(df):
    """Check if variables have appropriate data types.

    Returns DataFrame of issues, or None if no issues found.
    """
    issues = []

    categorical_cols = [
        "Gender", "Ethnicity", "Race", "EmplType", "Certification",
        "InjuryMech", "Vomit", "Seiz", "CTDone", "PosCT",
        "GCSEye", "GCSVerbal", "GCSMotor", "GCSGroup",
    ]

    for col in categorical_cols:
        if col in df.columns and df[col].dtype == "float64":
            non_missing = df[col].dropna()
            if len(non_missing) > 0 and (non_missing % 1 == 0).all():
                issues.append({
                    "column": col,
                    "current_type": "float64",
                    "issue": "categorical_as_float",
                    "recommended": "convert to Int64 (nullable integer)",
                })

    numeric_cols = ["AgeInMonth", "AgeinYears", "GCSTotal", "PatNum"]
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == "object":
            issues.append({
                "column": col,
                "current_type": "object",
                "issue": "numeric_as_object",
                "recommended": "convert to numeric",
            })

    return pd.DataFrame(issues) if issues else None


def validate_missing_codes(df, suspicious_codes=None):
    """Detect numeric values that might represent missing data.

    Checks for suspicious codes like -1, -9, 99, 999.
    """
    if suspicious_codes is None:
        suspicious_codes = [-1, -9, -99, 99, 999]

    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        for code in suspicious_codes:
            count = (df[col] == code).sum()
            if count > 0:
                results.append({
                    "column": col,
                    "code": code,
                    "count": int(count),
                    "pct": round(count / len(df) * 100, 2),
                })

    return pd.DataFrame(results) if results else None


def validate_outliers(df, method="iqr", threshold=1.5, min_observations=10):
    """Detect statistical outliers in continuous variables only.

    Checks AgeInMonth and AgeinYears; skips all coded/categorical columns.
    """
    outlier_summary = []

    continuous_vars = [c for c in ["AgeInMonth", "AgeinYears"] if c in df.columns]

    for var in continuous_vars:
        data = df[var].dropna()

        if len(data) < min_observations:
            print(f"  Skipping {var}: fewer than {min_observations} observations")
            continue

        if method == "iqr":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                print(f"  Skipping {var}: no variance (IQR = 0)")
                continue

            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outliers = (df[var] < lower) | (df[var] > upper)

        elif method == "zscore":
            if data.std() == 0:
                print(f"  Skipping {var}: no variance (std = 0)")
                continue
            z_scores = np.abs(stats.zscore(data))
            outliers = pd.Series(False, index=df.index)
            outliers.loc[data.index] = z_scores > threshold
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'iqr' or 'zscore'.")

        n_outliers = outliers.sum()

        if n_outliers > 0:
            outlier_values = df.loc[outliers, var]
            outlier_summary.append({
                "variable": var,
                "n_outliers": int(n_outliers),
                "pct_outliers": round(n_outliers / len(df) * 100, 2),
                "method": method,
                "threshold": threshold,
                "data_range": f"[{data.min():.1f}, {data.max():.1f}]",
                "Q1": round(q1, 2) if method == "iqr" else None,
                "median": round(data.median(), 2),
                "Q3": round(q3, 2) if method == "iqr" else None,
                "lower_bound": round(lower, 2) if method == "iqr" else None,
                "upper_bound": round(upper, 2) if method == "iqr" else None,
                "min_outlier": round(outlier_values.min(), 2),
                "max_outlier": round(outlier_values.max(), 2),
            })
            print(f"  {var}: {n_outliers} outliers ({n_outliers / len(df) * 100:.2f}%)")

    if not outlier_summary:
        return None

    return pd.DataFrame(outlier_summary).sort_values("pct_outliers", ascending=False)


def run_all_validations(df):
    """Run complete validation workflow. Returns a dict of outputs for inspection."""
    rules = build_column_rules(df)
    qc_values, not_validated = validate_values(df, rules, ALLOWED_VALUES)

    return {
        "qc_values": qc_values,
        "not_validated": not_validated,
        "patnum": validate_patnum(df),
        "age": validate_age(df),
        "parent_child_table": validate_parent_child_logic(df, CONDITIONAL_GROUPS),
        "gcs": validate_gcs_scores(df),
        "data_types": validate_data_types(df),
        "missing_codes": validate_missing_codes(df),
        "outliers": validate_outliers(df, method="iqr", threshold=3),
    }