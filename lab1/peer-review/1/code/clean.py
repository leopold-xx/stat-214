import pandas as pd
import numpy as np


def clean_data(raw_data):
    """
    Main data cleaning pipeline.

    Steps:
    1) Rename columns to snake_case names.
    2) Handle missing values (convert special codes to NaN, add missing indicator for dizziness, drop columns with too many missing values).
    3) Build and apply a drop plan to drop some selectedcolumns, while
    keeping key predictors like age and GCS.
    """

    df_renamed = rename_columns(raw_data)
    df_cleaned = handle_missing_values(df_renamed)
    plan = build_drop_plan(
        df_cleaned,
        timepoint="pre_ct",
        drop_ethnicity=True,
        high_missing_threshold=0.95,
        keep_age="age_two_plus",
        keep_gcs="gcs_total",
    )
    df_drop, _ = apply_drop_plan(df_cleaned, plan)

    return df_drop


def clean_data_without_dropping(raw_data):
    """
    Alternative cleaning pipeline that does not drop any columns, for comparison.
    """
    df_renamed = rename_columns(raw_data)
    df_cleaned = handle_missing_values(df_renamed)
    return df_cleaned


def rename_columns(df):
    """
    Rename raw PECARN column names to snake_case names.
    """

    rename_map = {
        "PatNum": "patient_id",
        "EmplType": "physician_position",
        "Certification": "physician_certification",
        "InjuryMech": "injury_mechanism",
        "High_impact_InjSev": "injury_severity_high_impact",
        "Amnesia_verb": "amnesia_present",
        "LOCSeparate": "loss_of_consciousness",
        "LocLen": "loc_duration",
        "Seiz": "posttraumatic_seizure",
        "SeizOccur": "seizure_timing",
        "SeizLen": "seizure_duration",
        "ActNorm": "acting_normal",
        "HA_verb": "headache",
        "HASeverity": "headache_severity",
        "HAStart": "headache_onset",
        "Vomit": "vomiting",
        "VomitNbr": "vomiting_count",
        "VomitStart": "vomiting_onset",
        "VomitLast": "vomiting_last",
        "Dizzy": "dizziness",
        "Intubated": "intubated",
        "Paralyzed": "paralyzed",
        "Sedated": "sedated",
        "GCSEye": "gcs_eye",
        "GCSVerbal": "gcs_verbal",
        "GCSMotor": "gcs_motor",
        "GCSTotal": "gcs_total",
        "GCSGroup": "gcs_group",
        "AMS": "ams",
        "AMSAgitated": "ams_agitated",
        "AMSSleep": "ams_sleepy",
        "AMSSlow": "ams_slow",
        "AMSRepeat": "ams_repetitive_questions",
        "AMSOth": "ams_other",
        "SFxPalp": "palpable_skull_fracture",
        "SFxPalpDepress": "palpable_skull_fracture_depressed",
        "FontBulg": "fontanelle_bulging",
        "SFxBas": "basilar_skull_fracture_signs",
        "SFxBasHem": "basilar_hemotympanum",
        "SFxBasOto": "basilar_otorrhea",
        "SFxBasPer": "basilar_periorbital_ecchymosis",
        "SFxBasRet": "basilar_retroauricular_ecchymosis",
        "SFxBasRhi": "basilar_rhinorrhea",
        "Hema": "scalp_hematoma",
        "HemaLoc": "scalp_hematoma_location",
        "HemaSize": "scalp_hematoma_size",
        "Clav": "trauma_above_clavicles",
        "ClavFace": "trauma_face",
        "ClavNeck": "trauma_neck",
        "ClavFro": "trauma_scalp_frontal",
        "ClavOcc": "trauma_scalp_occipital",
        "ClavPar": "trauma_scalp_parietal",
        "ClavTem": "trauma_scalp_temporal",
        "NeuroD": "neuro_deficit",
        "NeuroDMotor": "neuro_deficit_motor",
        "NeuroDSensory": "neuro_deficit_sensory",
        "NeuroDCranial": "neuro_deficit_cranial",
        "NeuroDReflex": "neuro_deficit_reflex",
        "NeuroDOth": "neuro_deficit_other",
        "OSI": "other_substantial_injury",
        "OSIExtremity": "other_injury_extremity",
        "OSICut": "other_injury_or_laceration",
        "OSICspine": "other_injury_cspine",
        "OSIFlank": "other_injury_chest_back_flank",
        "OSIAbdomen": "other_injury_abdomen",
        "OSIPelvis": "other_injury_pelvis",
        "OSIOth": "other_injury_other",
        "Drugs": "drug_or_alcohol_suspicion",
        "CTForm1": "ct_planned",
        "IndAge": "ct_reason_age",
        "IndAmnesia": "ct_reason_amnesia",
        "IndAMS": "ct_reason_ams",
        "IndClinSFx": "ct_reason_clinical_skull_fracture",
        "IndHA": "ct_reason_headache",
        "IndHema": "ct_reason_scalp_hematoma",
        "IndLOC": "ct_reason_loc",
        "IndMech": "ct_reason_mechanism",
        "IndNeuroD": "ct_reason_neuro_deficit",
        "IndRqstMD": "ct_reason_referring_md",
        "IndRqstParent": "ct_reason_parent_request",
        "IndRqstTrauma": "ct_reason_trauma_team",
        "IndSeiz": "ct_reason_seizure",
        "IndVomit": "ct_reason_vomiting",
        "IndXraySFx": "ct_reason_xray_skull_fracture",
        "IndOth": "ct_reason_other",
        "CTSed": "ct_sedation",
        "CTSedAgitate": "ct_sedation_agitated",
        "CTSedAge": "ct_sedation_age",
        "CTSedRqst": "ct_sedation_tech_request",
        "CTSedOth": "ct_sedation_other",
        "AgeInMonth": "age_months",
        "AgeinYears": "age_years",
        "AgeTwoPlus": "age_two_plus",
        "Gender": "gender",
        "Ethnicity": "ethnicity",
        "Race": "race",
        "Observed": "observed_in_ed",
        "EDDisposition": "ed_disposition",
        "CTDone": "ct_done",
        "EDCT": "ct_done_in_ed",
        "PosCT": "tbi_on_ct",
        "Finding1": "ct_finding_1",
        "Finding2": "ct_finding_2",
        "Finding3": "ct_finding_3",
        "Finding4": "ct_finding_4",
        "Finding5": "ct_finding_5",
        "Finding6": "ct_finding_6",
        "Finding7": "ct_finding_7",
        "Finding8": "ct_finding_8",
        "Finding9": "ct_finding_9",
        "Finding10": "ct_finding_10",
        "Finding11": "ct_finding_11",
        "Finding12": "ct_finding_12",
        "Finding13": "ct_finding_13",
        "Finding14": "ct_finding_14",
        "Finding20": "ct_finding_20",
        "Finding21": "ct_finding_21",
        "Finding22": "ct_finding_22",
        "Finding23": "ct_finding_23",
        "DeathTBI": "death_due_to_tbi",
        "HospHead": "hospitalized_head_injury",
        "HospHeadPosCT": "hospitalized_head_injury_with_tbi_on_ct",
        "Intub24Head": "intubated_over_24h_for_head_injury",
        "Neurosurgery": "neurosurgery",
        "PosIntFinal": "citbi",
    }

    df = df.rename(columns=rename_map)
    return df


def summarize_rename(df_before, df_after):
    """
    Simple sanity checks and quick summary.
    """
    out = {}
    out["n_rows_before"] = df_before.shape[0]
    out["n_cols_before"] = df_before.shape[1]
    out["n_rows_after"] = df_after.shape[0]
    out["n_cols_after"] = df_after.shape[1]

    # show columns that did not change name (often indicates missing mapping)
    unchanged = [c for c in df_before.columns if c in df_after.columns]
    out["unchanged_cols"] = unchanged

    return out


def identify_non_numeric_columns(df):
    """
    Identify columns that are not numeric dtype but contain numeric values.
    """

    suspect_cols = []

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                pd.to_numeric(df[col])
                suspect_cols.append(col)
            except (ValueError, TypeError):
                pass

    return suspect_cols


def get_allowed_values():
    """
    Allowed values per variable (after renaming).
    """

    allowed = {
        # identifiers / provider
        "patient_id": "numeric",
        "physician_position": [1, 2, 3, 4, 5, 90],
        "physician_certification": [1, 2, 3, 4, 90],
        # mechanism / history
        "injury_mechanism": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 90],
        "injury_severity_high_impact": [1, 2, 3, 90],
        "amnesia_present": [0, 1, 91],
        "loss_of_consciousness": [0, 1, 2, 90],
        "loc_duration": [1, 2, 3, 4, 92],
        "posttraumatic_seizure": [0, 1, 90],
        "seizure_timing": [1, 2, 3, 92],
        "seizure_duration": [1, 2, 3, 4, 92],
        "acting_normal": [0, 1, 90],
        "headache": [0, 1, 91],
        "headache_severity": [1, 2, 3, 92],
        "headache_onset": [1, 2, 3, 4, 92],
        "vomiting": [0, 1, 90],
        "vomiting_count": [1, 2, 3, 92],
        "vomiting_onset": [1, 2, 3, 4, 92],
        "vomiting_last": [1, 2, 3, 92],
        "dizziness": [0, 1, 91],
        # interventions / GCS
        "intubated": [0, 1, 90],
        "paralyzed": [0, 1, 90],
        "sedated": [0, 1, 90],
        "gcs_eye": [1, 2, 3, 4, 90],
        "gcs_verbal": [1, 2, 3, 4, 5, 90],
        "gcs_motor": [1, 2, 3, 4, 5, 6, 90],
        "gcs_total": "numeric",
        "gcs_group": [1, 2, 90],
        # AMS
        "ams": [0, 1, 90],
        "ams_agitated": [0, 1, 92],
        "ams_sleepy": [0, 1, 92],
        "ams_slow": [0, 1, 92],
        "ams_repetitive_questions": [0, 1, 92],
        "ams_other": [0, 1, 92],
        # skull fracture signs
        "palpable_skull_fracture": [0, 1, 2, 90],
        "palpable_skull_fracture_depressed": [0, 1, 92],
        "fontanelle_bulging": [0, 1, 90],
        "basilar_skull_fracture_signs": [0, 1, 90],
        "basilar_hemotympanum": [0, 1, 92],
        "basilar_otorrhea": [0, 1, 92],
        "basilar_periorbital_ecchymosis": [0, 1, 92],
        "basilar_retroauricular_ecchymosis": [0, 1, 92],
        "basilar_rhinorrhea": [0, 1, 92],
        # scalp hematoma / trauma above clavicles
        "scalp_hematoma": [0, 1, 90],
        "scalp_hematoma_location": [1, 2, 3, 92],
        "scalp_hematoma_size": [1, 2, 3, 92],
        "trauma_above_clavicles": [0, 1, 90],
        "trauma_face": [0, 1, 92],
        "trauma_neck": [0, 1, 92],
        "trauma_scalp_frontal": [0, 1, 92],
        "trauma_scalp_occipital": [0, 1, 92],
        "trauma_scalp_parietal": [0, 1, 92],
        "trauma_scalp_temporal": [0, 1, 92],
        # neuro deficit
        "neuro_deficit": [0, 1, 90],
        "neuro_deficit_motor": [0, 1, 92],
        "neuro_deficit_sensory": [0, 1, 92],
        "neuro_deficit_cranial": [0, 1, 92],
        "neuro_deficit_reflex": [0, 1, 92],
        "neuro_deficit_other": [0, 1, 92],
        # other injury
        "other_substantial_injury": [0, 1, 90],
        "other_injury_extremity": [0, 1, 92],
        "other_injury_or_laceration": [0, 1, 92],
        "other_injury_cspine": [0, 1, 92],
        "other_injury_chest_back_flank": [0, 1, 92],
        "other_injury_abdomen": [0, 1, 92],
        "other_injury_pelvis": [0, 1, 92],
        "other_injury_other": [0, 1, 92],
        # CT plan / indications / sedation
        "drug_or_alcohol_suspicion": [0, 1, 92],
        "ct_planned": [0, 1, 92],
        "ct_reason_age": [0, 1, 92],
        "ct_reason_amnesia": [0, 1, 92],
        "ct_reason_ams": [0, 1, 92],
        "ct_reason_clinical_skull_fracture": [0, 1, 92],
        "ct_reason_headache": [0, 1, 92],
        "ct_reason_scalp_hematoma": [0, 1, 92],
        "ct_reason_loc": [0, 1, 92],
        "ct_reason_mechanism": [0, 1, 92],
        "ct_reason_neuro_deficit": [0, 1, 92],
        "ct_reason_referring_md": [0, 1, 92],
        "ct_reason_parent_request": [0, 1, 92],
        "ct_reason_trauma_team": [0, 1, 92],
        "ct_reason_seizure": [0, 1, 92],
        "ct_reason_vomiting": [0, 1, 92],
        "ct_reason_xray_skull_fracture": [0, 1, 92],
        "ct_reason_other": [0, 1, 92],
        "ct_sedation": [0, 1, 92],
        "ct_sedation_agitated": [0, 1, 92],
        "ct_sedation_age": [0, 1, 92],
        "ct_sedation_tech_request": [0, 1, 92],
        "ct_sedation_other": [0, 1, 92],
        # demographics / disposition
        "age_months": "numeric",
        "age_years": "numeric",
        "age_two_plus": [
            1,
            2,
        ],  # benchmark calls this age_category【:contentReference[oaicite:4]{index=4}】
        "gender": [1, 2],
        "ethnicity": [1, 2],
        "race": [1, 2, 3, 4, 5, 90],
        "observed_in_ed": [0, 1],
        "ed_disposition": [1, 2, 3, 4, 5, 6, 7, 8, 90],
        "ct_done": [0, 1],
        "ct_done_in_ed": [0, 1, 92],
        "tbi_on_ct": [0, 1, 92],
    }

    for k in [
        "ct_finding_1",
        "ct_finding_2",
        "ct_finding_3",
        "ct_finding_4",
        "ct_finding_5",
        "ct_finding_6",
        "ct_finding_7",
        "ct_finding_8",
        "ct_finding_9",
        "ct_finding_10",
        "ct_finding_11",
        "ct_finding_12",
        "ct_finding_13",
        "ct_finding_14",
        "ct_finding_20",
        "ct_finding_21",
        "ct_finding_22",
        "ct_finding_23",
    ]:
        allowed[k] = [0, 1, 92]

    allowed.update(
        {
            "death_due_to_tbi": [0, 1],
            "hospitalized_head_injury": [0, 1],
            "hospitalized_head_injury_with_tbi_on_ct": [0, 1],
            "intubated_over_24h_for_head_injury": [0, 1],
            "neurosurgery": [0, 1],
            "citbi": [0, 1],
        }
    )

    return allowed


def validate_allowed_values(df, allowed_values=None, allow_na=True):
    """
    Validate all columns in allowed_values.
    Returns a tidy dataframe: column, invalid_value, n, examples
    """
    if allowed_values is None:
        allowed_values = get_allowed_values()

    rows = []

    for col, rule in allowed_values.items():
        if col not in df.columns:
            rows.append(
                {
                    "column": col,
                    "invalid_value": "<MISSING_COLUMN>",
                    "n": 1,
                    "examples": "",
                }
            )
            continue

        s = df[col]
        if allow_na:
            s_chk = s.dropna()
        else:
            s_chk = s

        if rule == "numeric":
            bad = pd.to_numeric(s_chk, errors="coerce").isna()
            if bad.any():
                bad_vals = s_chk[bad].value_counts(dropna=False)
                for v, n in bad_vals.items():
                    rows.append(
                        {
                            "column": col,
                            "invalid_value": v,
                            "n": int(n),
                            "examples": ", ".join(
                                map(str, s_chk[s_chk == v].head(5).tolist())
                            ),
                        }
                    )
            continue

        allowed_set = set(rule)
        bad_mask = ~s_chk.isin(allowed_set)
        if bad_mask.any():
            bad_vals = s_chk[bad_mask].value_counts(dropna=False)
            for v, n in bad_vals.items():
                rows.append(
                    {
                        "column": col,
                        "invalid_value": v,
                        "n": int(n),
                        "examples": ", ".join(
                            map(str, s_chk[s_chk == v].head(5).tolist())
                        ),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["column", "invalid_value", "n", "examples"])
    else:
        out = out.sort_values(["column", "n"], ascending=[True, False]).reset_index(
            drop=True
        )
    return out


def check_duplicates(df, key_col="patient_id"):
    """
    Check duplicates at two levels:
    1) duplicate keys (same patient_id appears more than once)
    2) fully duplicated rows (all columns identical)

    Returns a dictionary of summary stats.
    """
    out = {}

    # Duplicate key values
    if key_col in df.columns:
        dup_key_mask = df[key_col].duplicated(keep=False)
        out["n_dup_key_rows"] = int(dup_key_mask.sum())
        out["n_dup_key_values"] = int(df.loc[dup_key_mask, key_col].nunique())
    else:
        out["n_dup_key_rows"] = None
        out["n_dup_key_values"] = None

    # Fully duplicated rows
    dup_row_mask = df.duplicated(keep=False)
    out["n_dup_rows"] = int(dup_row_mask.sum())

    return out


### Missingness Analysis


def summarize_special_missing(df, codes=[90, 91, 92]):
    """
    Calculate proportion of special coded missing values per column.
    """
    summary = []

    for col in df.columns:
        for code in codes:
            prop = (df[col] == code).mean()
            if prop > 0:
                summary.append({"column": col, "code": code, "proportion": prop})

    return pd.DataFrame(summary)


def summarize_total_missing(df, codes=[90, 91, 92]):
    """
    Calculate total missing proportion including NaN and special codes.
    """
    summary = []

    for col in df.columns:
        mask = df[col].isna()
        for code in codes:
            mask |= df[col] == code

        prop = mask.mean()

        summary.append({"column": col, "total_missing_proportion": prop})

    return pd.DataFrame(summary).sort_values(
        "total_missing_proportion", ascending=False
    )


def convert_special_codes_to_nan(df, codes=(90, 91, 92)):
    """
    Convert special coded missing values (e.g., 90/91/92) to NaN.
    """
    return df.replace(list(codes), np.nan)


def add_missing_indicator(df, col):
    """
    Add a binary missing indicator for a given column.
    """
    if col in df.columns:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def summarize_missing(df):
    """
    Return proportion of missing values per column.
    """
    return df.isna().mean().sort_values(ascending=False)


def handle_missing_values(df):
    """
    Main missing value handling pipeline.

    Steps:
    1) Standardize coded missing values (90/91/92) as NaN (values are largely
    structural/design-driven).
    2) Create a missingness indicator for dizziness (missingness is
    informative).
    3) Drop ethnicity (high missingness and approximately MCAR; limited value
    for modeling here).
    4) Leave other low-missing NaNs as-is (no imputation).
    """
    # 1. Convert coded missing
    df = convert_special_codes_to_nan(df)

    # 2. Dizziness missing indicator
    df = add_missing_indicator(df, "dizziness")

    # 3. Drop ethnicity (high missing, likely not MCAR)
    cols_to_drop = ["ethnicity"]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


### Consistency checks


def _is_missing_like(s, missing_codes):
    """True where value is NA or in missing_codes."""
    if s is None:
        return None
    return s.isna() | s.isin(list(missing_codes))


def _safe_col(df, name):
    """Return df[name] if exists else None."""
    return df[name] if name in df.columns else None


def _add_rule(rows, group, rule, mask, n_rows, df=None, max_examples=5):
    """Append one rule summary row."""
    n = int(mask.sum())
    pct = 100.0 * n / n_rows if n_rows else 0.0

    examples = []
    if df is not None and n > 0:
        try:
            examples = df.index[mask].tolist()[:max_examples]
        except Exception:
            examples = []

    rows.append(
        {
            "group": group,
            "rule": rule,
            "n_inconsistent": n,
            "pct_rows": pct,
            "examples": examples,
        }
    )


def summarize_consistency(
    df,
    missing_codes=(90, 91, 92),
    skull_fracture_col="ct_finding_11",
    max_examples=5,
):
    """
    Run consistency checks and return a tidy summary table.
    This function reports issues only and does not modify df.
    """
    n_rows = len(df)
    rows = []

    miss_codes = tuple(missing_codes)

    # Cross-field Logic: Age
    age_years = _safe_col(df, "age_years")
    age_months = _safe_col(df, "age_months")
    age_two_plus = _safe_col(df, "age_two_plus")

    if age_years is not None and age_two_plus is not None:
        miss = _is_missing_like(age_years, miss_codes) | _is_missing_like(
            age_two_plus, miss_codes
        )
        m1 = (~miss) & (age_two_plus == 1) & (age_years >= 2)
        _add_rule(
            rows,
            "Cross-field Logic",
            "age_two_plus=1 but age_years>=2",
            m1,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

        m2 = (~miss) & (age_two_plus == 2) & (age_years < 2)
        _add_rule(
            rows,
            "Cross-field Logic",
            "age_two_plus=2 but age_years<2",
            m2,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if age_years is not None and age_months is not None:
        miss = _is_missing_like(age_years, miss_codes) | _is_missing_like(
            age_months, miss_codes
        )
        est_years = age_months / 12.0
        m = (~miss) & ((est_years - age_years).abs() > 1.0)
        _add_rule(
            rows,
            "Cross-field Logic",
            "age_months and age_years disagree by > 1 year",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # Cross-field Logic: GCS
    gcs_eye = _safe_col(df, "gcs_eye")
    gcs_verbal = _safe_col(df, "gcs_verbal")
    gcs_motor = _safe_col(df, "gcs_motor")
    gcs_total = _safe_col(df, "gcs_total")
    gcs_group = _safe_col(df, "gcs_group")

    if (
        gcs_eye is not None
        and gcs_verbal is not None
        and gcs_motor is not None
        and gcs_total is not None
    ):
        miss = (
            _is_missing_like(gcs_eye, miss_codes)
            | _is_missing_like(gcs_verbal, miss_codes)
            | _is_missing_like(gcs_motor, miss_codes)
            | _is_missing_like(gcs_total, miss_codes)
        )
        m = (~miss) & ((gcs_eye + gcs_verbal + gcs_motor) != gcs_total)
        _add_rule(
            rows,
            "Cross-field Logic",
            "gcs_total != gcs_eye + gcs_verbal + gcs_motor",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if gcs_total is not None and gcs_group is not None:
        miss = _is_missing_like(gcs_total, miss_codes) | _is_missing_like(
            gcs_group, miss_codes
        )
        m1 = (~miss) & (gcs_total >= 14) & (gcs_group != 2)
        _add_rule(
            rows,
            "Cross-field Logic",
            "gcs_total>=14 but gcs_group!=2",
            m1,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

        m2 = (~miss) & (gcs_total < 14) & (gcs_group != 1)
        _add_rule(
            rows,
            "Cross-field Logic",
            "gcs_total<14 but gcs_group!=1",
            m2,
            n_rows,
            df=df,
            max_examples=max_examples,
        )


    # CT & Outcomes
    ct_done = _safe_col(df, "ct_done")
    tbi_on_ct = _safe_col(df, "tbi_on_ct")

    ct_cols = [c for c in df.columns if c.startswith("ct_finding_")]
    non_skull_ct_cols = [c for c in ct_cols if c != skull_fracture_col]

    if ct_done is not None and ct_cols:
        miss = _is_missing_like(ct_done, miss_codes)
        m = (~miss) & (ct_done == 0) & (df[ct_cols] == 1).any(axis=1)
        _add_rule(
            rows,
            "CT & Outcomes",
            "ct_done=0 but a CT finding is marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if tbi_on_ct is not None and ct_done is not None:
        miss = _is_missing_like(tbi_on_ct, miss_codes) | _is_missing_like(
            ct_done, miss_codes
        )
        m = (~miss) & (tbi_on_ct == 1) & (ct_done == 0)
        _add_rule(
            rows,
            "CT & Outcomes",
            "tbi_on_ct=1 but ct_done=0",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if tbi_on_ct is not None and non_skull_ct_cols:
        miss = _is_missing_like(tbi_on_ct, miss_codes)
        m = (~miss) & (tbi_on_ct == 0) & (df[non_skull_ct_cols] == 1).any(axis=1)
        _add_rule(
            rows,
            "CT & Outcomes",
            "tbi_on_ct=0 but non-skull CT finding marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # ciTBI definition-related internal check (if outcome components exist)
    citbi = _safe_col(df, "citbi")
    death = _safe_col(df, "death_due_to_tbi")
    hosp = _safe_col(df, "hospitalized_head_injury")
    hosp_posct = _safe_col(df, "hospitalized_head_injury_with_tbi_on_ct")
    intub24 = _safe_col(df, "intubated_over_24h_for_head_injury")
    neuro = _safe_col(df, "neurosurgery")

    severe_parts = [
        x for x in [death, hosp, hosp_posct, intub24, neuro] if x is not None
    ]
    if citbi is not None and severe_parts:
        miss = _is_missing_like(citbi, miss_codes)
        severe_any = pd.DataFrame(severe_parts).T.eq(1).any(axis=1)
        m = (~miss) & severe_any & (citbi != 1)
        _add_rule(
            rows,
            "CT & Outcomes",
            "severe outcome component but citbi!=1",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )


    # Parent-child Logic

    # Headache -> severity/onset
    headache = _safe_col(df, "headache")
    headache_sev = _safe_col(df, "headache_severity")
    headache_onset = _safe_col(df, "headache_onset")

    if headache is not None and headache_sev is not None:
        miss = _is_missing_like(headache, miss_codes) | _is_missing_like(
            headache_sev, miss_codes
        )
        m = (~miss) & (headache == 0) & (~_is_missing_like(headache_sev, miss_codes))
        _add_rule(
            rows,
            "Parent-child Logic",
            "headache=0 but headache_severity recorded",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

        m = (
            (~_is_missing_like(headache, miss_codes))
            & (headache == 1)
            & _is_missing_like(headache_sev, miss_codes)
        )
        _add_rule(
            rows,
            "Parent-child Logic",
            "headache=1 but headache_severity missing",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if headache is not None and headache_onset is not None:
        miss = _is_missing_like(headache, miss_codes) | _is_missing_like(
            headache_onset, miss_codes
        )
        m = (~miss) & (headache == 0) & (~_is_missing_like(headache_onset, miss_codes))
        _add_rule(
            rows,
            "Parent-child Logic",
            "headache=0 but headache_onset recorded",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

        m = (
            (~_is_missing_like(headache, miss_codes))
            & (headache == 1)
            & _is_missing_like(headache_onset, miss_codes)
        )
        _add_rule(
            rows,
            "Parent-child Logic",
            "headache=1 but headache_onset missing",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # Vomiting -> count/onset/last
    vomiting = _safe_col(df, "vomiting")
    vomiting_count = _safe_col(df, "vomiting_count")
    vomiting_onset = _safe_col(df, "vomiting_onset")
    vomiting_last = _safe_col(df, "vomiting_last")

    if vomiting is not None and vomiting_count is not None:
        miss = _is_missing_like(vomiting, miss_codes) | _is_missing_like(
            vomiting_count, miss_codes
        )
        m = (~miss) & (vomiting == 0) & (~_is_missing_like(vomiting_count, miss_codes))
        _add_rule(
            rows,
            "Parent-child Logic",
            "vomiting=0 but vomiting_count recorded",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if vomiting is not None and vomiting_onset is not None:
        miss = _is_missing_like(vomiting, miss_codes) | _is_missing_like(
            vomiting_onset, miss_codes
        )
        m = (~miss) & (vomiting == 0) & (~_is_missing_like(vomiting_onset, miss_codes))
        _add_rule(
            rows,
            "Parent-child Logic",
            "vomiting=0 but vomiting_onset recorded",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if vomiting is not None and vomiting_last is not None:
        miss = _is_missing_like(vomiting, miss_codes) | _is_missing_like(
            vomiting_last, miss_codes
        )
        m = (~miss) & (vomiting == 0) & (~_is_missing_like(vomiting_last, miss_codes))
        _add_rule(
            rows,
            "Parent-child Logic",
            "vomiting=0 but vomiting_last recorded",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if vomiting is not None and vomiting_count is not None:
        miss_any = _is_missing_like(vomiting, miss_codes)
        miss_details = _is_missing_like(vomiting_count, miss_codes)
        if vomiting_onset is not None:
            miss_details = miss_details & _is_missing_like(vomiting_onset, miss_codes)
        if vomiting_last is not None:
            miss_details = miss_details & _is_missing_like(vomiting_last, miss_codes)

        m = (~miss_any) & (vomiting == 1) & miss_details
        _add_rule(
            rows,
            "Parent-child Logic",
            "vomiting=1 but all vomiting details missing",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # Seizure -> timing/duration
    seizure = _safe_col(df, "posttraumatic_seizure")
    seizure_timing = _safe_col(df, "seizure_timing")
    seizure_duration = _safe_col(df, "seizure_duration")

    if seizure is not None and seizure_timing is not None:
        miss = _is_missing_like(seizure, miss_codes) | _is_missing_like(
            seizure_timing, miss_codes
        )
        m = (~miss) & (seizure == 0) & (~_is_missing_like(seizure_timing, miss_codes))
        _add_rule(
            rows,
            "Parent-child Logic",
            "seizure=0 but seizure_timing recorded",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )
        m = (
            (~_is_missing_like(seizure, miss_codes))
            & (seizure == 1)
            & _is_missing_like(seizure_timing, miss_codes)
        )
        _add_rule(
            rows,
            "Parent-child Logic",
            "seizure=1 but seizure_timing missing",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if seizure is not None and seizure_duration is not None:
        miss = _is_missing_like(seizure, miss_codes) | _is_missing_like(
            seizure_duration, miss_codes
        )
        m = (~miss) & (seizure == 0) & (~_is_missing_like(seizure_duration, miss_codes))
        _add_rule(
            rows,
            "Parent-child Logic",
            "seizure=0 but seizure_duration recorded",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )
        m = (
            (~_is_missing_like(seizure, miss_codes))
            & (seizure == 1)
            & _is_missing_like(seizure_duration, miss_codes)
        )
        _add_rule(
            rows,
            "Parent-child Logic",
            "seizure=1 but seizure_duration missing",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # LOC -> duration
    loc = _safe_col(df, "loss_of_consciousness")
    loc_dur = _safe_col(df, "loc_duration")

    if loc is not None and loc_dur is not None:
        miss = _is_missing_like(loc, miss_codes) | _is_missing_like(loc_dur, miss_codes)
        m = (~miss) & (loc == 0) & (~_is_missing_like(loc_dur, miss_codes))
        _add_rule(
            rows,
            "Parent-child Logic",
            "loss_of_consciousness=0 but loc_duration recorded",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )
        m = (
            (~_is_missing_like(loc, miss_codes))
            & (loc == 1)
            & _is_missing_like(loc_dur, miss_codes)
        )
        _add_rule(
            rows,
            "Parent-child Logic",
            "loss_of_consciousness=1 but loc_duration missing",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # AMS -> subtypes (both directions)
    ams = _safe_col(df, "ams")
    ams_cols = [
        c
        for c in [
            "ams_agitated",
            "ams_sleepy",
            "ams_slow",
            "ams_repetitive_questions",
            "ams_other",
        ]
        if c in df.columns
    ]
    if ams is not None and ams_cols:
        miss_ams = _is_missing_like(ams, miss_codes)
        any_sym = (df[ams_cols] == 1).any(axis=1)

        m = (~miss_ams) & (ams == 0) & any_sym
        _add_rule(
            rows,
            "Parent-child Logic",
            "ams=0 but an AMS subtype is marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

        m = (~miss_ams) & (ams == 1) & (~any_sym)
        _add_rule(
            rows,
            "Parent-child Logic",
            "ams=1 but no AMS subtype marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # Basilar signs -> specific signs
    bas = _safe_col(df, "basilar_skull_fracture_signs")
    bas_cols = [
        c
        for c in [
            "basilar_hemotympanum",
            "basilar_otorrhea",
            "basilar_periorbital_ecchymosis",
            "basilar_retroauricular_ecchymosis",
            "basilar_rhinorrhea",
        ]
        if c in df.columns
    ]
    if bas is not None and bas_cols:
        miss_bas = _is_missing_like(bas, miss_codes)
        any_sign = (df[bas_cols] == 1).any(axis=1)

        m = (~miss_bas) & (bas == 0) & any_sign
        _add_rule(
            rows,
            "Parent-child Logic",
            "basilar_signs=0 but a specific basilar sign is marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

        m = (~miss_bas) & (bas == 1) & (~any_sign)
        _add_rule(
            rows,
            "Parent-child Logic",
            "basilar_signs=1 but no specific basilar sign marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # Hematoma -> location/size
    hema = _safe_col(df, "scalp_hematoma")
    hema_loc = _safe_col(df, "scalp_hematoma_location")
    hema_size = _safe_col(df, "scalp_hematoma_size")

    if hema is not None and hema_loc is not None:
        miss = _is_missing_like(hema, miss_codes) | _is_missing_like(
            hema_loc, miss_codes
        )
        m = (~miss) & (hema == 0) & (~_is_missing_like(hema_loc, miss_codes))
        _add_rule(
            rows,
            "Parent-child Logic",
            "scalp_hematoma=0 but hematoma_location recorded",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )
        m = (
            (~_is_missing_like(hema, miss_codes))
            & (hema == 1)
            & _is_missing_like(hema_loc, miss_codes)
        )
        _add_rule(
            rows,
            "Parent-child Logic",
            "scalp_hematoma=1 but hematoma_location missing",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if hema is not None and hema_size is not None:
        miss = _is_missing_like(hema, miss_codes) | _is_missing_like(
            hema_size, miss_codes
        )
        m = (~miss) & (hema == 0) & (~_is_missing_like(hema_size, miss_codes))
        _add_rule(
            rows,
            "Parent-child Logic",
            "scalp_hematoma=0 but hematoma_size recorded",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )
        m = (
            (~_is_missing_like(hema, miss_codes))
            & (hema == 1)
            & _is_missing_like(hema_size, miss_codes)
        )
        _add_rule(
            rows,
            "Parent-child Logic",
            "scalp_hematoma=1 but hematoma_size missing",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # Palpable skull fracture -> depressed
    psf = _safe_col(df, "palpable_skull_fracture")
    psf_dep = _safe_col(df, "palpable_skull_fracture_depressed")

    if psf is not None and psf_dep is not None:
        miss = _is_missing_like(psf, miss_codes) | _is_missing_like(psf_dep, miss_codes)
        m = (~miss) & (psf == 0) & (psf_dep == 1)
        _add_rule(
            rows,
            "Parent-child Logic",
            "palpable_skull_fracture=0 but depressed=1",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )
        m = (
            (~_is_missing_like(psf, miss_codes))
            & (psf == 1)
            & _is_missing_like(psf_dep, miss_codes)
        )
        _add_rule(
            rows,
            "Parent-child Logic",
            "palpable_skull_fracture=1 but depressed missing",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # Neuro deficit -> subtypes
    nd = _safe_col(df, "neuro_deficit")
    nd_cols = [
        c
        for c in [
            "neuro_deficit_motor",
            "neuro_deficit_sensory",
            "neuro_deficit_cranial",
            "neuro_deficit_reflex",
            "neuro_deficit_other",
        ]
        if c in df.columns
    ]
    if nd is not None and nd_cols:
        miss_nd = _is_missing_like(nd, miss_codes)
        any_sub = (df[nd_cols] == 1).any(axis=1)

        m = (~miss_nd) & (nd == 0) & any_sub
        _add_rule(
            rows,
            "Parent-child Logic",
            "neuro_deficit=0 but a specific deficit is marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

        m = (~miss_nd) & (nd == 1) & (~any_sub)
        _add_rule(
            rows,
            "Parent-child Logic",
            "neuro_deficit=1 but no specific deficit marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # Other substantial injury -> specific injuries
    osi = _safe_col(df, "other_substantial_injury")
    osi_cols = [
        c
        for c in [
            "other_injury_extremity",
            "other_injury_or_laceration",
            "other_injury_cspine",
            "other_injury_chest_back_flank",
            "other_injury_abdomen",
            "other_injury_pelvis",
            "other_injury_other",
        ]
        if c in df.columns
    ]
    if osi is not None and osi_cols:
        miss_osi = _is_missing_like(osi, miss_codes)
        any_osi = (df[osi_cols] == 1).any(axis=1)

        m = (~miss_osi) & (osi == 0) & any_osi
        _add_rule(
            rows,
            "Parent-child Logic",
            "other_substantial_injury=0 but specific injury marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

        m = (~miss_osi) & (osi == 1) & (~any_osi)
        _add_rule(
            rows,
            "Parent-child Logic",
            "other_substantial_injury=1 but no specific injury marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # Trauma above clavicles -> locations
    clav = _safe_col(df, "trauma_above_clavicles")
    clav_cols = [
        c
        for c in [
            "trauma_face",
            "trauma_neck",
            "trauma_scalp_frontal",
            "trauma_scalp_occipital",
            "trauma_scalp_parietal",
            "trauma_scalp_temporal",
        ]
        if c in df.columns
    ]
    if clav is not None and clav_cols:
        miss_clav = _is_missing_like(clav, miss_codes)
        any_loc = (df[clav_cols] == 1).any(axis=1)

        m = (~miss_clav) & (clav == 0) & any_loc
        _add_rule(
            rows,
            "Parent-child Logic",
            "trauma_above_clavicles=0 but location marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

        m = (~miss_clav) & (clav == 1) & (~any_loc)
        _add_rule(
            rows,
            "Parent-child Logic",
            "trauma_above_clavicles=1 but no location marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # CT planned -> reasons
    ct_planned = _safe_col(df, "ct_planned")
    ct_reason_cols = [
        c
        for c in [
            "ct_reason_age",
            "ct_reason_amnesia",
            "ct_reason_ams",
            "ct_reason_clinical_skull_fracture",
            "ct_reason_headache",
            "ct_reason_scalp_hematoma",
            "ct_reason_loc",
            "ct_reason_mechanism",
            "ct_reason_neuro_deficit",
            "ct_reason_referring_md",
            "ct_reason_parent_request",
            "ct_reason_trauma_team",
            "ct_reason_seizure",
            "ct_reason_vomiting",
            "ct_reason_xray_skull_fracture",
            "ct_reason_other",
        ]
        if c in df.columns
    ]
    if ct_planned is not None and ct_reason_cols:
        miss_ctp = _is_missing_like(ct_planned, miss_codes)
        any_reason = (df[ct_reason_cols] == 1).any(axis=1)

        m = (~miss_ctp) & (ct_planned == 0) & any_reason
        _add_rule(
            rows,
            "Parent-child Logic",
            "ct_planned=0 but a CT reason is marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

        m = (~miss_ctp) & (ct_planned == 1) & (~any_reason)
        _add_rule(
            rows,
            "Parent-child Logic",
            "ct_planned=1 but no CT reason is marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # CT sedation -> subreasons
    ct_sed = _safe_col(df, "ct_sedation")
    ct_sed_cols = [
        c
        for c in [
            "ct_sedation_agitated",
            "ct_sedation_age",
            "ct_sedation_tech_request",
            "ct_sedation_other",
        ]
        if c in df.columns
    ]
    if ct_sed is not None and ct_sed_cols:
        miss_s = _is_missing_like(ct_sed, miss_codes)
        any_s = (df[ct_sed_cols] == 1).any(axis=1)

        m = (~miss_s) & (ct_sed == 0) & any_s
        _add_rule(
            rows,
            "Parent-child Logic",
            "ct_sedation=0 but sedation detail marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

        m = (~miss_s) & (ct_sed == 1) & (~any_s)
        _add_rule(
            rows,
            "Parent-child Logic",
            "ct_sedation=1 but no sedation detail marked",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    # Ranges & Coding
    if age_months is not None:
        miss = _is_missing_like(age_months, miss_codes)
        m = (~miss) & ((age_months < 0) | (age_months > 240))
        _add_rule(
            rows,
            "Ranges & Coding",
            "age_months outside [0, 240]",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if age_years is not None:
        miss = _is_missing_like(age_years, miss_codes)
        m = (~miss) & ((age_years < 0) | (age_years > 18))
        _add_rule(
            rows,
            "Ranges & Coding",
            "age_years outside [0, 18]",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if gcs_eye is not None:
        miss = _is_missing_like(gcs_eye, miss_codes)
        m = (~miss) & ((gcs_eye < 1) | (gcs_eye > 4))
        _add_rule(
            rows,
            "Ranges & Coding",
            "gcs_eye outside [1, 4]",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if gcs_verbal is not None:
        miss = _is_missing_like(gcs_verbal, miss_codes)
        m = (~miss) & ((gcs_verbal < 1) | (gcs_verbal > 5))
        _add_rule(
            rows,
            "Ranges & Coding",
            "gcs_verbal outside [1, 5]",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if gcs_motor is not None:
        miss = _is_missing_like(gcs_motor, miss_codes)
        m = (~miss) & ((gcs_motor < 1) | (gcs_motor > 6))
        _add_rule(
            rows,
            "Ranges & Coding",
            "gcs_motor outside [1, 6]",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    if gcs_total is not None:
        miss = _is_missing_like(gcs_total, miss_codes)
        m = (~miss) & ((gcs_total < 3) | (gcs_total > 15))
        _add_rule(
            rows,
            "Ranges & Coding",
            "gcs_total outside [3, 15]",
            m,
            n_rows,
            df=df,
            max_examples=max_examples,
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["n_inconsistent", "group", "rule"], ascending=[False, True, True]
    )
    out = out.reset_index(drop=True)
    return out


### Column Reduction

def find_constant_columns(df):
    """Return columns with <=1 unique non-missing value."""
    constant = []
    for c in df.columns:
        nunique = df[c].dropna().nunique()
        if nunique <= 1:
            constant.append(c)
    return constant


def find_high_missing_columns(df, threshold=0.95):
    """Return columns with missing proportion > threshold."""
    miss = df.isna().mean()
    return miss[miss > threshold].index.tolist()


def find_ct_finding_columns(df):
    """Return CT finding columns (ct_finding_*)."""
    return [c for c in df.columns if c.startswith("ct_finding_")]


def build_drop_plan(
    df,
    timepoint="pre_ct",
    label_col="citbi",
    id_cols=None,
    drop_ethnicity=True,
    high_missing_threshold=0.95,
    keep_age="age_years",
    keep_gcs="gcs_total",
):
    """
    Build a column drop plan with reasons.
    Returns a DataFrame with: column, category, reason.
    """
    if id_cols is None:
        id_cols = ["patient_id"]

    plan = []

    def add(cols, category, reason):
        for c in cols:
            if c in df.columns:
                plan.append(
                    {
                        "column": c,
                        "category": category,
                        "reason": reason,
                    }
                )

    # 0) Never drop label (but we may exclude it from predictors later)
    protected = set([label_col])

    # 1) ID-like columns (usually excluded from modeling)
    add([c for c in id_cols if c not in protected], "id", "identifier")

    # 2) Optional: ethnicity (you said you chose to drop it)
    if drop_ethnicity and "ethnicity" in df.columns and "ethnicity" not in protected:
        add(["ethnicity"], "sensitive_or_unhelpful", "dropped by choice")

    # 3) Leakage / post-outcome / post-imaging columns
    if timepoint == "pre_ct":
        ct_cols = [
            "ct_done",
            "ct_done_in_ed",
            "tbi_on_ct",
        ] + find_ct_finding_columns(df)

        outcome_component_cols = [
            "death_due_to_tbi",
            "hospitalized_head_injury",
            "hospitalized_head_injury_with_tbi_on_ct",
            "intubated_over_24h_for_head_injury",
            "neurosurgery",
        ]

        # Some datasets include more post-ED columns; keep this conservative.
        add(
            [c for c in ct_cols if c not in protected],
            "leakage_post_ct",
            "CT/imaging result (post-decision)",
        )
        add(
            [c for c in outcome_component_cols if c not in protected],
            "leakage_outcomes",
            "downstream outcome/treatment component",
        )

    # 4) Redundant encodings (keep one representation)
    # Age: keep one scale, drop the others if present
    age_family = ["age_months", "age_years", "age_two_plus"]
    if keep_age in age_family:
        drop_age = [c for c in age_family if c != keep_age]
        add(
            [c for c in drop_age if c not in protected],
            "redundant",
            "redundant age encoding",
        )

    # GCS: keep either total or components; drop group by default if keeping total
    gcs_components = ["gcs_eye", "gcs_verbal", "gcs_motor"]
    gcs_family = gcs_components + ["gcs_total", "gcs_group"]

    if keep_gcs == "gcs_total":
        drop_gcs = [c for c in gcs_family if c != "gcs_total"]
        add(
            [c for c in drop_gcs if c not in protected],
            "redundant",
            "redundant GCS encoding (keep gcs_total)",
        )
    elif keep_gcs == "components":
        drop_gcs = ["gcs_total", "gcs_group"]
        add(
            [c for c in drop_gcs if c not in protected],
            "redundant",
            "redundant GCS encoding (keep components)",
        )

    # 5) Constant columns
    const_cols = [c for c in find_constant_columns(df) if c not in protected]
    add(const_cols, "no_variation", "constant / no variation")

    # 6) Extremely high missingness columns
    high_miss_cols = [
        c
        for c in find_high_missing_columns(df, threshold=high_missing_threshold)
        if c not in protected
    ]
    add(
        high_miss_cols,
        "high_missing",
        "missingness above threshold",
    )

    plan_df = pd.DataFrame(plan).drop_duplicates(subset=["column"])

    # Make sure we never drop protected columns
    if not plan_df.empty:
        plan_df = plan_df[~plan_df["column"].isin(protected)].copy()
        plan_df = plan_df.sort_values(["category", "column"]).reset_index(drop=True)

    return plan_df


def apply_drop_plan(df, plan_df):
    """
    Drop columns according to plan_df.
    Returns: df_out, dropped_cols.
    """
    if plan_df is None or plan_df.empty:
        return df.copy(), []

    cols = plan_df["column"].tolist()
    df_out = df.drop(columns=cols, errors="ignore").copy()
    return df_out, cols


def summarize_drop_plan(plan_df):
    """
    Summarize drop plan by category.
    Returns: summary_df, details_df.
    """
    if plan_df is None or plan_df.empty:
        summary = pd.DataFrame(columns=["category", "n_cols"])
        details = pd.DataFrame(columns=["category", "column", "reason"])
        return summary, details

    summary = (
        plan_df.groupby("category")["column"]
        .count()
        .reset_index()
        .rename(columns={"column": "n_cols"})
        .sort_values("n_cols", ascending=False)
        .reset_index(drop=True)
    )

    details = plan_df[["category", "column", "reason"]].copy()
    return summary, details

### Key Rates for comparison to Kuppermann et al. (2009)

def _rate(numer, denom):
    if denom == 0:
        return np.nan
    return numer / denom


def _summarize_rates(df):
    """
    Compute key outcome rates for one dataframe slice.
    """
    n = len(df)

    ct_done = df["ct_done"].eq(1)
    n_ct = int(ct_done.sum())

    tbi_on_ct = df["tbi_on_ct"].eq(1)
    citbi = df["citbi"].eq(1)
    neurosurgery = df["neurosurgery"].eq(1)

    out = {
        "n": n,
        "n_ct_done": n_ct,
        "ct_done_rate": _rate(n_ct, n),
        "tbi_on_ct_rate_among_ct": _rate(
            int((ct_done & tbi_on_ct).sum()),
            n_ct,
        ),
        "citbi_rate": _rate(int(citbi.sum()), n),
        "neurosurgery_rate": _rate(int(neurosurgery.sum()), n),
    }
    return out


def compute_key_rates_for_report(df_renamed):
    """
    Create key-rate tables overall, by age group, and for GCS 14-15 subset.
    """
    df = df_renamed.copy()

    age_u2 = df["age_two_plus"] < 2
    gcs_14_15 = df["gcs_total"].isin([14, 15])

    strata = {
        "All (full dataset)": df,
        "Age <2 (full dataset)": df.loc[age_u2],
        "Age ≥2 (full dataset)": df.loc[~age_u2],
        "All (GCS 14-15)": df.loc[gcs_14_15],
        "Age <2 (GCS 14-15)": df.loc[age_u2 & gcs_14_15],
        "Age ≥2 (GCS 14-15)": df.loc[(~age_u2) & gcs_14_15],
    }

    rows = []
    for name, dfi in strata.items():
        s = _summarize_rates(dfi)
        s["stratum"] = name
        rows.append(s)

    out = pd.DataFrame(rows).set_index("stratum")

    # pretty percent columns for display
    pct_cols = [
        "ct_done_rate",
        "tbi_on_ct_rate_among_ct",
        "citbi_rate",
        "neurosurgery_rate",
    ]
    out_pct = out.copy()
    for c in pct_cols:
        out_pct[c] = (100 * out_pct[c]).round(2)

    return out, out_pct


def compare_with_kuppermann_2009(df_renamed):
    """
    Compare my GCS 14-15 subset to Kuppermann et al. (2009) Table 1 outcomes.

    Notes:
    - Paper reports TBI on CT as n/N among CT-imaged patients.
    - Paper reports ciTBI and neurosurgery as n/N among enrolled patients.
    """
    _, out_pct = compute_key_rates_for_report(df_renamed)

    mine_u2 = out_pct.loc["Age <2 (GCS 14-15)"]
    mine_o2 = out_pct.loc["Age ≥2 (GCS 14-15)"]

    paper = pd.DataFrame(
        {
            "paper_tbi_on_ct_%": [8.1, 9.8, 4.1, 5.2],
            "paper_citbi_%": [0.9, 1.1, 0.9, 1.0],
            "paper_neurosurgery_%": [0.2, 0.2, 0.1, 0.2],
        },
        index=[
            "Age <2 (Derivation)",
            "Age <2 (Validation)",
            "Age ≥2 (Derivation)",
            "Age ≥2 (Validation)",
        ],
    )

    mine = pd.DataFrame(
        {
            "my_ct_done_%": [mine_u2["ct_done_rate"], mine_o2["ct_done_rate"]],
            "my_tbi_on_ct_among_ct_%": [
                mine_u2["tbi_on_ct_rate_among_ct"],
                mine_o2["tbi_on_ct_rate_among_ct"],
            ],
            "my_citbi_%": [mine_u2["citbi_rate"], mine_o2["citbi_rate"]],
            "my_neurosurgery_%": [
                mine_u2["neurosurgery_rate"],
                mine_o2["neurosurgery_rate"],
            ],
        },
        index=["Age <2 (my data, GCS 14-15)", "Age ≥2 (my data, GCS 14-15)"],
    )

    return mine, paper
