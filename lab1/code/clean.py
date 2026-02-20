from __future__ import annotations

from pathlib import Path
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Sequence, Optional, Callable, Any

import pandas as pd
import numpy as np


# =========================================================
# 0) Column name normalization helpers
# =========================================================

def _snake_case(s: str) -> str:
    s2 = str(s).strip()
    s2 = re.sub(r"[^\w]+", "_", s2, flags=re.UNICODE)
    s2 = re.sub(r"_+", "_", s2).strip("_")
    return s2.lower()

def _make_unique(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out


# ============================================================
# 1) Mapping 
# ============================================================

MEANINGFUL_RENAME: dict[str, str] = {
    "patnum": "patient_id",
    "empltype": "clinician_role",
    "certification": "clinician_certification",

    "injurymech": "injury_mechanism",
    "high_impact_injsev": "injury_mechanism_severity",

    "amnesia_verb": "amnesia_for_event",
    "locseparate": "loss_of_consciousness_history",
    "loclen": "loss_of_consciousness_duration",

    "seiz": "post_traumatic_seizure",
    "seizoccur": "seizure_timing",
    "seizlen": "seizure_duration",

    "actnorm": "parent_reports_acting_normal",

    "ha_verb": "headache_at_ed_eval",
    "haseverity": "headache_severity",
    "hastart": "headache_onset_time",

    "vomit": "vomiting_anytime_after_injury",
    "vomitnbr": "vomiting_episode_count",
    "vomitstart": "vomiting_onset_time",
    "vomitlast": "vomiting_last_episode_time_before_ed",

    "dizzy": "dizziness_at_ed_eval",

    "intubated": "eval_after_intubation",
    "paralyzed": "eval_after_paralysis",
    "sedated": "eval_after_sedation",

    "gcseye": "gcs_eye",
    "gcsverbal": "gcs_verbal",
    "gcsmotor": "gcs_motor",
    "gcstotal": "gcs_total",
    "gcsgroup": "gcs_14_15_group",

    "ams": "altered_mental_status",
    "amsagitated": "ams_agitated",
    "amssleep": "ams_sleepy",
    "amsslow": "ams_slow_to_respond",
    "amsrepeat": "ams_repetitive_questions",
    "amsoth": "ams_other",

    "sfxpalp": "palpable_skull_fracture",
    "sfxpalpdepress": "palpable_skull_fracture_depressed",

    "fontbulg": "bulging_fontanelle",

    "sfxbas": "basilar_skull_fracture_signs",
    "sfxbashem": "basilar_hemotympanum",
    "sfxbasoto": "basilar_csf_otorrhea",
    "sfxbasper": "basilar_raccoon_eyes",
    "sfxbasret": "basilar_battles_sign",
    "sfxbasrhi": "basilar_csf_rhinorrhea",

    "hema": "scalp_hematoma_or_swelling",
    "hemaloc": "scalp_hematoma_location",
    "hemasize": "scalp_hematoma_size",

    "clav": "trauma_above_clavicles_any",
    "clavface": "trauma_above_clavicles_face",
    "clavneck": "trauma_above_clavicles_neck",
    "clavfro": "trauma_above_clavicles_scalp_frontal",
    "clavocc": "trauma_above_clavicles_scalp_occipital",
    "clavpar": "trauma_above_clavicles_scalp_parietal",
    "clavtem": "trauma_above_clavicles_scalp_temporal",

    "neurod": "neuro_deficit_non_mental_status",
    "neurodmotor": "neuro_deficit_motor",
    "neurodsensory": "neuro_deficit_sensory",
    "neurodcranial": "neuro_deficit_cranial_nerve_or_pupils",
    "neurodreflex": "neuro_deficit_reflexes",
    "neurodoth": "neuro_deficit_other",

    "osi": "other_substantial_injury_non_head",
    "osiextremity": "osi_extremity",
    "osicut": "osi_laceration_or_repair_in_or",
    "osicspine": "osi_c_spine",
    "osiflank": "osi_chest_back_flank",
    "osiabdomen": "osi_intra_abdominal",
    "osipelvis": "osi_pelvis",
    "osioth": "osi_other",

    "drugs": "suspected_alcohol_or_drug_intoxication",

    "ctform1": "imaging_planned_on_form",

    "indage": "ct_indication_young_age",
    "indamnesia": "ct_indication_amnesia",
    "indams": "ct_indication_decreased_mental_status",
    "indclinsfx": "ct_indication_clinical_skull_fracture",
    "indha": "ct_indication_headache",
    "indhema": "ct_indication_scalp_hematoma",
    "indloc": "ct_indication_loss_of_consciousness",
    "indmech": "ct_indication_mechanism",
    "indneurod": "ct_indication_neuro_deficit",
    "indrqstmd": "ct_indication_referring_md_request",
    "indrqstparent": "ct_indication_parent_request_or_anxiety",
    "indrqsttrauma": "ct_indication_trauma_team_request",
    "indseiz": "ct_indication_seizure",
    "indvomit": "ct_indication_vomiting",
    "indxraysfx": "ct_indication_skull_fracture_on_xray",
    "indoth": "ct_indication_other",

    "ctsed": "ct_sedation",
    "ctsedagitate": "ct_sedation_reason_agitation",
    "ctsedage": "ct_sedation_reason_young_age",
    "ctsedrqst": "ct_sedation_reason_ct_tech_request",
    "ctsedoth": "ct_sedation_reason_other",

    "ageinmonth": "age_months",
    "ageinyears": "age_years",
    "agetwoplus": "age_group_under2_vs_2plus",
    "gender": "sex",
    "ethnicity": "ethnicity",
    "race": "race",

    "observed": "observed_in_ed_after_initial_eval",
    "eddisposition": "ed_disposition",
    "ctdone": "head_ct_done_anywhere",
    "edct": "head_ct_done_in_ed",
    "posct": "tbi_on_ct",

    "finding1": "ct_finding_cerebellar_hemorrhage",
    "finding2": "ct_finding_cerebral_contusion",
    "finding3": "ct_finding_cerebral_edema",
    "finding4": "ct_finding_intracerebral_hemorrhage_or_hematoma",
    "finding5": "ct_finding_skull_diastasis",
    "finding6": "ct_finding_epidural_hematoma",
    "finding7": "ct_finding_extra_axial_hematoma",
    "finding8": "ct_finding_intraventricular_hemorrhage",
    "finding9": "ct_finding_midline_shift",
    "finding10": "ct_finding_pneumocephalus",
    "finding11": "ct_finding_skull_fracture",
    "finding12": "ct_finding_subarachnoid_hemorrhage",
    "finding13": "ct_finding_subdural_hematoma",
    "finding14": "ct_finding_traumatic_infarction",
    "finding20": "ct_finding_diffuse_axonal_injury",
    "finding21": "ct_finding_herniation",
    "finding22": "ct_finding_shear_injury",
    "finding23": "ct_finding_sigmoid_sinus_thrombosis",

    "deathtbi": "death_due_to_tbi",
    "hosphead": "hospitalized_2plus_nights_due_to_head_injury",
    "hospheadposct": "hospitalized_2plus_nights_head_injury_and_tbi_on_ct",
    "intub24head": "intubated_over_24h_for_head_trauma",
    "neurosurgery": "neurosurgery_performed",
    "posintfinal": "clinically_important_tbi",
}


# =========================================================
# 2) Minimal printing + anomaly tracking
# =========================================================

@dataclass
class AnomTracker:
    counts: Dict[str, int] = field(default_factory=dict)
    masks: Dict[str, pd.Series] = field(default_factory=dict)

    def add(self, name: str, mask: pd.Series):
        mask = mask.fillna(False)
        n = int(mask.sum())
        self.counts[name] = self.counts.get(name, 0) + n
        if name in self.masks:
            self.masks[name] = self.masks[name] | mask
        else:
            self.masks[name] = mask

    def combined_mask(self, index: pd.Index) -> pd.Series:
        if not self.masks:
            return pd.Series(False, index=index)
        cm = None
        for m in self.masks.values():
            cm = m if cm is None else (cm | m)
        return cm.fillna(False).reindex(index, fill_value=False)


def _ensure_cols(df: pd.DataFrame, cols: Sequence[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

def _coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _coerce_int(df: pd.DataFrame, col: str) -> pd.Series:
    return _coerce_int_series(df[col])

def coerce_int_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = _coerce_int(df, c)
    return df

def _set_na_by_codes(s: pd.Series, na_codes: Iterable[int]) -> pd.Series:
    s2 = s.copy()
    for code in na_codes:
        s2 = s2.mask(s2 == code, pd.NA)
    return s2

def _yn_to_bool(s: pd.Series, yes: int = 1, no: int = 0) -> pd.Series:
    s = s.astype("Int64")
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out = out.mask(s == no, False)
    out = out.mask(s == yes, True)
    return out


# =========================================================
# 2.5) Consistency
# =========================================================
def _pc_rule_only_setna_when_parent_no_or_na(
    df: pd.DataFrame,
    *,
    tr: AnomTracker,
    parent_no_or_na: pd.Series,
    child_cols: List[str],
    warn_name: str,
    do_print: bool,
    set_na: bool,
) -> pd.DataFrame:
    if not child_cols:
        return df

    child_any_filled = df[child_cols].notna().any(axis=1)
    bad = parent_no_or_na.fillna(True) & child_any_filled

    _check(bad, warn_name, tr=tr, do_print=do_print)
    if set_na and bad.any():
        df.loc[bad, child_cols] = pd.NA
    return df


# =========================================================
# 3) Step runner + printing
# =========================================================

@dataclass
class StepCtrl:
    enabled: bool = True
    do_print: bool = True
    drop: bool = False
    set_na: bool = True  

def _print_line(msg: str, *, do_print: bool):
    if do_print:
        print(msg)

def _check(mask: pd.Series, name: str, *, tr: Optional[AnomTracker], do_print: bool) -> int:
    mask = mask.fillna(False)
    n = int(mask.sum())
    if tr is not None:
        tr.add(name, mask)
    return n

def _print_step_report(step_name: str, *, tr: AnomTracker, index: pd.Index, do_print: bool):
    if not do_print:
        return
    issues = int(tr.combined_mask(index).sum())
    _print_line(f"[{step_name}] issues rows={issues}", do_print=True)
    for k, n in tr.counts.items():
        if n > 0:
            _print_line(f"  [WARN] {k}: {n}", do_print=True)

def _step_notes(name: str, lines: List[str], *, do_print: bool):
    if not do_print:
        return
    if not lines:
        _print_line(f"[{name}] issues rows=0", do_print=True)
        return
    _print_line(f"[{name}] -> notes:", do_print=True)
    for ln in lines:
        _print_line(f"  {ln}", do_print=True)

def run_step(
    name: str,
    df: pd.DataFrame,
    fn: Callable[..., Tuple[pd.DataFrame, Optional[pd.Series]]],
    *,
    ctrl: StepCtrl,
    source: bool = False,
    df_or_path: Any = None,
    **kwargs
) -> pd.DataFrame:
    if not ctrl.enabled:
        return df

    if ctrl.do_print:
        print("")
        _print_line(f"[{name}] rows_before={len(df)}", do_print=True)

    if source:
        df2, drop_mask = fn(df_or_path, do_print=ctrl.do_print, **kwargs)
    else:
        df2, drop_mask = fn(df, do_print=ctrl.do_print, **kwargs)

    if ctrl.drop and drop_mask is not None:
        drop_mask = drop_mask.fillna(False).reindex(df2.index, fill_value=False)
        n = int(drop_mask.sum())
        if ctrl.do_print:
            _print_line(f"[{name}] [DROP] dropping {n} rows", do_print=True)
        if n > 0:
            df2 = df2.loc[~drop_mask].copy()
        if ctrl.do_print:
            _print_line(f"[{name}] rows_after={len(df2)}", do_print=True)
    else:
        if ctrl.do_print:
            _print_line(f"[{name}] rows_after={len(df2)}", do_print=True)

    return df2


# =========================================================
# 4) Validate user config
# =========================================================

BOOL_COLS_YN: List[str] = []  # filled later (keep linter quiet)

def validate_user_config(*, do_print: bool = True):
    notes: List[str] = []

    bad_keys = [k for k in MEANINGFUL_RENAME.keys() if not k or k != _snake_case(k)]
    if bad_keys:
        notes.append(f"[WARN] MEANINGFUL_RENAME has non-snake keys: {bad_keys[:10]}")

    vals = list(MEANINGFUL_RENAME.values())
    vals_sc = [_snake_case(v) for v in vals]
    dup = pd.Series(vals_sc).duplicated(keep=False)
    if dup.any():
        dups = sorted(set(pd.Series(vals_sc)[dup].tolist()))
        notes.append(f"[WARN] MEANINGFUL_RENAME has duplicate target names after snake: {dups[:20]}")

    _step_notes("Validate config", notes, do_print=do_print)


# =========================================================
# Step01: load + rename
# =========================================================

def step01_load_and_rename(
    df_or_path: Any,
    *,
    rename_map: Dict[str, str] = MEANINGFUL_RENAME,
    encoding: Optional[str] = None,
    do_print: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df_or_path
    if isinstance(df_or_path, (str, Path)):
        p = Path(df_or_path)
        df = pd.read_csv(p) if encoding is None else pd.read_csv(p, encoding=encoding)

    original_cols = list(df.columns)
    snake_cols = _make_unique([_snake_case(c) for c in original_cols])

    df_snake = df.copy()
    df_snake.columns = snake_cols

    meaningful_cols: List[str] = []
    unmapped = 0
    for c in snake_cols:
        if c in rename_map:
            meaningful_cols.append(_snake_case(rename_map[c]))
        else:
            meaningful_cols.append(c)
            unmapped += 1

    meaningful_cols = _make_unique(meaningful_cols)
    df_out = df_snake.copy()
    df_out.columns = meaningful_cols

    core = [
        "patient_id",
        "age_months", "age_years", "age_group_under2_vs_2plus",
        "gcs_total", "gcs_eye", "gcs_verbal", "gcs_motor",
        "clinically_important_tbi",
        "head_ct_done_anywhere", "tbi_on_ct",
    ]
    missing_core = [c for c in core if c not in df_out.columns]

    notes: List[str] = []
    if unmapped > 0:
        notes.append(f"[INFO] unmapped columns: {unmapped} / {len(snake_cols)}")
    if missing_core:
        notes.append(f"[WARN] missing core fields: {len(missing_core)} -> {missing_core}")

    if "clinically_important_tbi" in df_out.columns:
        citbi = pd.to_numeric(df_out["clinically_important_tbi"], errors="coerce")
        df_out.attrs["_citbi_na_mask"] = citbi.isna()
    else:
        df_out.attrs["_citbi_na_mask"] = None

    _step_notes("Step01 load+rename", notes, do_print=do_print)
    return df_out, None


# =========================================================
# Step02: exclude_low_gcs early + drop ciTBI NA
# =========================================================

def step02_exclude_low_gcs_early(
    df: pd.DataFrame,
    *,
    drop_citbi_na: bool = True,
    do_print: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df.copy()

    exclude_low_gcs = pd.Series(False, index=df.index)

    if "gcs_14_15_group" in df.columns:
        grp = _coerce_int(df, "gcs_14_15_group")
        exclude_low_gcs = exclude_low_gcs | (grp == 1)

    if "gcs_total" in df.columns:
        gtot = pd.to_numeric(df["gcs_total"], errors="coerce")
        exclude_low_gcs = exclude_low_gcs | (gtot.notna() & (gtot < 14))

    df["exclude_low_gcs"] = exclude_low_gcs.astype("boolean")
    n_excl = int(df["exclude_low_gcs"].fillna(False).sum())

    citbi_na_mask = df.attrs.pop("_citbi_na_mask", None)
    citbi_missing_col = ("clinically_important_tbi" not in df.columns)

    if citbi_na_mask is None:
        citbi_na_mask = pd.Series(False, index=df.index)
        citbi_na_info_missing = True
        n_citbi_na = 0
    else:
        citbi_na_mask = citbi_na_mask.reindex(df.index, fill_value=False).fillna(False)
        citbi_na_info_missing = False
        n_citbi_na = int(citbi_na_mask.sum())

    if citbi_missing_col:
        df["analysis_eligible"] = (~df["exclude_low_gcs"].fillna(False)).astype("boolean")
        n_eligible = int(df["analysis_eligible"].fillna(False).sum())
    else:
        citbi = pd.to_numeric(df["clinically_important_tbi"], errors="coerce")
        df["analysis_eligible"] = ((~df["exclude_low_gcs"].fillna(False)) & citbi.notna()).astype("boolean")
        n_eligible = int(df["analysis_eligible"].fillna(False).sum())

    drop_mask = df["exclude_low_gcs"].fillna(False)
    if drop_citbi_na:
        drop_mask = drop_mask | citbi_na_mask
    n_drop = int(drop_mask.sum())

    notes: List[str] = []
    if citbi_missing_col:
        notes.append("[WARN] clinically_important_tbi column missing")
    if citbi_na_info_missing:
        notes.append("[WARN] _citbi_na_mask not found in df.attrs (Step01 may not have stored it)")
    if n_excl > 0:
        notes.append(f"[INFO] exclude_low_gcs = {n_excl}")
    if n_citbi_na > 0:
        notes.append(f"[INFO] clinically_important_tbi is NA = {n_citbi_na} (drop_citbi_na={drop_citbi_na})")
    if n_drop > 0:
        notes.append(f"[INFO] drop_mask total = {n_drop}")
    if len(df) > 0:
        frac = n_eligible / len(df)
        if n_eligible == 0:
            notes.append("[WARN] analysis_eligible = 0 (check rules / data)")
        elif frac < 0.1:
            notes.append(f"[WARN] analysis_eligible low: {n_eligible}/{len(df)} ({frac:.1%})")

    _step_notes("Step02 exclude_low_gcs early", notes, do_print=do_print)
    return df, drop_mask


# =========================================================
# Step03: basic schema
# =========================================================

def step03_basic_schema(df: pd.DataFrame, *, do_print: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df.copy()
    tr = AnomTracker()

    _ensure_cols(df, ["patient_id"])
    df["patient_id"] = _coerce_int(df, "patient_id")

    m1 = df["patient_id"].isna()
    m2 = df["patient_id"].duplicated(keep=False)

    _check(m1, "patient_id is NA", tr=tr, do_print=do_print)
    _check(m2, "duplicate patient_id (flag only)", tr=tr, do_print=do_print)

    _print_step_report("Step03 basic schema", tr=tr, index=df.index, do_print=do_print)
    return df, tr.combined_mask(df.index)


# =========================================================
# Step04: Normalize missing codes
# =========================================================

NA_CODE_MAP: Dict[str, Tuple[int, ...]] = {
    "loss_of_consciousness_duration": (92,),
    "seizure_timing": (92,),
    "seizure_duration": (92,),
    "headache_severity": (92,),
    "headache_onset_time": (92,),
    "vomiting_episode_count": (92,),
    "vomiting_onset_time": (92,),
    "vomiting_last_episode_time_before_ed": (92,),
    "ams_agitated": (92,),
    "ams_sleepy": (92,),
    "ams_slow_to_respond": (92,),
    "ams_repetitive_questions": (92,),
    "ams_other": (92,),
    "palpable_skull_fracture_depressed": (92,),
    "basilar_hemotympanum": (92,),
    "basilar_csf_otorrhea": (92,),
    "basilar_raccoon_eyes": (92,),
    "basilar_battles_sign": (92,),
    "basilar_csf_rhinorrhea": (92,),
    "scalp_hematoma_location": (92,),
    "scalp_hematoma_size": (92,),
    "trauma_above_clavicles_face": (92,),
    "trauma_above_clavicles_neck": (92,),
    "trauma_above_clavicles_scalp_frontal": (92,),
    "trauma_above_clavicles_scalp_occipital": (92,),
    "trauma_above_clavicles_scalp_parietal": (92,),
    "trauma_above_clavicles_scalp_temporal": (92,),
    "neuro_deficit_motor": (92,),
    "neuro_deficit_sensory": (92,),
    "neuro_deficit_cranial_nerve_or_pupils": (92,),
    "neuro_deficit_reflexes": (92,),
    "neuro_deficit_other": (92,),
    "osi_extremity": (92,),
    "osi_laceration_or_repair_in_or": (92,),
    "osi_c_spine": (92,),
    "osi_chest_back_flank": (92,),
    "osi_intra_abdominal": (92,),
    "osi_pelvis": (92,),
    "osi_other": (92,),
    "ct_indication_young_age": (92,),
    "ct_indication_amnesia": (92,),
    "ct_indication_decreased_mental_status": (92,),
    "ct_indication_clinical_skull_fracture": (92,),
    "ct_indication_headache": (92,),
    "ct_indication_scalp_hematoma": (92,),
    "ct_indication_loss_of_consciousness": (92,),
    "ct_indication_mechanism": (92,),
    "ct_indication_neuro_deficit": (92,),
    "ct_indication_referring_md_request": (92,),
    "ct_indication_parent_request_or_anxiety": (92,),
    "ct_indication_trauma_team_request": (92,),
    "ct_indication_seizure": (92,),
    "ct_indication_vomiting": (92,),
    "ct_indication_skull_fracture_on_xray": (92,),
    "ct_indication_other": (92,),
    "ct_sedation": (92,),
    "ct_sedation_reason_agitation": (92,),
    "ct_sedation_reason_young_age": (92,),
    "ct_sedation_reason_ct_tech_request": (92,),
    "ct_sedation_reason_other": (92,),
    "head_ct_done_in_ed": (92,),
    "tbi_on_ct": (92,),
    "ct_finding_cerebellar_hemorrhage": (92,),
    "ct_finding_cerebral_contusion": (92,),
    "ct_finding_cerebral_edema": (92,),
    "ct_finding_intracerebral_hemorrhage_or_hematoma": (92,),
    "ct_finding_skull_diastasis": (92,),
    "ct_finding_epidural_hematoma": (92,),
    "ct_finding_extra_axial_hematoma": (92,),
    "ct_finding_intraventricular_hemorrhage": (92,),
    "ct_finding_midline_shift": (92,),
    "ct_finding_pneumocephalus": (92,),
    "ct_finding_skull_fracture": (92,),
    "ct_finding_subarachnoid_hemorrhage": (92,),
    "ct_finding_subdural_hematoma": (92,),
    "ct_finding_traumatic_infarction": (92,),
    "ct_finding_diffuse_axonal_injury": (92,),
    "ct_finding_herniation": (92,),
    "ct_finding_shear_injury": (92,),
    "ct_finding_sigmoid_sinus_thrombosis": (92,),
}
PV_CODE_MAP: Dict[str, Tuple[int, ...]] = {
    "amnesia_for_event": (91,),
    "headache_at_ed_eval": (91,),
}

def step04_normalize_missing_codes(
    df: pd.DataFrame,
    *,
    do_print: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df.copy()
    tr = AnomTracker()

    cols_to_int = set(NA_CODE_MAP.keys()) | set(PV_CODE_MAP.keys())
    df = coerce_int_cols(df, cols_to_int)

    for col, codes in PV_CODE_MAP.items():
        if col in df.columns:
            df[f"{col}__is_preverbal_or_nonverbal"] = df[col].isin(list(codes))
            df[col] = _set_na_by_codes(df[col], codes)

    for col, codes in NA_CODE_MAP.items():
        if col in df.columns:
            df[col] = _set_na_by_codes(df[col], codes)

    for col, codes in PV_CODE_MAP.items():
        if col in df.columns:
            left = df[col].isin(list(codes))
            _check(left, f"PV not fully cleaned: {col} left {list(codes)}", tr=tr, do_print=do_print)

    for col, codes in NA_CODE_MAP.items():
        if col in df.columns:
            left = df[col].isin(list(codes))
            _check(left, f"NA not fully cleaned: {col} left {list(codes)}", tr=tr, do_print=do_print)

    _print_step_report("Step04 normalize missing codes", tr=tr, index=df.index, do_print=do_print)
    return df, None


# =========================================================
# Step05: Standardize yes/no fields into boolean columns
# =========================================================

BOOL_COLS_YN = [
    "amnesia_for_event",
    "post_traumatic_seizure",
    "parent_reports_acting_normal",
    "headache_at_ed_eval",
    "vomiting_anytime_after_injury",
    "dizziness_at_ed_eval",
    "eval_after_intubation",
    "eval_after_paralysis",
    "eval_after_sedation",
    "altered_mental_status",
    "ams_agitated",
    "ams_sleepy",
    "ams_slow_to_respond",
    "ams_repetitive_questions",
    "ams_other",
    "palpable_skull_fracture_depressed",
    "bulging_fontanelle",
    "basilar_skull_fracture_signs",
    "basilar_hemotympanum",
    "basilar_csf_otorrhea",
    "basilar_raccoon_eyes",
    "basilar_battles_sign",
    "basilar_csf_rhinorrhea",
    "scalp_hematoma_or_swelling",
    "trauma_above_clavicles_any",
    "trauma_above_clavicles_face",
    "trauma_above_clavicles_neck",
    "trauma_above_clavicles_scalp_frontal",
    "trauma_above_clavicles_scalp_occipital",
    "trauma_above_clavicles_scalp_parietal",
    "trauma_above_clavicles_scalp_temporal",
    "neuro_deficit_non_mental_status",
    "neuro_deficit_motor",
    "neuro_deficit_sensory",
    "neuro_deficit_cranial_nerve_or_pupils",
    "neuro_deficit_reflexes",
    "neuro_deficit_other",
    "other_substantial_injury_non_head",
    "osi_extremity",
    "osi_laceration_or_repair_in_or",
    "osi_c_spine",
    "osi_chest_back_flank",
    "osi_intra_abdominal",
    "osi_pelvis",
    "osi_other",
    "suspected_alcohol_or_drug_intoxication",
    "imaging_planned_on_form",
    "ct_indication_young_age",
    "ct_indication_amnesia",
    "ct_indication_decreased_mental_status",
    "ct_indication_clinical_skull_fracture",
    "ct_indication_headache",
    "ct_indication_scalp_hematoma",
    "ct_indication_loss_of_consciousness",
    "ct_indication_mechanism",
    "ct_indication_neuro_deficit",
    "ct_indication_referring_md_request",
    "ct_indication_parent_request_or_anxiety",
    "ct_indication_trauma_team_request",
    "ct_indication_seizure",
    "ct_indication_vomiting",
    "ct_indication_skull_fracture_on_xray",
    "ct_indication_other",
    "ct_sedation",
    "ct_sedation_reason_agitation",
    "ct_sedation_reason_young_age",
    "ct_sedation_reason_ct_tech_request",
    "ct_sedation_reason_other",
    "observed_in_ed_after_initial_eval",
    "head_ct_done_anywhere",
    "head_ct_done_in_ed",
    "tbi_on_ct",
    "ct_finding_cerebellar_hemorrhage",
    "ct_finding_cerebral_contusion",
    "ct_finding_cerebral_edema",
    "ct_finding_intracerebral_hemorrhage_or_hematoma",
    "ct_finding_skull_diastasis",
    "ct_finding_epidural_hematoma",
    "ct_finding_extra_axial_hematoma",
    "ct_finding_intraventricular_hemorrhage",
    "ct_finding_midline_shift",
    "ct_finding_pneumocephalus",
    "ct_finding_skull_fracture",
    "ct_finding_subarachnoid_hemorrhage",
    "ct_finding_subdural_hematoma",
    "ct_finding_traumatic_infarction",
    "ct_finding_diffuse_axonal_injury",
    "ct_finding_herniation",
    "ct_finding_shear_injury",
    "ct_finding_sigmoid_sinus_thrombosis",
    "death_due_to_tbi",
    "hospitalized_2plus_nights_due_to_head_injury",
    "hospitalized_2plus_nights_head_injury_and_tbi_on_ct",
    "intubated_over_24h_for_head_trauma",
    "neurosurgery_performed",
    "clinically_important_tbi",
]

def step05_make_boolean_columns(df: pd.DataFrame, *, do_print: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df.copy()
    tr = AnomTracker()

    for col in BOOL_COLS_YN:
        if col not in df.columns:
            continue
        df[col] = _coerce_int(df, col)
        invalid = ~(df[col].isin([0, 1]) | df[col].isna())
        _check(invalid, f"{col} invalid codes (expected 0/1/NA)", tr=tr, do_print=do_print)
        df[col + "__b"] = _yn_to_bool(df[col], yes=1, no=0)

    _print_step_report("Step05 make boolean columns", tr=tr, index=df.index, do_print=do_print)
    return df, tr.combined_mask(df.index)


# =========================================================
# Step06: Age fields + age group consistency
# =========================================================

def step06_age_clean(df: pd.DataFrame, *, do_print: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df.copy()
    tr = AnomTracker()

    for col in ["age_months", "age_years", "age_group_under2_vs_2plus"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    under2_from_months = (df["age_months"] < 24) if "age_months" in df.columns else pd.Series(pd.NA, index=df.index)
    under2_from_years = (df["age_years"] < 2) if "age_years" in df.columns else pd.Series(pd.NA, index=df.index)

    if "age_group_under2_vs_2plus" in df.columns:
        grp = pd.to_numeric(df["age_group_under2_vs_2plus"], errors="coerce")
        df["age_under2__coded"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        df.loc[grp == 1, "age_under2__coded"] = True
        df.loc[grp == 2, "age_under2__coded"] = False
        invalid = ~(grp.isin([1, 2]) | grp.isna())
        _check(invalid, "age_group_under2_vs_2plus invalid codes", tr=tr, do_print=do_print)
    else:
        df["age_under2__coded"] = pd.Series(pd.NA, index=df.index, dtype="boolean")

    df["age_under2"] = df["age_under2__coded"]
    df.loc[df["age_under2"].isna() & under2_from_months.notna(), "age_under2"] = under2_from_months.astype("boolean")
    df.loc[df["age_under2"].isna() & under2_from_years.notna(), "age_under2"] = under2_from_years.astype("boolean")

    if "age_months" in df.columns:
        _check(df["age_months"].notna() & (df["age_months"] < 0), "age_months < 0", tr=tr, do_print=do_print)
        _check(df["age_months"].notna() & (df["age_months"] > 240), "age_months > 240 (check units)", tr=tr, do_print=do_print)

    if "age_years" in df.columns:
        _check(df["age_years"].notna() & (df["age_years"] < 0), "age_years < 0", tr=tr, do_print=do_print)
        _check(df["age_years"].notna() & (df["age_years"] > 20), "age_years > 20 (check)", tr=tr, do_print=do_print)

    if "age_under2__coded" in df.columns and "age_months" in df.columns:
        disagree = df["age_under2__coded"].notna() & under2_from_months.notna() & (df["age_under2__coded"] != under2_from_months)
        _check(disagree, "age_under2 coded vs age_months disagree", tr=tr, do_print=do_print)

    _print_step_report("Step06 age clean", tr=tr, index=df.index, do_print=do_print)
    return df, tr.combined_mask(df.index)


# =========================================================
# Step07: parent child consistency
# =========================================================

def step07_parent_child_consistency(
    df: pd.DataFrame,
    *,
    do_print: bool = True,
    set_na: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df.copy()
    tr = AnomTracker()

    # ---------- LOC: history -> duration
    if "loss_of_consciousness_history" in df.columns and "loss_of_consciousness_duration" in df.columns:
        h = _coerce_int(df, "loss_of_consciousness_history")
        df["loss_of_consciousness_history"] = h
        parent_no_or_na = (h != 1) | h.isna()  # 0/2/NA 都视为非Yes
        df = _pc_rule_only_setna_when_parent_no_or_na(
            df, tr=tr,
            parent_no_or_na=parent_no_or_na,
            child_cols=["loss_of_consciousness_duration"],
            warn_name="LOC history!=Yes but duration filled (set NA)",
            do_print=do_print,
            set_na=set_na,
        )

    # ---------- Seizure: parent bool -> timing/duration
    if "post_traumatic_seizure__b" in df.columns:
        parent_no_or_na = (df["post_traumatic_seizure__b"] != True) | df["post_traumatic_seizure__b"].isna()
        for col in ["seizure_timing", "seizure_duration"]:
            if col in df.columns:
                df = _pc_rule_only_setna_when_parent_no_or_na(
                    df, tr=tr,
                    parent_no_or_na=parent_no_or_na,
                    child_cols=[col],
                    warn_name=f"{col} filled but post_traumatic_seizure!=Yes (set NA)",
                    do_print=do_print,
                    set_na=set_na,
                )

    # ---------- Headache: headache_at_ed_eval -> severity/onset
    if "headache_at_ed_eval" in df.columns:
        ha = _coerce_int(df, "headache_at_ed_eval")
        df["headache_at_ed_eval"] = ha
        parent_no_or_na = (ha != 1) | ha.isna()
        for col in ["headache_severity", "headache_onset_time"]:
            if col in df.columns:
                df = _pc_rule_only_setna_when_parent_no_or_na(
                    df, tr=tr,
                    parent_no_or_na=parent_no_or_na,
                    child_cols=[col],
                    warn_name=f"{col} filled but headache_at_ed_eval!=Yes (set NA)",
                    do_print=do_print,
                    set_na=set_na,
                )

    # ---------- Vomiting: parent bool -> detail fields
    if "vomiting_anytime_after_injury__b" in df.columns:
        parent_no_or_na = (df["vomiting_anytime_after_injury__b"] != True) | df["vomiting_anytime_after_injury__b"].isna()
        for col in ["vomiting_episode_count", "vomiting_onset_time", "vomiting_last_episode_time_before_ed"]:
            if col in df.columns:
                df = _pc_rule_only_setna_when_parent_no_or_na(
                    df, tr=tr,
                    parent_no_or_na=parent_no_or_na,
                    child_cols=[col],
                    warn_name=f"{col} filled but vomiting!=Yes (set NA)",
                    do_print=do_print,
                    set_na=set_na,
                )

    # ---------- AMS: altered_mental_status -> ams_*
    if "altered_mental_status" in df.columns:
        df["altered_mental_status"] = _coerce_int(df, "altered_mental_status")
        ams = df["altered_mental_status"]
        parent_no_or_na = (ams != 1) | ams.isna()
        child_cols = [c for c in ["ams_agitated", "ams_sleepy", "ams_slow_to_respond", "ams_repetitive_questions", "ams_other"] if c in df.columns]
        if child_cols:
            df = coerce_int_cols(df, child_cols)
            df = _pc_rule_only_setna_when_parent_no_or_na(
                df, tr=tr,
                parent_no_or_na=parent_no_or_na,
                child_cols=child_cols,
                warn_name="AMS parent!=Yes but ams_* filled (set NA)",
                do_print=do_print,
                set_na=set_na,
            )

    # ---------- Palpable skull fracture: parent -> depressed
    if "palpable_skull_fracture" in df.columns and "palpable_skull_fracture_depressed" in df.columns:
        psf = _coerce_int(df, "palpable_skull_fracture")
        df["palpable_skull_fracture"] = psf
        parent_no_or_na = (psf != 1) | psf.isna()  # 0/2/NA 视为非Yes
        df["palpable_skull_fracture_depressed"] = _coerce_int(df, "palpable_skull_fracture_depressed")
        df = _pc_rule_only_setna_when_parent_no_or_na(
            df, tr=tr,
            parent_no_or_na=parent_no_or_na,
            child_cols=["palpable_skull_fracture_depressed"],
            warn_name="PSF parent!=Yes but depressed filled (set NA)",
            do_print=do_print,
            set_na=set_na,
        )

    # ---------- Scalp hematoma: parent bool -> location/size
    if "scalp_hematoma_or_swelling__b" in df.columns:
        parent_no_or_na = (df["scalp_hematoma_or_swelling__b"] != True) | df["scalp_hematoma_or_swelling__b"].isna()
        for col in ["scalp_hematoma_location", "scalp_hematoma_size"]:
            if col in df.columns:
                df = _pc_rule_only_setna_when_parent_no_or_na(
                    df, tr=tr,
                    parent_no_or_na=parent_no_or_na,
                    child_cols=[col],
                    warn_name=f"{col} filled but scalp_hematoma_or_swelling!=Yes (set NA)",
                    do_print=do_print,
                    set_na=set_na,
                )

    # ---------- Basilar signs: parent 0/1 -> children
    if "basilar_skull_fracture_signs" in df.columns:
        b = _coerce_int(df, "basilar_skull_fracture_signs")
        df["basilar_skull_fracture_signs"] = b
        parent_no_or_na = (b != 1) | b.isna()
        child_cols = [
            "basilar_hemotympanum",
            "basilar_csf_otorrhea",
            "basilar_raccoon_eyes",
            "basilar_battles_sign",
            "basilar_csf_rhinorrhea",
        ]
        present_children = [c for c in child_cols if c in df.columns]
        if present_children:
            df = coerce_int_cols(df, present_children)
            df = _pc_rule_only_setna_when_parent_no_or_na(
                df, tr=tr,
                parent_no_or_na=parent_no_or_na,
                child_cols=present_children,
                warn_name="Basilar parent!=Yes but basilar_* filled (set NA)",
                do_print=do_print,
                set_na=set_na,
            )

    # ---------- Trauma above clavicles: parent bool -> region fields
    if "trauma_above_clavicles_any__b" in df.columns:
        parent_no_or_na = (df["trauma_above_clavicles_any__b"] != True) | df["trauma_above_clavicles_any__b"].isna()
        region_cols = [
            c for c in df.columns
            if c.startswith("trauma_above_clavicles_")
            and c != "trauma_above_clavicles_any"
            and not c.endswith("__b")
        ]
        if region_cols:
            df = _pc_rule_only_setna_when_parent_no_or_na(
                df, tr=tr,
                parent_no_or_na=parent_no_or_na,
                child_cols=region_cols,
                warn_name="trauma_above_clavicles_any!=Yes but trauma_above_clavicles_* filled (set NA)",
                do_print=do_print,
                set_na=set_na,
            )

    # ---------- Neuro deficit: parent bool -> sub fields
    if "neuro_deficit_non_mental_status__b" in df.columns:
        parent_no_or_na = (df["neuro_deficit_non_mental_status__b"] != True) | df["neuro_deficit_non_mental_status__b"].isna()
        sub_cols = [
            c for c in df.columns
            if c.startswith("neuro_deficit_")
            and c != "neuro_deficit_non_mental_status"
            and not c.endswith("__b")
        ]
        if sub_cols:
            df = _pc_rule_only_setna_when_parent_no_or_na(
                df, tr=tr,
                parent_no_or_na=parent_no_or_na,
                child_cols=sub_cols,
                warn_name="neuro_deficit_non_mental_status!=Yes but neuro_deficit_* filled (set NA)",
                do_print=do_print,
                set_na=set_na,
            )

    # ---------- OSI: parent bool -> osi_*
    if "other_substantial_injury_non_head__b" in df.columns:
        parent_no_or_na = (df["other_substantial_injury_non_head__b"] != True) | df["other_substantial_injury_non_head__b"].isna()
        osi_cols = [c for c in df.columns if c.startswith("osi_") and not c.endswith("__b")]
        if osi_cols:
            df = _pc_rule_only_setna_when_parent_no_or_na(
                df, tr=tr,
                parent_no_or_na=parent_no_or_na,
                child_cols=osi_cols,
                warn_name="OSI parent!=Yes but osi_* filled (set NA)",
                do_print=do_print,
                set_na=set_na,
            )

    # ---------------- CT consistency (planned/done/sedation reasons) ----------------

    if "imaging_planned_on_form__b" not in df.columns and "imaging_planned_on_form" in df.columns:
        df["imaging_planned_on_form__b"] = _yn_to_bool(_coerce_int(df, "imaging_planned_on_form"))

    if "head_ct_done_anywhere__b" not in df.columns and "head_ct_done_anywhere" in df.columns:
        df["head_ct_done_anywhere__b"] = _yn_to_bool(_coerce_int(df, "head_ct_done_anywhere"))

    # CT planned -> ct_indication_*
    if "imaging_planned_on_form__b" in df.columns:
        parent_no_or_na = (df["imaging_planned_on_form__b"] != True) | df["imaging_planned_on_form__b"].isna()

        ind_cols = [c for c in df.columns if c.startswith("ct_indication_") and not c.endswith("__b")]
        if ind_cols:
            df = coerce_int_cols(df, ind_cols)
            df = _pc_rule_only_setna_when_parent_no_or_na(
                df, tr=tr,
                parent_no_or_na=parent_no_or_na,
                child_cols=ind_cols,
                warn_name="CT planned!=Yes but ct_indication_* filled (set NA)",
                do_print=do_print,
                set_na=set_na,
            )

        # CT planned -> ct_sedation 
        sed_cols = [c for c in df.columns if c == "ct_sedation"]
        if sed_cols:
            df = coerce_int_cols(df, sed_cols)
            df = _pc_rule_only_setna_when_parent_no_or_na(
                df, tr=tr,
                parent_no_or_na=parent_no_or_na,
                child_cols=sed_cols,
                warn_name="CT planned!=Yes but ct_sedation* filled (set NA)",
                do_print=do_print,
                set_na=set_na,
            )

    # CT done anywhere -> tbi_on_ct + ct_finding_*
    if "head_ct_done_anywhere__b" in df.columns:
        parent_no_or_na = (df["head_ct_done_anywhere__b"] != True) | df["head_ct_done_anywhere__b"].isna()

        if "tbi_on_ct" in df.columns:
            df["tbi_on_ct"] = _coerce_int(df, "tbi_on_ct")
            df = _pc_rule_only_setna_when_parent_no_or_na(
                df, tr=tr,
                parent_no_or_na=parent_no_or_na,
                child_cols=["tbi_on_ct"],
                warn_name="CT done!=Yes but tbi_on_ct filled (set NA)",
                do_print=do_print,
                set_na=set_na,
            )

        finding_cols = [c for c in df.columns if c.startswith("ct_finding_") and not c.endswith("__b")]
        if finding_cols:
            df = coerce_int_cols(df, finding_cols)
            df = _pc_rule_only_setna_when_parent_no_or_na(
                df, tr=tr,
                parent_no_or_na=parent_no_or_na,
                child_cols=finding_cols,
                warn_name="CT done!=Yes but ct_finding_* filled (set NA)",
                do_print=do_print,
                set_na=set_na,
            )

    # ct_sedation (parent) -> ct_sedation_reason_*
    if "ct_sedation" in df.columns:
        df["ct_sedation"] = _coerce_int(df, "ct_sedation")
        sed_reason_cols = [c for c in df.columns if c.startswith("ct_sedation_reason_") and not c.endswith("__b")]
        if sed_reason_cols:
            df = coerce_int_cols(df, sed_reason_cols)
            parent_no_or_na = (df["ct_sedation"] != 1) | df["ct_sedation"].isna()
            df = _pc_rule_only_setna_when_parent_no_or_na(
                df, tr=tr,
                parent_no_or_na=parent_no_or_na,
                child_cols=sed_reason_cols,
                warn_name="ct_sedation!=Yes but ct_sedation_reason_* filled (set NA)",
                do_print=do_print,
                set_na=set_na,
            )

    _print_step_report("Step07 parent-child consistency", tr=tr, index=df.index, do_print=do_print)
    return df, tr.combined_mask(df.index)


# =========================================================
# Step08: Mechanism checks
# =========================================================

def step08_mechanism_clean(
    df: pd.DataFrame,
    *,
    do_print: bool = True,
    set_na: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df.copy()
    tr = AnomTracker()

    # ---------------- injury_mechanism ----------------
    if "injury_mechanism" in df.columns:
        df["injury_mechanism"] = _coerce_int(df, "injury_mechanism")
        invalid_mech = ~(
            df["injury_mechanism"].isin([1,2,3,4,5,6,7,8,9,10,11,12,90])
            | df["injury_mechanism"].isna()
        )
        _check(invalid_mech, "injury_mechanism invalid codes", tr=tr, do_print=do_print)
        if set_na and invalid_mech.any():
            df.loc[invalid_mech, "injury_mechanism"] = pd.NA

    # ---------------- injury_mechanism_severity + derived ----------------
    if "injury_mechanism_severity" in df.columns:
        df["injury_mechanism_severity"] = _coerce_int(df, "injury_mechanism_severity")
        invalid_sev = ~(
            df["injury_mechanism_severity"].isin([1,2,3])
            | df["injury_mechanism_severity"].isna()
        )
        _check(invalid_sev, "injury_mechanism_severity invalid codes", tr=tr, do_print=do_print)

        if set_na and invalid_sev.any():
            df.loc[invalid_sev, "injury_mechanism_severity"] = pd.NA

        sev_map = {1: "low", 2: "moderate", 3: "high"}
        # 注意：如果 severity 被 set NA 了，这里 map 会自然得到 NA
        df["mechanism_severity_3"] = df["injury_mechanism_severity"].map(sev_map)

        # injury_mechanism present but severity missing -> severity stays NA already
        if "injury_mechanism" in df.columns:
            bad_missing_sev = df["injury_mechanism"].notna() & df["injury_mechanism_severity"].isna()
            _check(bad_missing_sev, "injury_mechanism present but injury_mechanism_severity missing", tr=tr, do_print=do_print)

    _print_step_report("Step08 mechanism clean (with set_na)", tr=tr, index=df.index, do_print=do_print)
    return df, tr.combined_mask(df.index)

# =========================================================
# Step09: ciTBI consistency
# =========================================================

def step09_citbi_consistency(df: pd.DataFrame, *, do_print: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df.copy()
    tr = AnomTracker()

    for col in [
        "clinically_important_tbi",
        "neurosurgery_performed",
        "intubated_over_24h_for_head_trauma",
        "death_due_to_tbi",
        "hospitalized_2plus_nights_head_injury_and_tbi_on_ct",
    ]:
        if col in df.columns:
            df[col] = _coerce_int(df, col)

    def _is_yes(col: str) -> pd.Series:
        return (df[col] == 1) if col in df.columns else pd.Series(False, index=df.index)

    citbi_recalc = (
        _is_yes("neurosurgery_performed")
        | _is_yes("intubated_over_24h_for_head_trauma")
        | _is_yes("death_due_to_tbi")
        | _is_yes("hospitalized_2plus_nights_head_injury_and_tbi_on_ct")
    )
    df["clinically_important_tbi__recalc"] = citbi_recalc.astype("boolean")

    mismatch = pd.Series(False, index=df.index)

    if "clinically_important_tbi" in df.columns:
        citbi = df["clinically_important_tbi"]
        citbi_b = pd.Series(pd.NA, index=df.index, dtype="boolean")
        citbi_b = citbi_b.mask(citbi == 0, False)
        citbi_b = citbi_b.mask(citbi == 1, True)
        df["clinically_important_tbi__b"] = citbi_b

        mismatch = citbi_b.notna() & (citbi_b != df["clinically_important_tbi__recalc"])
        _check(mismatch, "clinically_important_tbi disagrees with components (recalc)", tr=tr, do_print=do_print)

        bad2 = (citbi == 1) & (~citbi_recalc)
        _check(bad2, "clinically_important_tbi==Yes but all components are not Yes", tr=tr, do_print=do_print)

    _print_step_report("Step09 ciTBI consistency", tr=tr, index=df.index, do_print=do_print)
    return df, mismatch


# =========================================================
# CONFIG : each step has enabled/print/drop/set_na
# =========================================================

@dataclass
class PipelineConfig:
    # default: enabled=True, do_print=True, drop=False, set_na=True
    s01: StepCtrl = field(default_factory=StepCtrl)
    s02: StepCtrl = field(default_factory=StepCtrl)
    s03: StepCtrl = field(default_factory=StepCtrl)
    s04: StepCtrl = field(default_factory=StepCtrl)
    s05: StepCtrl = field(default_factory=StepCtrl)
    s06: StepCtrl = field(default_factory=StepCtrl)
    s07: StepCtrl = field(default_factory=StepCtrl)  # parent-child consistency (uses set_na)
    s08: StepCtrl = field(default_factory=StepCtrl)
    s09: StepCtrl = field(default_factory=StepCtrl)

    # step-specific extras
    step02_drop_citbi_na: bool = True



def _apply_overrides(
    cfg: PipelineConfig,
    *,
    drop: Optional[Dict[str, bool]] = None,
    set_na: Optional[Dict[str, bool]] = None,
    do_print: Optional[Dict[str, bool]] = None,
    enabled: Optional[Dict[str, bool]] = None,
) -> PipelineConfig:
    def _get_step(k: str) -> StepCtrl:
        kk = k.zfill(2)
        if not (1 <= int(kk) <= 11):
            raise KeyError(f"Only steps 01..11 exist now, got {kk}")
        return getattr(cfg, f"s{kk}")

    if drop:
        for k, v in drop.items():
            _get_step(k).drop = bool(v)
    if set_na:
        for k, v in set_na.items():
            _get_step(k).set_na = bool(v)
    if do_print:
        for k, v in do_print.items():
            _get_step(k).do_print = bool(v)
    if enabled:
        for k, v in enabled.items():
            _get_step(k).enabled = bool(v)
    return cfg


# =========================================================
# One-click pipeline + public clean_data()
# =========================================================

def clean_pipeline(
    csv_path: Path | str,
    *,
    config: Optional[PipelineConfig] = None,
) -> pd.DataFrame:
    cfg = PipelineConfig() if config is None else config
    df = pd.DataFrame()

    validate_user_config(do_print=cfg.s01.do_print)

    df = run_step("Step01 load+rename", df, step01_load_and_rename, ctrl=cfg.s01, source=True, df_or_path=csv_path)
    df = run_step("Step02 exclude_low_gcs early", df, step02_exclude_low_gcs_early, ctrl=cfg.s02, drop_citbi_na=cfg.step02_drop_citbi_na)

    df = run_step("Step03 basic schema", df, step03_basic_schema, ctrl=cfg.s03)
    df = run_step("Step04 normalize missing codes", df, step04_normalize_missing_codes, ctrl=cfg.s04)
    df = run_step("Step05 make boolean columns", df, step05_make_boolean_columns, ctrl=cfg.s05)
    df = run_step("Step06 age clean", df, step06_age_clean, ctrl=cfg.s06)

    df = run_step("Step07 parent-child consistency", df, step07_parent_child_consistency, ctrl=cfg.s07, set_na=cfg.s07.set_na)

    df = run_step("Step08 mechanism clean", df, step08_mechanism_clean, ctrl=cfg.s08, set_na=cfg.s08.set_na)
    df = run_step("Step09 age-specific checks", df, step09_citbi_consistency, ctrl=cfg.s09)

    return df


def clean_data(
    csv_path: Path | str,
    *,
    drop: Optional[Dict[str, bool]] = None,
    set_na: Optional[Dict[str, bool]] = None,
    do_print: Optional[Dict[str, bool]] = None,
    enabled: Optional[Dict[str, bool]] = None,

    drop_citbi_na_in_step02: bool = True,
) -> pd.DataFrame:
    cfg = PipelineConfig()
    cfg.step02_drop_citbi_na = bool(drop_citbi_na_in_step02)

    cfg = _apply_overrides(cfg, drop=drop, set_na=set_na, do_print=do_print, enabled=enabled)
    return clean_pipeline(csv_path, config=cfg)


if __name__ == "__main__":
    RAW_CSV = Path("./stat-214-gsi/lab1/data/TBI PUD 10-08-2013.csv")

    df_clean = clean_data(
        RAW_CSV,
        drop={"02": True},
        set_na={"07": True,
                "08": True,
                },
    )
    df_analysis = df_clean[df_clean.get("analysis_eligible", True) == True].copy()

    print(f"\n[FINAL] df_clean rows={len(df_clean)} | df_analysis rows={len(df_analysis)}")