# figures.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# optional deps (used by some figures)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)

pio.renderers.default = "png"



# ============================================================
# Global settings
# ============================================================

def _set_plotly_png_renderer():
    # nbconvert-friendly
    pio.renderers.default = "png"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_cleaned(clean_csv: Path) -> pd.DataFrame:
    return pd.read_csv(clean_csv, low_memory=False)


def _to01(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return pd.to_numeric(s.map({"true": 1, "false": 0, "1": 1, "0": 0}), errors="coerce")


def _to01_bool(series: pd.Series) -> pd.Series:
    if str(series.dtype) == "bool":
        return series.astype("float")
    ss = series.astype(str).str.strip().str.lower()
    out = ss.map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    if out.isna().all():
        out = pd.to_numeric(series, errors="coerce")
    out = out.where(out.isin([0.0, 1.0]), np.nan)
    return out


def _require_cols(df: pd.DataFrame, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns: {miss}")


def _wilson_ci(k, n, z=1.96):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.where(n > 0, k / n, np.nan)
    den = 1 + z**2 / n
    center = (p + z**2/(2*n)) / den
    half = (z * np.sqrt((p*(1-p)/n) + (z**2/(4*n**2)))) / den
    lo = np.clip(center - half, 0, 1)
    hi = np.clip(center + half, 0, 1)
    return p, lo, hi


def _wilson_ci_vec(k, n, z=1.96):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.where(n > 0, k / n, np.nan)
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = (z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / den
    lo = np.clip(center - half, 0, 1)
    hi = np.clip(center + half, 0, 1)
    return lo, hi

def _fmt_pct_text(z: np.ndarray, *, decimals: int = 0) -> np.ndarray:
    """
    Return string array like ['12%', '3%', ...] with same shape as z.
    Uses np.char.add for NumPy versions that don't support ndarray + 'str'.
    """
    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, "", dtype=object)

    mask = np.isfinite(z)
    if not mask.any():
        return out

    if decimals == 0:
        vals = np.rint(z[mask] * 100).astype(int).astype(str)
    else:
        vals = np.round(z[mask] * 100, decimals=decimals).astype(str)

    out[mask] = np.char.add(vals, "%")
    return out


def _fmt_pct_text_series(s: pd.Series, *, decimals: int = 2) -> pd.Series:
    # For pandas Series (sometimes nicer than numpy)
    x = pd.to_numeric(s, errors="coerce")
    if decimals == 0:
        return (np.rint(x * 100).astype("Int64").astype(str) + "%").where(x.notna(), "")
    return (x.mul(100).round(decimals).astype(str) + "%").where(x.notna(), "")

# ============================================================
# Fig 1: Age distribution + mechanism bar (clean-only)
# ============================================================

def make_fig1(clean_csv: Path, out_dir: Path) -> Path:
    _set_plotly_png_renderer()
    out_dir = _ensure_dir(out_dir)
    df = _read_cleaned(clean_csv)

    AGE_Y   = "age_years"
    OUTCOME = "clinically_important_tbi__recalc"
    MECH3   = "mechanism_severity_3"

    y = df[OUTCOME].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    _ = float(y.mean()) if y.notna().any() else np.nan
    _ = len(df)

    # A) Age histogram (percent)
    age = pd.to_numeric(df[AGE_Y], errors="coerce").dropna()
    age_mean = float(age.mean()) if len(age) else np.nan
    age_med  = float(age.median()) if len(age) else np.nan

    fig_age = px.histogram(
        age.to_frame(name=AGE_Y),
        x=AGE_Y,
        nbins=30,
        histnorm="percent",
        opacity=0.85
    )
    fig_age.update_traces(hovertemplate="Age=%{x:.2f}<br>%{y:.2f}%<extra></extra>")
    fig_age.update_layout(template="plotly_white")

    # B) Mechanism bar
    mech = df[MECH3].astype("object")
    mech = mech.where(mech.isin(["low", "moderate", "high"]), other=np.nan)

    tmp = (
        mech.fillna("Missing")
            .value_counts(dropna=False)
            .rename_axis("mech")
            .reset_index(name="count")
    )
    tmp["pct"] = tmp["count"] / tmp["count"].sum()

    order = ["low", "moderate", "high", "Missing"]
    tmp["mech"] = pd.Categorical(tmp["mech"], categories=order, ordered=True)
    tmp = tmp.sort_values("mech")
    tmp["label"] = tmp.apply(lambda r: f"{int(r['count']):,}\n({r['pct']:.1%})", axis=1)

    fig_mech = px.bar(tmp, x="mech", y="count", text="label")
    fig_mech.update_traces(
        textposition="outside",
        cliponaxis=False,
        opacity=0.9,
        hovertemplate="Mech=%{x}<br>Count=%{y:,}<br>Share=%{customdata[0]:.1%}<extra></extra>",
        customdata=np.c_[tmp["pct"].to_numpy()],
    )
    fig_mech.update_layout(template="plotly_white")

    # Combine
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.12,
        subplot_titles=("A. Age distribution", "B. Mechanism severity (3-level)")
    )
    for tr in fig_age.data:
        fig.add_trace(tr, row=1, col=1)
    for tr in fig_mech.data:
        fig.add_trace(tr, row=1, col=2)

    ann_y = 0.93
    ann_yshift = -6

    if np.isfinite(age_mean):
        fig.add_vline(x=age_mean, row=1, col=1, opacity=0.6, line_width=2)
        fig.add_annotation(
            x=age_mean, y=ann_y, xref="x1", yref="paper",
            text="Mean", showarrow=False,
            yanchor="top", yshift=ann_yshift,
            bgcolor="rgba(255,255,255,0.7)"
        )
    if np.isfinite(age_med):
        fig.add_vline(x=age_med, row=1, col=1, opacity=0.6, line_width=2, line_dash="dot")
        fig.add_annotation(
            x=age_med, y=ann_y, xref="x1", yref="paper",
            text="Median", showarrow=False,
            yanchor="top", yshift=ann_yshift,
            bgcolor="rgba(255,255,255,0.7)"
        )

    fig.update_xaxes(title_text="Age (years)", row=1, col=1, showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(title_text="Patients (%)", row=1, col=1, showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_xaxes(title_text="Mechanism severity", row=1, col=2, showgrid=False)
    fig.update_yaxes(title_text="Number of patients", row=1, col=2, showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    fig.update_layout(
        template="plotly_white",
        height=420,
        width=1080,
        font=dict(size=13),
        margin=dict(l=70, r=30, t=105, b=60),
        showlegend=False
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=12)

    out_path = out_dir / "fig1.png"
    fig.write_image(out_path.as_posix(), scale=3)
    return out_path


# ============================================================
# Fig 2: Missingness heatmap by age + funnel (clean-only)
# ============================================================

def make_fig2(clean_csv: Path, out_dir: Path) -> Path:
    _set_plotly_png_renderer()
    out_dir = _ensure_dir(out_dir)
    df = _read_cleaned(clean_csv)

    AGE_Y = "age_years"
    df[AGE_Y] = pd.to_numeric(df[AGE_Y], errors="coerce")
    df["age_group"] = pd.cut(
        df[AGE_Y],
        bins=[-0.1, 2, 5, 10, 17],
        labels=["0–2", "3–5", "6–10", "11–17"]
    )

    vars_to_show = [
        "amnesia_for_event",
        "headache_at_ed_eval",
        "vomiting_anytime_after_injury",
        "loss_of_consciousness_history",
        "post_traumatic_seizure",
        "parent_reports_acting_normal",
    ]
    vars_to_show = [v for v in vars_to_show if v in df.columns]

    rename_map = {
        "amnesia_for_event": "Amnesia",
        "headache_at_ed_eval": "Headache",
        "vomiting_anytime_after_injury": "Vomiting",
        "loss_of_consciousness_history": "LOC",
        "post_traumatic_seizure": "Seizure",
        "parent_reports_acting_normal": "Acting normal",
    }

    rows = []
    for v in vars_to_show:
        tmp = df.groupby("age_group", observed=False)[v].apply(lambda s: s.isna().mean())
        for g, frac in tmp.items():
            rows.append({
                "age_group": str(g),
                "variable_label": rename_map.get(v, v),
                "missing_fraction": float(frac),
            })
    miss_by_age = pd.DataFrame(rows)

    heat = miss_by_age.pivot(index="variable_label", columns="age_group", values="missing_fraction")
    heat = heat.reindex(index=[rename_map.get(v, v) for v in vars_to_show])
    heat = heat.reindex(columns=["0–2", "3–5", "6–10", "11–17"])

    z = heat.to_numpy(dtype=float)
    text = _fmt_pct_text(z, decimals=0)
    zmax = float(np.nanmax(z)) if np.isfinite(np.nanmax(z)) else 0.0
    zmax = max(zmax, 0.001)

    # Funnel numbers (as in your code)
    n_raw = 43399
    n_low_gcs = 969
    n_citbi_na = 20
    n_final = 42412

    labels = [
        f"Raw<br>N={n_raw:,}",
        f"Low GCS<br>−{n_low_gcs:,}",
        f"Missing ciTBI<br>−{n_citbi_na:,}",
        f"Final<br>N={n_final:,}",
    ]
    values = [n_raw, n_low_gcs, n_citbi_na, n_final]

    combo = make_subplots(
        rows=1, cols=2,
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.16,
        subplot_titles=(
            "Missingness by age group (selected variables)",
            "Cohort selection and exclusions"
        ),
        specs=[[{"type": "heatmap"}, {"type": "funnel"}]]
    )

    combo.add_trace(
        go.Heatmap(
            z=z,
            x=heat.columns.tolist(),
            y=heat.index.tolist(),
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=12),
            zmin=0,
            zmax=zmax,
            colorscale="Blues",
            xgap=2,
            ygap=2,
            hovertemplate="Age: %{x}<br>Variable: %{y}<br>Missing: %{z:.1%}<extra></extra>",
            colorbar=dict(
                title=dict(text="Missing", side="top"),
                tickformat=".0%",
                len=0.88,
                thickness=14,
                x=-0.16,
                xanchor="left",
                xpad=12,
                y=0.5,
                yanchor="middle",
                outlinewidth=0
            )
        ),
        row=1, col=1
    )

    combo.add_trace(
        go.Funnel(
            y=labels,
            x=values,
            textinfo="value+percent initial",
            textfont=dict(size=14),
            opacity=0.95
        ),
        row=1, col=2
    )

    combo.update_xaxes(row=1, col=1, title_text="Age group (years)", showgrid=False, zeroline=False)
    combo.update_yaxes(row=1, col=1, title_text="", showgrid=False, zeroline=False,
                       automargin=True, tickfont=dict(size=13))

    combo.update_xaxes(row=1, col=2, title_text="", showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    combo.update_yaxes(row=1, col=2, title_text="", showgrid=False, zeroline=False,
                       automargin=True, tickfont=dict(size=13))

    combo.update_layout(
        template="plotly_white",
        width=1550,
        height=580,
        font=dict(size=14),
        margin=dict(l=230, r=50, t=120, b=70),
        showlegend=False
    )
    for ann in combo.layout.annotations:
        ann.font = dict(size=14)

    out_path = out_dir / "fig2.png"
    combo.write_image(out_path.as_posix(), scale=3)
    return out_path


# ============================================================
# Fig 3: ciTBI prevalence by age + heatmap age×mech (clean-only)
# ============================================================

def make_fig3(clean_csv: Path, out_dir: Path) -> Path:
    _set_plotly_png_renderer()
    out_dir = _ensure_dir(out_dir)
    df = _read_cleaned(clean_csv)

    OUTCOME = "clinically_important_tbi__recalc"
    AGE_Y = "age_years"
    MECH3 = "mechanism_severity_3"
    _require_cols(df, [OUTCOME, AGE_Y, MECH3])

    df2 = df.copy()
    df2[AGE_Y] = pd.to_numeric(df2[AGE_Y], errors="coerce")

    age_order = ["0–2", "3–5", "6–10", "11–17"]
    df2["age_group"] = pd.cut(df2[AGE_Y], bins=[-0.1, 2, 5, 10, 17], labels=age_order)

    df2["mech"] = df2[MECH3].astype("object")
    df2["mech"] = df2["mech"].where(df2["mech"].isin(["low", "moderate", "high"]), other="Missing")

    y = _to01_bool(df2[OUTCOME])

    tmp = (
        pd.DataFrame({"age_group": df2["age_group"], "y": y})
        .dropna(subset=["age_group", "y"])
        .groupby("age_group", observed=False)["y"]
        .agg(n="count", k="sum")
        .reset_index()
    )
    tmp["age_group"] = pd.Categorical(tmp["age_group"], categories=age_order, ordered=True)
    tmp = tmp.sort_values("age_group")
    tmp["rate"] = tmp["k"] / tmp["n"]

    ymax = float(np.nanmax(tmp["rate"])) if len(tmp) else 0.03
    ymax = max(ymax, 0.03)
    ymax_pad = min(1.0, ymax * 1.35 + 0.005)

    tab = (
        pd.DataFrame({"age_group": df2["age_group"], "mech": df2["mech"], "y": y})
        .dropna(subset=["age_group", "mech", "y"])
        .groupby(["mech", "age_group"], observed=False)["y"]
        .agg(n="count", k="sum")
        .reset_index()
    )
    tab["rate"] = tab["k"] / tab["n"]

    pivot = tab.pivot(index="mech", columns="age_group", values="rate").reindex(columns=age_order)
    mech_order = ["low", "moderate", "high", "Missing"]
    pivot = pivot.reindex([m for m in mech_order if m in pivot.index] + [m for m in pivot.index if m not in mech_order])

    z_mat = pivot.to_numpy(dtype=float)
    text_mat = _fmt_pct_text(z_mat, decimals=2)
    zmax_hm = float(np.nanmax(z_mat)) if np.isfinite(np.nanmax(z_mat)) else 0.01
    zmax_hm = max(zmax_hm, 0.001)

    combo = make_subplots(
        rows=1, cols=2,
        column_widths=[0.42, 0.58],
        horizontal_spacing=0.14,
        subplot_titles=("ciTBI prevalence by age group", "ciTBI rate by age group × mechanism severity"),
        specs=[[{"type": "bar"}, {"type": "heatmap"}]]
    )

    combo.add_trace(
        go.Bar(
            x=tmp["age_group"].astype(str),
            y=tmp["rate"],
            text=(tmp["rate"] * 100).round(2),
            texttemplate="%{text:.2f}%",
            textposition="outside",
            cliponaxis=False,
            opacity=0.9,
            customdata=np.stack([tmp["n"].to_numpy(), tmp["k"].to_numpy()], axis=1),
            hovertemplate=("Age group: %{x}<br>Rate: %{y:.2%}<br>n=%{customdata[0]:,}, k=%{customdata[1]:,}<extra></extra>"),
            name="Prevalence"
        ),
        row=1, col=1
    )

    combo.add_trace(
        go.Heatmap(
            z=z_mat,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            text=text_mat,
            texttemplate="%{text}",
            textfont=dict(size=12),
            zmin=0,
            zmax=zmax_hm,
            colorscale="Blues",
            xgap=2,
            ygap=2,
            hovertemplate="Age group: %{x}<br>Mech: %{y}<br>Rate: %{z:.2%}<extra></extra>",
            colorbar=dict(
                title=dict(text="ciTBI rate", side="top"),
                tickformat=".0%",
                thickness=14,
                len=0.86,
                x=1.02,
                xanchor="left",
                outlinewidth=0
            ),
            name="Rate"
        ),
        row=1, col=2
    )

    combo.update_xaxes(title_text="Age group (years)", row=1, col=1)
    combo.update_yaxes(title_text="ciTBI prevalence", row=1, col=1, tickformat=".1%")
    combo.update_yaxes(range=[0, ymax_pad], row=1, col=1, showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    combo.update_xaxes(title_text="Age group (years)", row=1, col=2)
    combo.update_yaxes(title_text="Mechanism severity", row=1, col=2, automargin=True, tickfont=dict(size=13))
    combo.update_xaxes(row=1, col=2, showgrid=False)

    combo.update_layout(
        template="plotly_white",
        width=1450,
        height=460,
        font=dict(size=13),
        margin=dict(l=85, r=95, t=95, b=60),
        showlegend=False
    )
    for ann in combo.layout.annotations:
        ann.font = dict(size=13)

    out_path = out_dir / "fig3.png"
    combo.write_image(out_path.as_posix(), scale=3)
    return out_path


# ============================================================
# Shared GLM builder for figf1/figf2/figs1
# ============================================================

def _build_glm_df(df_in: pd.DataFrame) -> pd.DataFrame:
    OUTCOME = "clinically_important_tbi__recalc"
    y = _to01(df_in[OUTCOME])

    X = pd.DataFrame({
        "y": y,
        "age_years": pd.to_numeric(df_in["age_years"], errors="coerce"),
        "mech": df_in["mechanism_severity_3"].astype("object"),
        "vomit": df_in["vomiting_anytime_after_injury__b"].astype("float"),
        "ams": df_in["altered_mental_status__b"].astype("float"),
        "basilar": df_in["basilar_skull_fracture_signs__b"].astype("float"),
        "hema": df_in["scalp_hematoma_or_swelling__b"].astype("float"),
        "loc_raw": pd.to_numeric(df_in["loss_of_consciousness_history"], errors="coerce"),
        "psf_raw": pd.to_numeric(df_in["palpable_skull_fracture"], errors="coerce"),
    })

    X["mech"] = X["mech"].where(X["mech"].isin(["low", "moderate", "high"]), other=np.nan)

    X["loc_yes"] = X["loc_raw"].map({1: 1, 0: 0}).fillna(0).astype(int)
    X["loc_missing"] = X["loc_raw"].isna().astype(int)
    X["psf_yes"] = X["psf_raw"].map({1: 1, 0: 0}).fillna(0).astype(int)
    X["psf_missing"] = X["psf_raw"].isna().astype(int)

    X = X.drop(columns=["loc_raw", "psf_raw"]).dropna(
        subset=["y", "age_years", "mech", "vomit", "ams", "basilar", "hema"]
    ).copy()
    return X


def _fit_glm(X: pd.DataFrame):
    formula = (
        "y ~ age_years + C(mech, Treatment(reference='low')) "
        "+ vomit + ams + basilar + hema + loc_yes + loc_missing + psf_yes + psf_missing"
    )
    res = smf.glm(formula=formula, data=X, family=sm.families.Binomial()).fit(cov_type="HC1")
    return res


# ============================================================
# Fig f1: Calibration + risk concentration (glm; clean-only)
# ============================================================

def make_figf1(clean_csv: Path, out_dir: Path) -> Path:
    _set_plotly_png_renderer()
    out_dir = _ensure_dir(out_dir)
    df = _read_cleaned(clean_csv)

    X = _build_glm_df(df)
    res = _fit_glm(X)
    p = res.predict(X)

    # Left: calibration by deciles
    cal = pd.DataFrame({"p": p, "y": X["y"].astype(int)})

    # NOTE: your original used qcut(cal["p"], 10). This can break on ties.
    # Keep your original behavior; if it errors, switch to rank(method="first").
    cal["decile"] = pd.qcut(cal["p"], 10, labels=[f"D{i}" for i in range(1, 11)])

    tabA = (
        cal.groupby("decile", observed=False)
           .agg(n=("y", "count"), k=("y", "sum"),
                obs_rate=("y", "mean"), pred_mean=("p", "mean"))
           .reset_index()
    )
    _, lo, hi = _wilson_ci(tabA["k"], tabA["n"], z=1.96)
    tabA["lo"] = lo
    tabA["hi"] = hi
    tabA["decile_str"] = tabA["decile"].astype(str)

    xmin = float(tabA["pred_mean"].min())
    xmax = float(tabA["pred_mean"].max())
    ymin = float(min(tabA["obs_rate"].min(), tabA["lo"].min()))
    ymax = float(max(tabA["obs_rate"].max(), tabA["hi"].max()))
    xpad = 0.05 * (xmax - xmin) if xmax > xmin else 0.01
    ypad = 0.08 * (ymax - ymin) if ymax > ymin else 0.01

    xline = np.linspace(max(0, xmin - xpad), min(1, xmax + xpad), 200)

    traceA_points = go.Scatter(
        x=tabA["pred_mean"], y=tabA["obs_rate"],
        mode="markers",
        marker=dict(size=10, opacity=0.55),
        error_y=dict(
            type="data",
            array=(tabA["hi"]-tabA["obs_rate"]),
            arrayminus=(tabA["obs_rate"]-tabA["lo"]),
            thickness=1.2,
            width=0
        ),
        customdata=np.c_[tabA["decile_str"], tabA["n"], tabA["k"]],
        hovertemplate=("Decile: %{customdata[0]}<br>"
                       "n: %{customdata[1]:,.0f}, k: %{customdata[2]:,.0f}<br>"
                       "Pred mean: %{x:.3%}<br>"
                       "Observed: %{y:.3%}<extra></extra>"),
        showlegend=False
    )

    traceA_line = go.Scatter(x=xline, y=xline, mode="lines", opacity=0.25, line=dict(width=2), showlegend=False)

    label_keep = {"D8", "D9", "D10"}
    lab = tabA.loc[tabA["decile_str"].isin(label_keep)].copy().reset_index(drop=True)
    dxs = np.array([+0.006, +0.006, +0.006])
    dys = np.array([+0.004, +0.001, -0.002])
    traceA_labels = go.Scatter(
        x=lab["pred_mean"] + dxs[:len(lab)],
        y=lab["obs_rate"] + dys[:len(lab)],
        mode="text",
        text=lab["decile_str"],
        textfont=dict(size=13),
        hoverinfo="skip",
        showlegend=False
    )

    # Right: risk concentration
    pred = pd.DataFrame({"p": p, "y": X["y"].astype(int)}).sort_values("p", ascending=False).reset_index(drop=True)
    N = len(pred)
    K = int(pred["y"].sum())
    baseline = (K / N) if N > 0 else np.nan

    fracs = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    rows = []
    for f in fracs:
        m = max(1, int(round(f * N)))
        captured = int(pred.loc[:m-1, "y"].sum())
        capture_rate = (captured / K) if K > 0 else np.nan
        top_rate = captured / m
        lift = (top_rate / baseline) if baseline > 0 else np.nan
        rows.append({"top_frac": f, "capture": capture_rate, "lift": lift})
    tabB = pd.DataFrame(rows)

    cap_min, cap_max = float(np.nanmin(tabB["capture"])), float(np.nanmax(tabB["capture"]))
    lift_min, lift_max = float(np.nanmin(tabB["lift"])), float(np.nanmax(tabB["lift"]))
    cap_pad = 0.08 * (cap_max - cap_min) if cap_max > cap_min else 0.05
    lift_pad = 0.10 * (lift_max - lift_min) if lift_max > lift_min else 0.10

    traceB_capture = go.Scatter(
        x=tabB["top_frac"], y=tabB["capture"],
        mode="lines+markers+text",
        text=[f"{int(f*100)}%" for f in tabB["top_frac"]],
        textposition="top center",
        textfont=dict(size=12),
        name="Capture",
        opacity=0.9,
        hovertemplate="Top frac: %{x:.0%}<br>Capture: %{y:.1%}<extra></extra>",
        showlegend=True
    )

    traceB_lift = go.Scatter(
        x=tabB["top_frac"], y=tabB["lift"],
        mode="lines+markers",
        name="Lift",
        opacity=0.9,
        hovertemplate="Top frac: %{x:.0%}<br>Lift: %{y:.2f}×<extra></extra>",
        showlegend=True
    )

    combo = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        horizontal_spacing=0.14,
        subplot_titles=("A. Calibration (deciles; Wilson 95% CI)", "B. Risk concentration (high-risk tail)"),
        specs=[[{"type": "xy"}, {"type": "xy", "secondary_y": True}]]
    )

    combo.add_trace(traceA_points, row=1, col=1)
    combo.add_trace(traceA_line,   row=1, col=1)
    combo.add_trace(traceA_labels, row=1, col=1)

    combo.add_trace(traceB_capture, row=1, col=2, secondary_y=False)
    combo.add_trace(traceB_lift,    row=1, col=2, secondary_y=True)

    combo.update_xaxes(
        title_text="Mean predicted risk (by decile)",
        tickformat=".2%",
        range=[max(0, xmin - xpad), min(1, xmax + xpad)],
        row=1, col=1
    )
    combo.update_yaxes(
        title_text="Observed ciTBI rate",
        tickformat=".2%",
        range=[max(0, ymin - ypad), min(1, ymax + ypad)],
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        row=1, col=1
    )

    combo.update_xaxes(title_text="Top fraction by predicted risk", tickformat=".0%", row=1, col=2)
    combo.update_yaxes(
        title_text="Capture",
        tickformat=".0%",
        range=[max(0, cap_min - cap_pad), min(1, cap_max + cap_pad)],
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        row=1, col=2, secondary_y=False
    )
    combo.update_yaxes(
        title_text="Lift (× baseline)",
        range=[max(0, lift_min - lift_pad), lift_max + lift_pad],
        row=1, col=2, secondary_y=True
    )

    combo.update_layout(
        template="plotly_white",
        height=560,
        width=1400,
        margin=dict(l=90, r=90, t=120, b=85),
        font=dict(size=13),
        legend=dict(
            orientation="h",
            x=0.985, xanchor="right",
            y=0.985, yanchor="top",
            bgcolor="rgba(255,255,255,0.75)"
        )
    )
    for ann in combo.layout.annotations:
        ann.font = dict(size=13)

    out_path = out_dir / "figf1.png"
    combo.write_image(out_path.as_posix(), scale=3)
    return out_path


# ============================================================
# Fig f2: Adjusted OR forest + permutation importance (glm)
# ============================================================

def make_figf2(clean_csv: Path, out_dir: Path) -> Path:
    _set_plotly_png_renderer()
    out_dir = _ensure_dir(out_dir)
    df = _read_cleaned(clean_csv)

    # Fit GLM
    X = _build_glm_df(df)
    res = _fit_glm(X)

    # Panel A: Adjusted OR forest (drop psf_missing)
    params = res.params
    ci = res.conf_int()

    or_df = pd.DataFrame({
        "term": params.index,
        "OR": np.exp(params.values),
        "OR_lo": np.exp(ci[0].values),
        "OR_hi": np.exp(ci[1].values),
    })

    drop_terms = {"Intercept", "psf_missing"}
    or_df = or_df[~or_df["term"].isin(drop_terms)].copy()

    short = {
        "age_years": "Age (yr)",
        "vomit": "Vomit",
        "ams": "AMS",
        "basilar": "Basilar",
        "hema": "Hematoma",
        "loc_yes": "LOC=1",
        "loc_missing": "LOC NA",
        "psf_yes": "PSF=1",
        "C(mech, Treatment(reference='low'))[T.moderate]": "Mech: Mod",
        "C(mech, Treatment(reference='low'))[T.high]": "Mech: High",
    }
    or_df["label"] = or_df["term"].map(short).fillna(or_df["term"])
    or_df = or_df.sort_values("OR", ascending=True).reset_index(drop=True)

    xs, ys = [], []
    for _, r in or_df.iterrows():
        xs += [r["OR_lo"], r["OR_hi"], None]
        ys += [r["label"], r["label"], None]
    trace_ci = go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=3), opacity=0.55,
                          hoverinfo="skip", showlegend=False)

    trace_or = go.Scatter(
        x=or_df["OR"], y=or_df["label"],
        mode="markers",
        marker=dict(size=11, opacity=0.90),
        customdata=np.c_[or_df["term"], or_df["OR_lo"], or_df["OR_hi"]],
        hovertemplate=("%{y}<br>OR=%{x:.2f}<br>95% CI [%{customdata[1]:.2f}, %{customdata[2]:.2f}]<extra></extra>"),
        showlegend=False
    )

    # Panel B: Permutation importance (Δ log loss) drop psf_missing
    def neg_logloss(y, p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

    p0 = res.predict(X).to_numpy()
    y0 = X["y"].astype(float).to_numpy()
    base_loss = neg_logloss(y0, p0)

    features = ["age_years", "mech", "vomit", "ams", "basilar", "hema", "loc_yes", "loc_missing", "psf_yes"]

    rng = np.random.default_rng(123)
    B = 10
    rows = []
    for f in features:
        losses = []
        for _ in range(B):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].values)
            pp = res.predict(Xp).to_numpy()
            losses.append(neg_logloss(y0, pp))
        rows.append({"feature": f, "delta_logloss": float(np.mean(losses) - base_loss)})

    imp = pd.DataFrame(rows)
    imp["label"] = imp["feature"].map(short).fillna(imp["feature"])
    imp = imp.sort_values("delta_logloss", ascending=False).reset_index(drop=True)

    trace_imp = go.Bar(
        x=imp["delta_logloss"], y=imp["label"],
        orientation="h",
        text=imp["delta_logloss"].map(lambda v: f"{v:.4f}"),
        textposition="outside",
        cliponaxis=False,
        opacity=0.9,
        hovertemplate="%{y}<br>Δ log loss=%{x:.4f}<extra></extra>",
        showlegend=False
    )

    font_size = 13
    combo = make_subplots(
        rows=1, cols=2,
        column_widths=[0.54, 0.46],
        horizontal_spacing=0.14,
        subplot_titles=("A. Adjusted odds ratios (95% CI)", "B. Permutation importance (Δ log loss)"),
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    combo.add_trace(trace_ci, row=1, col=1)
    combo.add_trace(trace_or, row=1, col=1)
    combo.add_vline(x=1.0, line_width=2, opacity=0.55, row=1, col=1)

    combo.add_trace(trace_imp, row=1, col=2)
    combo.add_vline(x=0.0, line_width=2, opacity=0.25, row=1, col=2)

    combo.update_xaxes(title_text="Adj OR (log scale)", type="log",
                       showgrid=True, gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
    combo.update_yaxes(title_text="", automargin=True, row=1, col=1)

    combo.update_xaxes(title_text="Δ log loss (higher = more important)",
                       showgrid=True, gridcolor="rgba(0,0,0,0.06)", row=1, col=2)
    combo.update_yaxes(title_text="", automargin=True,
                       categoryorder="array", categoryarray=imp["label"].tolist()[::-1], row=1, col=2)

    combo.update_layout(
        template="plotly_white",
        width=1400,
        height=560,
        font=dict(size=font_size),
        margin=dict(l=110, r=120, t=105, b=80),
        showlegend=False
    )
    for ann in combo.layout.annotations:
        ann.font = dict(size=font_size)

    out_path = out_dir / "figf2.png"
    combo.write_image(out_path.as_posix(), scale=3)
    return out_path


# ============================================================
# Fig f3: Under-2 hematoma × acting normal + RD bootstrap
# ============================================================

def make_figf3(clean_csv: Path, out_dir: Path) -> Path:
    _set_plotly_png_renderer()
    out_dir = _ensure_dir(out_dir)
    df = _read_cleaned(clean_csv)

    OUTCOME = "clinically_important_tbi__recalc"
    ACT = "parent_reports_acting_normal__b"
    HEMA = "scalp_hematoma_or_swelling__b"

    y = _to01(df[OUTCOME])

    under2 = (
        df["age_under2"].astype(str).str.strip().str.lower()
          .map({"true": True, "false": False})
          .fillna(False)
    )

    sub = df.loc[under2].copy()
    sub_y = y.loc[under2]

    tab = pd.DataFrame({
        "acting_normal": sub[ACT],
        "hematoma": sub[HEMA],
        "y": sub_y
    }).dropna()

    tab["acting_normal"] = tab["acting_normal"].astype(bool)
    tab["hematoma"] = tab["hematoma"].astype(bool)
    tab["y"] = pd.to_numeric(tab["y"], errors="coerce")
    tab = tab.dropna(subset=["y"])

    # LEFT plot
    g = (
        tab.groupby(["hematoma", "acting_normal"], observed=False)["y"]
           .agg(n="count", k="sum")
           .reset_index()
    )
    g["rate"] = g["k"] / g["n"]
    _, lo, hi = _wilson_ci(g["k"], g["n"], z=1.96)
    g["lo"], g["hi"] = lo, hi

    hema_map = {False: "No\nhema", True: "Hema\npresent"}
    act_map  = {True: "Acting\nnormal", False: "Not acting\nnormal"}
    g["hema_lab"] = g["hematoma"].map(hema_map)
    g["act_lab"]  = g["acting_normal"].map(act_map)

    x_order = ["No\nhema", "Hema\npresent"]

    left_traces = []
    for act in [True, False]:
        gg = g.loc[g["acting_normal"] == act].copy()
        gg["hema_lab"] = pd.Categorical(gg["hema_lab"], categories=x_order, ordered=True)
        gg = gg.sort_values("hema_lab")

        name = act_map[act]
        left_traces.append(
            go.Bar(
                x=gg["hema_lab"].astype(str),
                y=gg["rate"],
                name=name,
                text=(gg["rate"] * 100).round(2),
                textposition="outside",
                cliponaxis=False,
                opacity=0.9,
                error_y=dict(type="data",
                             array=(gg["hi"] - gg["rate"]),
                             arrayminus=(gg["rate"] - gg["lo"])),
                customdata=np.c_[gg["n"].to_numpy(), gg["k"].to_numpy()],
                hovertemplate=("Hematoma: %{x}<br>"
                               f"{name}<br>"
                               "ciTBI rate: %{y:.2%}<br>"
                               "n=%{customdata[0]:,.0f}, k=%{customdata[1]:,.0f}<extra></extra>"),
                offsetgroup=name,
            )
        )

    # RIGHT plot: RD vs reference with bootstrap CI
    hema2 = np.where(tab["hematoma"].to_numpy(), "Hema", "NoHema")
    act2  = np.where(tab["acting_normal"].to_numpy(), "Normal", "NotNormal")
    tab["group"] = pd.Series(hema2).astype(str).str.cat(pd.Series(act2).astype(str), sep=" × ")

    ref = "NoHema × Normal"
    rates = tab.groupby("group")["y"].mean()
    if ref not in rates.index:
        raise ValueError(f"Reference group '{ref}' not found. Present: {rates.index.tolist()}")

    rd_obs = (rates - rates.loc[ref]).reset_index()
    rd_obs.columns = ["group", "risk_diff"]

    rng = np.random.default_rng(7)
    B = 500
    groups = rd_obs["group"].tolist()
    rd_boot = {gg: [] for gg in groups}

    for _ in range(B):
        idx = rng.integers(0, len(tab), len(tab))
        samp = tab.iloc[idx]
        r = samp.groupby("group")["y"].mean()
        if ref not in r.index:
            continue
        for gg in groups:
            if gg in r.index:
                rd_boot[gg].append(float(r[gg] - r[ref]))

    lo_list, hi_list = [], []
    for gg in groups:
        vals = np.asarray(rd_boot[gg], dtype=float)
        if len(vals) >= 30:
            lo_, hi_ = np.quantile(vals, [0.025, 0.975])
        else:
            lo_, hi_ = (np.nan, np.nan)
        lo_list.append(lo_)
        hi_list.append(hi_)

    rd_obs["lo"] = lo_list
    rd_obs["hi"] = hi_list

    order = [ref] + [gg for gg in groups if gg != ref]
    rd_obs["group"] = pd.Categorical(rd_obs["group"], categories=order, ordered=True)
    rd_obs = rd_obs.sort_values("group")

    label_map = {
        "NoHema × Normal": "No hema\n× Normal\n(ref)",
        "NoHema × NotNormal": "No hema\n× Not normal",
        "Hema × Normal": "Hema\n× Normal",
        "Hema × NotNormal": "Hema\n× Not normal",
    }
    rd_obs_plot = rd_obs.copy()
    rd_obs_plot["group_lab"] = rd_obs_plot["group"].astype(str).map(label_map).fillna(rd_obs_plot["group"].astype(str))

    rd_obs_plot["rd_pp"] = rd_obs_plot["risk_diff"] * 100.0
    rd_obs_plot["lo_pp"] = rd_obs_plot["lo"] * 100.0
    rd_obs_plot["hi_pp"] = rd_obs_plot["hi"] * 100.0

    right_trace = go.Bar(
        x=rd_obs_plot["group_lab"],
        y=rd_obs_plot["rd_pp"],
        name="Risk diff vs ref",
        text=rd_obs_plot["rd_pp"].map(lambda v: f"{v:.2f}"),
        textposition="outside",
        cliponaxis=False,
        opacity=0.92,
        marker=dict(color="rgba(0, 114, 178, 0.82)",
                    line=dict(width=0.8, color="rgba(0,0,0,0.25)")),
        error_y=dict(type="data",
                     array=(rd_obs_plot["hi_pp"] - rd_obs_plot["rd_pp"]),
                     arrayminus=(rd_obs_plot["rd_pp"] - rd_obs_plot["lo_pp"])),
        hovertemplate="%{x}<br>RD vs ref: %{y:.2f} pp<extra></extra>",
    )

    color_act = {
        "Acting\nnormal": "rgba(86, 180, 233, 0.85)",
        "Not acting\nnormal": "rgba(230, 159, 0, 0.85)",
    }
    for tr in left_traces:
        tr.marker = dict(color=color_act.get(tr.name, "rgba(120,120,120,0.85)"),
                         line=dict(width=0.8, color="rgba(0,0,0,0.25)"))
        tr.textfont = dict(size=12)

    combo = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        horizontal_spacing=0.12,
        subplot_titles=("A. Under-2: ciTBI rate by hematoma × acting normal",
                        "B. Under-2: risk difference vs reference group")
    )

    combo.add_trace(left_traces[0], row=1, col=1)
    combo.add_trace(left_traces[1], row=1, col=1)

    combo.add_trace(right_trace, row=1, col=2)
    combo.add_hline(y=0.0, opacity=0.55, line_width=2, row=1, col=2)

    ymax_left = float(np.nanmax(g["rate"])) if len(g) else 0.0
    combo.update_yaxes(
        title_text="ciTBI rate",
        tickformat=".1%",
        range=[0, min(1.0, ymax_left * 1.35 + 0.01)],
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
        row=1, col=1
    )
    combo.update_xaxes(title_text="", row=1, col=1)

    ymax_right = float(np.nanmax(np.abs(rd_obs_plot["rd_pp"]))) if len(rd_obs_plot) else 0.0
    combo.update_yaxes(
        title_text="Risk difference (percentage points)",
        ticksuffix=" pp",
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
        range=[-(ymax_right*1.25 + 0.2), (ymax_right*1.25 + 0.2)],
        row=1, col=2
    )
    combo.update_xaxes(title_text="", tickangle=0, automargin=True, row=1, col=2)

    combo.update_layout(
        template="plotly_white",
        barmode="group",
        bargap=0.25,
        bargroupgap=0.10,
        width=1320,
        height=520,
        font=dict(size=13),
        margin=dict(l=90, r=40, t=95, b=95),
        legend=dict(
            orientation="v",
            x=0.02, xanchor="left",
            y=0.98, yanchor="top",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.12)",
            borderwidth=1
        )
    )
    for ann in combo.layout.annotations:
        ann.font = dict(size=13)

    out_path = out_dir / "figf3.png"
    combo.write_image(out_path.as_posix(), scale=3)
    return out_path


# ============================================================
# Fig s1: Stability BEFORE vs AFTER bootstrap calibration-by-deciles
# ============================================================

def make_figs1(clean_csv: Path, out_dir: Path) -> Path:
    _set_plotly_png_renderer()
    out_dir = _ensure_dir(out_dir)
    df = _read_cleaned(clean_csv)

    def calibration_table(df_for_fit: pd.DataFrame) -> pd.DataFrame:
        X = _build_glm_df(df_for_fit)
        res = _fit_glm(X)
        p = res.predict(X)
        cal = pd.DataFrame({"p": p, "y": X["y"].astype(int)})

        cal["decile"] = pd.qcut(
            cal["p"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)]
        )

        tab = (
            cal.groupby("decile", observed=False)
               .agg(n=("y", "count"), k=("y", "sum"), obs=("y", "mean"), pred=("p", "mean"))
               .reset_index()
        )
        lo, hi = _wilson_ci_vec(tab["k"], tab["n"], z=1.96)
        tab["lo"], tab["hi"] = lo, hi
        tab["decile_str"] = tab["decile"].astype(str)
        return tab

    def add_calibration_panel(fig, tab, *, row, col, show_legend, point_opacity):
        fig.add_trace(
            go.Scatter(
                x=tab["pred"], y=tab["obs"],
                mode="markers",
                marker=dict(size=10, opacity=point_opacity),
                error_y=dict(type="data",
                             array=(tab["hi"] - tab["obs"]),
                             arrayminus=(tab["obs"] - tab["lo"])),
                customdata=np.c_[tab["decile_str"], tab["n"], tab["k"]],
                hovertemplate=("Decile: %{customdata[0]}<br>"
                               "n=%{customdata[1]:,.0f}, k=%{customdata[2]:,.0f}<br>"
                               "Pred mean: %{x:.3%}<br>"
                               "Obs: %{y:.3%}<extra></extra>"),
                name="Deciles",
                showlegend=show_legend
            ),
            row=row, col=col
        )

        keep = {"D8", "D9", "D10"}
        lab = tab.loc[tab["decile_str"].isin(keep)].copy()
        dx = 0.0025
        dy = 0.0025
        fig.add_trace(
            go.Scatter(
                x=lab["pred"] + dx,
                y=lab["obs"] + dy,
                mode="text",
                text=lab["decile_str"],
                textfont=dict(size=13),
                hoverinfo="skip",
                showlegend=False
            ),
            row=row, col=col
        )

        x_min = float(np.nanmin(tab["pred"]))
        x_max = float(np.nanmax(tab["pred"]))
        xline = np.linspace(x_min, x_max, 120)
        fig.add_trace(
            go.Scatter(x=xline, y=xline, mode="lines", opacity=0.35, line=dict(width=2),
                       name="y = x", showlegend=show_legend),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Mean predicted risk (decile)", tickformat=".2%", row=row, col=col)
        fig.update_yaxes(
            title_text="Observed ciTBI rate (Wilson 95% CI)" if col == 1 else "",
            tickformat=".2%",
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            row=row, col=col
        )

    tab_before = calibration_table(df)

    rng = np.random.default_rng(2026)
    boot_idx = rng.integers(0, len(df), len(df))
    df_boot = df.iloc[boot_idx].reset_index(drop=True)
    tab_after = calibration_table(df_boot)

    xmin = float(np.nanmin([tab_before["pred"].min(), tab_after["pred"].min()]))
    xmax = float(np.nanmax([tab_before["pred"].max(), tab_after["pred"].max()]))
    ymin = float(np.nanmin([tab_before["obs"].min(), tab_after["obs"].min()]))
    ymax = float(np.nanmax([tab_before["obs"].max(), tab_after["obs"].max()]))

    xpad = (xmax - xmin) * 0.08 if xmax > xmin else 0.01
    ypad = (ymax - ymin) * 0.10 if ymax > ymin else 0.01
    x_range = [max(0, xmin - xpad), min(1, xmax + xpad)]
    y_range = [max(0, ymin - ypad), min(1, ymax + ypad)]

    combo = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.10,
        subplot_titles=("Stability (Before): original cohort", "Stability (After): bootstrap resample")
    )

    add_calibration_panel(combo, tab_before, row=1, col=1, show_legend=True,  point_opacity=0.45)
    add_calibration_panel(combo, tab_after,  row=1, col=2, show_legend=False, point_opacity=0.75)

    combo.update_xaxes(range=x_range, row=1, col=1)
    combo.update_xaxes(range=x_range, row=1, col=2)
    combo.update_yaxes(range=y_range, row=1, col=1)
    combo.update_yaxes(range=y_range, row=1, col=2)

    combo.update_layout(
        template="plotly_white",
        width=1280,
        height=520,
        font=dict(size=13),
        margin=dict(l=95, r=40, t=110, b=70),
        legend=dict(
            orientation="h",
            x=0.02, xanchor="left",
            y=1.10, yanchor="bottom",
            bgcolor="rgba(255,255,255,0.75)"
        )
    )
    for ann in combo.layout.annotations:
        ann.font = dict(size=13)

    out_path = out_dir / "figs1.png"
    combo.write_image(out_path.as_posix(), scale=3)
    return out_path

# ============================================================
# Part 2 (Model figures): read outputs from model.py
# Produces:
#   figm1.png (ROC + PR)
#   figm2.png (3 confusion matrices)
#   figm3.png (OR forest + permutation importance)
# ===========================================================


def _read_model_npz(model_out_dir: Path) -> dict:
    npz_path = Path(model_out_dir) / "model_outputs.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}. Run model.py first.")
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _shorten_feature_name(s: str) -> str:
    s = str(s)

    for p in ["num__", "cat__", "bin__", "remainder__"]:
        if s.startswith(p):
            s = s[len(p):]

    rep = {
        "age_years": "Age (years)",
        "mechanism_severity_3": "Mech",
        "vomiting_anytime_after_injury__b": "Vomit",
        "altered_mental_status__b": "AMS",
        "basilar_skull_fracture_signs__b": "Basilar signs",
        "scalp_hematoma_or_swelling__b": "Scalp hematoma",
        "parent_reports_acting_normal__b": "Acting normal",
    }
    for k, v in rep.items():
        s = s.replace(k, v)

    s = s.replace("_missing_indicator", " missing")
    s = s.replace("__missing_indicator", " missing")

    # common onehot naming patterns -> nicer mechanism labels
    s = s.replace("Mech_", "Mech: ")
    s = s.replace("Mech = ", "Mech: ")

    s = s.replace("__b", "")
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


def make_figm1(model_out_dir: Path, out_dir: Path) -> Path:
    d = _read_model_npz(model_out_dir)

    y_test = d["y_test"].astype(int)
    p_logit = d["p_logit_test"].astype(float)
    p_gb = d["p_gb_test"].astype(float)
    pred_cdr = d["pred_cdr"].astype(int)

    # ROC
    fpr_l, tpr_l, _ = roc_curve(y_test, p_logit)
    fpr_g, tpr_g, _ = roc_curve(y_test, p_gb)
    auc_l = roc_auc_score(y_test, p_logit)
    auc_g = roc_auc_score(y_test, p_gb)

    tn, fp, fn, tp = confusion_matrix(y_test, pred_cdr).ravel()
    tpr_cdr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr_cdr = fp / (fp + tn) if (fp + tn) else 0.0

    # PR
    prec_l, rec_l, _ = precision_recall_curve(y_test, p_logit)
    prec_g, rec_g, _ = precision_recall_curve(y_test, p_gb)
    ap_l = average_precision_score(y_test, p_logit)
    ap_g = average_precision_score(y_test, p_gb)

    tn, fp, fn, tp = confusion_matrix(y_test, pred_cdr).ravel()
    precision_cdr = tp / (tp + fp) if (tp + fp) else 0.0
    recall_cdr = tp / (tp + fn) if (tp + fn) else 0.0

    baseline = float(np.mean(y_test))

    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.10,
        subplot_titles=("ROC (test set)", "Precision–Recall (test set)")
    )

    # ROC
    fig.add_trace(go.Scatter(x=fpr_l, y=tpr_l, mode="lines",
                             name=f"Logit (AUROC={auc_l:.3f})"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fpr_g, y=tpr_g, mode="lines",
                             name=f"GB (AUROC={auc_g:.3f})"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[fpr_cdr], y=[tpr_cdr], mode="markers",
                             name="CDR (pt)", marker=dict(size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Chance", opacity=0.45), row=1, col=1)

    fig.update_xaxes(title_text="FPR", row=1, col=1)
    fig.update_yaxes(title_text="TPR (sensitivity)", row=1, col=1)

    # PR
    fig.add_trace(go.Scatter(x=rec_l, y=prec_l, mode="lines",
                             name=f"Logit (AP={ap_l:.3f})"), row=1, col=2)
    fig.add_trace(go.Scatter(x=rec_g, y=prec_g, mode="lines",
                             name=f"GB (AP={ap_g:.3f})"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[recall_cdr], y=[precision_cdr], mode="markers",
                             name="CDR (pt)", marker=dict(size=10)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode="lines",
                             name=f"Base={baseline:.3%}", opacity=0.55), row=1, col=2)

    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=420,
        width=1100,
        margin=dict(l=70, r=30, t=90, b=60),
        font=dict(size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.08,
            xanchor="left", x=0.0,
            font=dict(size=12)
        ),
    )

    out_path = Path(out_dir) / "figm1.png"
    fig.write_image(out_path, scale=3)
    return out_path


def _confmat_trace(y_true, y_pred, *, zmax: int) -> go.Heatmap:
    cm = confusion_matrix(y_true, y_pred)
    z = cm.astype(int)
    text = [[f"{z[i, j]:,}" for j in range(z.shape[1])] for i in range(z.shape[0])]

    return go.Heatmap(
        z=z,
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=16),
        zmin=0, zmax=int(zmax),
        showscale=False,
        hovertemplate="%{y} / %{x}<br>Count: %{z:,}<extra></extra>",
    )


def make_figm2(model_out_dir: Path, out_dir: Path) -> Path:
    d = _read_model_npz(model_out_dir)

    y_test = d["y_test"].astype(int)
    pred_cdr = d["pred_cdr"].astype(int)
    pred_logit = d["pred_logit"].astype(int)
    pred_gb = d["pred_gb"].astype(int)

    cm_max = int(max(
        confusion_matrix(y_test, pred_cdr).max(),
        confusion_matrix(y_test, pred_logit).max(),
        confusion_matrix(y_test, pred_gb).max(),
    ))

    fig = make_subplots(
        rows=1, cols=3,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Confusion: PECARN CDR",
            "Confusion: Logistic (operating point)",
            "Confusion: Boosting (operating point)"
        )
    )
    fig.add_trace(_confmat_trace(y_test, pred_cdr, zmax=cm_max), row=1, col=1)
    fig.add_trace(_confmat_trace(y_test, pred_logit, zmax=cm_max), row=1, col=2)
    fig.add_trace(_confmat_trace(y_test, pred_gb, zmax=cm_max), row=1, col=3)

    for c in [1, 2, 3]:
        fig.update_yaxes(autorange="reversed", row=1, col=c)

    fig.update_layout(
        template="plotly_white",
        height=360,
        width=1100,
        margin=dict(l=60, r=30, t=90, b=50),
        font=dict(size=14),
    )

    out_path = Path(out_dir) / "figm2.png"
    fig.write_image(out_path, scale=3)
    return out_path


def make_figm3(model_out_dir: Path, out_dir: Path, *, top_k: int = 20) -> Path:
    d = _read_model_npz(model_out_dir)

    # ---------- Panel A: OR forest from exported logit coefs ----------
    if ("logit_feature_names" not in d) or ("logit_coef" not in d):
        raise KeyError("Missing logit_feature_names/logit_coef in model_outputs.npz. "
                       "Make sure model.py exports them.")

    fn = [str(x) for x in d["logit_feature_names"]]
    coef = d["logit_coef"].astype(float).reshape(-1)

    df_or = pd.DataFrame({"feature": fn, "coef": coef})
    df_or["abs_coef"] = df_or["coef"].abs()
    df_or = df_or.sort_values("abs_coef", ascending=False).head(top_k).copy()
    df_or["OR"] = np.exp(df_or["coef"])
    df_or["label"] = df_or["feature"].map(_shorten_feature_name)
    df_or = df_or.sort_values("OR", ascending=True)

    fig_or = go.Figure()
    fig_or.add_trace(go.Scatter(
        x=df_or["OR"],
        y=df_or["label"],
        mode="markers",
        marker=dict(size=10, opacity=0.85),
        customdata=df_or["coef"].to_numpy(),
        hovertemplate="Feature: %{y}<br>OR: %{x:.3f}<br>coef: %{customdata:.3f}<extra></extra>",
        showlegend=False
    ))

    # ---------- Panel B: Permutation importance from CSV exported by model.py ----------
    imp_path = Path(model_out_dir) / "perm_importance.csv"
    if not imp_path.exists():
        raise FileNotFoundError(
            f"Missing {imp_path}. Please patch model.py to export permutation importance, then re-run model.py."
        )

    imp = pd.read_csv(imp_path)
    # keep top_k, plot smallest at bottom -> nice horizontal bar
    imp = imp.sort_values("mean", ascending=False).head(top_k).copy()
    imp["label"] = imp["feature"].map(_shorten_feature_name)
    imp = imp.sort_values("mean", ascending=True)

    fig_pi = go.Figure()
    fig_pi.add_trace(go.Bar(
        x=imp["mean"],
        y=imp["label"],
        orientation="h",
        opacity=0.9,
        error_x=dict(type="data", array=imp["std"], visible=True),
        customdata=imp["std"].to_numpy(),
        hovertemplate="Feature: %{y}<br>ΔAP: %{x:.5f} ± %{customdata:.5f}<extra></extra>",
        showlegend=False
    ))

    # ---------- Combine ----------
    combo = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.10,
        subplot_titles=("A. Logistic odds ratios", "B. Boosting permutation importance (AP)")
    )

    for tr in fig_or.data:
        combo.add_trace(tr, row=1, col=1)
    for tr in fig_pi.data:
        combo.add_trace(tr, row=1, col=2)

    combo.add_vline(x=1.0, line_width=2, opacity=0.55, row=1, col=1)

    combo.update_xaxes(
        type="log",
        title_text="Odds ratio (log)",
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        row=1, col=1
    )
    combo.update_yaxes(automargin=True, row=1, col=1)

    combo.update_xaxes(
        title_text="ΔAP after permutation",
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        row=1, col=2
    )
    combo.update_yaxes(automargin=True, row=1, col=2)

    combo.update_layout(
        template="plotly_white",
        height=520,
        width=1250,
        margin=dict(l=50, r=30, t=90, b=60),
        font=dict(size=14),
        showlegend=False,
    )
    for ann in combo.layout.annotations:
        ann.font = dict(size=14)

    out_path = Path(out_dir) / "figm3.png"
    combo.write_image(out_path, scale=3)
    return out_path

def _read_model_npz(model_out_dir: Path) -> dict:
    npz_path = Path(model_out_dir) / "model_outputs.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}. Run model.py first.")
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def make_figm4(model_out_dir: Path, out_dir: Path) -> Path:
    d = _read_model_npz(model_out_dir)

    # read needed arrays
    pred_cdr_orig = d["pred_cdr_orig"].astype(int)
    pred_cdr_pert = d["pred_cdr_pert"].astype(int)
    pred_logit_orig = d["pred_logit_orig"].astype(int)
    pred_logit_pert = d["pred_logit_pert"].astype(int)
    pred_gb_orig = d["pred_gb_orig"].astype(int)
    pred_gb_pert = d["pred_gb_pert"].astype(int)

    p_logit_test = d["p_logit_test"].astype(float)
    p_logit_pert = d["p_logit_pert_test"].astype(float)
    p_gb_test = d["p_gb_test"].astype(float)
    p_gb_pert = d["p_gb_pert_test"].astype(float)

    # -----------------------------
    # Figure pieces
    # -----------------------------
    rates = pd.DataFrame([
        {"model": "PECARN CDR", "setting": "Original",  "ct_recommend_rate": float(pred_cdr_orig.mean())},
        {"model": "PECARN CDR", "setting": "Perturbed", "ct_recommend_rate": float(pred_cdr_pert.mean())},
        {"model": "Logistic",   "setting": "Original",  "ct_recommend_rate": float(pred_logit_orig.mean())},
        {"model": "Logistic",   "setting": "Perturbed", "ct_recommend_rate": float(pred_logit_pert.mean())},
        {"model": "Boosting (calibrated)", "setting": "Original",  "ct_recommend_rate": float(pred_gb_orig.mean())},
        {"model": "Boosting (calibrated)", "setting": "Perturbed", "ct_recommend_rate": float(pred_gb_pert.mean())},
    ])

    fig_rate = px.bar(
        rates, x="model", y="ct_recommend_rate", color="setting", barmode="group",
        text=rates["ct_recommend_rate"].map(lambda v: f"{v:.1%}")
    )
    fig_rate.update_traces(textposition="outside", cliponaxis=False, opacity=0.9)
    fig_rate.update_layout(
        template="plotly_white",
        yaxis_title="CT recommendation rate",
        xaxis_title=None,
        margin=dict(l=70, r=30, t=70, b=60),
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_rate.update_yaxes(tickformat=".0%")

    delta = pd.DataFrame({
        "delta_p": np.concatenate([p_logit_pert - p_logit_test, p_gb_pert - p_gb_test]),
        "model": (["Logistic"] * len(p_logit_test)) + (["Boosting (calibrated)"] * len(p_gb_test))
    })

    fig_delta = px.violin(delta, x="model", y="delta_p", box=True, points=False)
    fig_delta.update_layout(
        template="plotly_white",
        yaxis_title="Δ predicted risk (perturbed − original)",
        xaxis_title=None,
        margin=dict(l=70, r=30, t=70, b=60),
        font=dict(size=14),
    )
    fig_delta.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    # -----------------------------
    # Combine into figm4
    # -----------------------------
    def add_all_traces(dst_fig, src_fig, row, col):
        for tr in src_fig.data:
            dst_fig.add_trace(tr, row=row, col=col)

    combo12 = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.12,
        subplot_titles=(
            "A. CT recommendation rate (Original vs Perturbed)",
            "B. Δ predicted risk distribution (Perturbed − Original)"
        )
    )
    add_all_traces(combo12, fig_rate, 1, 1)
    add_all_traces(combo12, fig_delta, 1, 2)

    combo12.update_yaxes(tickformat=".0%", title_text="CT recommendation rate", row=1, col=1)
    combo12.update_xaxes(title_text=None, row=1, col=1)

    combo12.update_yaxes(title_text="Δ predicted risk", row=1, col=2,
                         showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    combo12.update_xaxes(title_text=None, row=1, col=2)

    combo12.update_layout(
        template="plotly_white",
        width=1250,
        height=520,
        margin=dict(l=70, r=30, t=110, b=70),
        font=dict(size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1.0
        )
    )
    for ann in combo12.layout.annotations:
        ann.font = dict(size=14)

    out_path = Path(out_dir) / "figm4.png"
    combo12.write_image(out_path, scale=3)
    return out_path


def make_figm5(model_out_dir: Path, out_dir: Path) -> Path:
    d = _read_model_npz(model_out_dir)

    p_logit_test = d["p_logit_test"].astype(float)
    p_logit_pert = d["p_logit_pert_test"].astype(float)
    p_gb_test = d["p_gb_test"].astype(float)
    p_gb_pert = d["p_gb_pert_test"].astype(float)

    def scatter_before_after(p0, p1, title):
        df_sc = pd.DataFrame({"original": p0, "perturbed": p1})
        fig = px.scatter(df_sc, x="original", y="perturbed", opacity=0.35)

        m0 = float(np.nanmin([np.nanmin(p0), np.nanmin(p1)]))
        m1 = float(np.nanmax([np.nanmax(p0), np.nanmax(p1)]))
        fig.add_trace(go.Scatter(x=[m0, m1], y=[m0, m1], mode="lines", name="y=x", opacity=0.7))

        fig.update_layout(
            template="plotly_white",
            title=dict(text=title, x=0.02, xanchor="left"),
            xaxis_title="Predicted risk (original)",
            yaxis_title="Predicted risk (perturbed)",
            margin=dict(l=70, r=30, t=70, b=60),
            font=dict(size=14),
        )
        fig.update_xaxes(tickformat=".2%")
        fig.update_yaxes(tickformat=".2%", showgrid=True, gridcolor="rgba(0,0,0,0.06)")
        return fig

    fig_sc_log = scatter_before_after(
        p_logit_test, p_logit_pert,
        "Logistic regression: predicted risk before vs after perturbation"
    )
    fig_sc_gb = scatter_before_after(
        p_gb_test, p_gb_pert,
        "Boosting (calibrated): predicted risk before vs after perturbation"
    )

    # helper: move traces into subplots
    def add_all_traces(dst_fig, src_fig, row, col, *, secondary_y=False):
        for tr in src_fig.data:
            dst_fig.add_trace(tr, row=row, col=col, secondary_y=secondary_y)

    combo34 = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.10,
        subplot_titles=(
            "C. Logistic: predicted risk (original vs perturbed)",
            "D. Boosting: predicted risk (original vs perturbed)"
        )
    )

    add_all_traces(combo34, fig_sc_log, row=1, col=1)
    add_all_traces(combo34, fig_sc_gb,  row=1, col=2)

    # Match axes ranges (like your COMBO 2)
    def axis_minmax(p0, p1):
        m0 = float(np.nanmin([np.nanmin(p0), np.nanmin(p1)]))
        m1 = float(np.nanmax([np.nanmax(p0), np.nanmax(p1)]))
        return m0, m1

    m0_l, m1_l = axis_minmax(p_logit_test, p_logit_pert)
    m0_g, m1_g = axis_minmax(p_gb_test, p_gb_pert)
    m0 = min(m0_l, m0_g)
    m1 = max(m1_l, m1_g)

    # Left panel: shared range from data
    combo34.update_xaxes(range=[m0, m1], tickformat=".2%", title_text="Predicted risk (original)", row=1, col=1)
    combo34.update_yaxes(range=[m0, m1], tickformat=".2%", title_text="Predicted risk (perturbed)", row=1, col=1,
                         showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    # Right panel: FIXED range [0, 0.40] (exactly like your snippet)
    combo34.update_xaxes(range=[0.0, 0.40], tickformat=".2%", title_text="Predicted risk (original)", row=1, col=2)
    combo34.update_yaxes(range=[0.0, 0.40], tickformat=".2%", title_text="Predicted risk (perturbed)", row=1, col=2,
                         showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    combo34.update_layout(
        template="plotly_white",
        width=1250,
        height=520,
        margin=dict(l=70, r=30, t=110, b=70),
        font=dict(size=14),
        showlegend=False
    )
    for ann in combo34.layout.annotations:
        ann.font = dict(size=14)

    out_path = Path(out_dir) / "figm5.png"
    combo34.write_image(out_path, scale=3)
    return out_path
# ============================================================
# Model-dependent figures placeholder (figm1..)
# ============================================================

def make_model_figures(artifacts: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    """
    Placeholder for figm1..figm5 that depend on model outputs.

    artifacts should include things like:
      - y_test
      - p_logit_test, p_gb_test
      - pred_cdr, pred_logit, pred_gb
      - and any perturbed predictions you used

    We'll fill this after model.py is finalized.
    """
    _ = artifacts
    _ensure_dir(out_dir)
    return {}


# ============================================================
# One-click driver (clean-only + glm-only)
# ============================================================

def make_all_figures(
    clean_csv: Path,
    out_dir: Path = Path("figs"),
) -> Dict[str, Path]:
    out_dir = _ensure_dir(out_dir)
    results: Dict[str, Path] = {}

    results["fig1"]  = make_fig1(clean_csv, out_dir)
    results["fig2"]  = make_fig2(clean_csv, out_dir)
    results["fig3"]  = make_fig3(clean_csv, out_dir)
    results["figf1"] = make_figf1(clean_csv, out_dir)
    results["figf2"] = make_figf2(clean_csv, out_dir)
    results["figf3"] = make_figf3(clean_csv, out_dir)
    results["figs1"] = make_figs1(clean_csv, out_dir)
    results["figm1"] = make_figm1(out_dir, out_dir)
    results["figm2"] = make_figm2(out_dir, out_dir)
    results["figm3"] = make_figm3(out_dir, out_dir, top_k=20)
    results["figm4"] = make_figm4(out_dir, out_dir)
    results["figm5"] = make_figm5(out_dir, out_dir)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="figs")
    args = parser.parse_args()

    res = make_all_figures(Path(args.clean_csv), Path(args.out_dir))
    for k, v in res.items():
        print(f"[figures] {k}: {v}")