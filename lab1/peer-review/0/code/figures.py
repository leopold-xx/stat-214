# figures.py
"""
Publication-quality figure generation for PECARN TBI Lab 1.

Figures:
  1. plot_finding1        — CT ordering rate vs diagnostic yield by symptom & age group
  2. plot_finding2        — Racial disparities in CT utilization vs clinical severity
  3. plot_finding3        — CT use vs yield by race (stacked bars)
  4. plot_stability_model — Model stability under cleaning perturbations (Fig 5)
  5. plot_stability_sensitivity — Sensitivity by model and age group (Fig 6)
  6. plot_stability_finding1   — Before/after symptom yield (stability check)
  7. plot_stability_finding3   — Before/after race CT bars (stability check)

All functions save to ../figures/ by default.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import chi2_contingency

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FIGURES_DIR = os.path.join("..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """Wilson 95% confidence interval for a proportion, returned as percentages."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - margin) * 100, (center + margin) * 100


def clean_binary(series):
    """Replace sentinel 92 (not applicable) with NaN, keep 0/1."""
    return series.replace(92, np.nan)


def group_stats(df, var, group_col, group):
    """Return (n, k, p) for a binary variable within one group."""
    s = clean_binary(df.loc[df[group_col] == group, var]).dropna()
    n = len(s)
    k = int(s.sum())
    p = s.mean() if n > 0 else np.nan
    return n, k, p


def gap_with_ci(df, var, group_col, g1="White", g2="Black", alpha=0.05):
    """
    Difference in proportions (g1 - g2) with Wald 95% CI and chi-square p-value.
    Returns: gap_pp, ci_lo, ci_hi, p_value  (all in percentage points).
    """
    n1, k1, p1 = group_stats(df, var, group_col, g1)
    n2, k2, p2 = group_stats(df, var, group_col, g2)

    if any(x == 0 for x in [n1, n2]) or any(np.isnan(x) for x in [p1, p2]):
        return np.nan, np.nan, np.nan, np.nan

    gap = (p1 - p2) * 100
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) * 100
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lo, ci_hi = gap - z * se, gap + z * se

    contingency = np.array([[k1, n1 - k1], [k2, n2 - k2]])
    _, p_val, _, _ = stats.chi2_contingency(contingency, correction=False)
    return gap, ci_lo, ci_hi, p_val


def build_race_df(df_clean):
    """Recode race into White / Black / Other and drop missing."""
    df_race = df_clean.copy()
    df_race["RaceClean"] = df_race["Race"].replace({3: 90, 4: 90, 5: 90})
    df_race["RaceLabel"] = df_race["RaceClean"].map({1: "White", 2: "Black", 90: "Other"})
    df_race = df_race.dropna(subset=["RaceLabel"]).copy()
    df_race["CTDone"] = df_race["CTDone"].astype(int)
    return df_race


def build_sym_df(df, symptoms, flip_cols=("ActNorm",), severity_cols=("HASeverity",)):
    """
    Compute CT ordering rate and TBI diagnostic yield for each symptom.
    Returns a DataFrame sorted by CT rate (ascending).
    """
    rows = []
    for col, label in symptoms.items():
        if col in flip_cols:
            sub = df[df[col] == 0].copy()
        elif col in severity_cols:
            sub = df[df[col] == 3].copy()
        else:
            sub = df[df[col].isin([1, 2])].copy()

        n_symptom = len(sub)
        ct_rate = sub["CTDone"].mean() * 100 if n_symptom > 0 else np.nan
        sub_ct = sub[sub["CTDone"] == 1]
        n_ct = len(sub_ct)
        tbi_yield = (
            (sub_ct["PosCT"].replace(92, np.nan) == 1).mean() * 100
            if n_ct > 0 else np.nan
        )
        rows.append(dict(
            label=label, ct_rate=ct_rate, tbi_yield=tbi_yield, n_symptom=n_symptom,
        ))
    return pd.DataFrame(rows).sort_values("ct_rate", ascending=True).reset_index(drop=True)


# Symptom dictionaries (shared across figures)
SYMPTOMS_UNDER2 = {
    "AMS":                "Altered mental status",
    "LOCSeparate":        "Loss of consciousness (any)",
    "Seiz":               "Seizure",
    "ActNorm":            "Not acting normally",
    "SFxBas":             "Basilar skull fx signs",
    "SFxPalp":            "Palpable skull fracture",
    "NeuroD":             "Neurological deficit",
    "Hema":               "Scalp hematoma (any)",
    "HemaLoc":            "Non-frontal hematoma",
    "High_impact_InjSev": "High-impact mechanism",
}

SYMPTOMS_OVER2 = {
    "AMS":                "Altered mental status",
    "LOCSeparate":        "Loss of consciousness (any)",
    "Seiz":               "Seizure",
    "HA_verb":            "Headache (any)",
    "HASeverity":         "Severe headache",
    "Vomit":              "Vomiting",
    "Amnesia_verb":       "Amnesia",
    "ActNorm":            "Not acting normally",
    "Dizzy":              "Dizziness",
    "SFxBas":             "Basilar skull fx signs",
    "SFxPalp":            "Palpable skull fracture",
    "NeuroD":             "Neurological deficit",
    "Hema":               "Scalp hematoma (any)",
    "High_impact_InjSev": "High-impact mechanism",
}


# ── Figure 1: CT ordering rate vs diagnostic yield by symptom ─────────────────

def plot_finding1(df_clean, save_path=None):
    """
    Figure 1: Dot plot comparing CT ordering rate (blue) and TBI diagnostic yield
    (red) by symptom, stratified by age group (under 2 / 2 and older).
    """
    df_under2 = df_clean[df_clean["AgeTwoPlus"] == 1].copy()
    df_over2 = df_clean[df_clean["AgeTwoPlus"] == 2].copy()

    sym_under2 = build_sym_df(df_under2, SYMPTOMS_UNDER2, flip_cols=("ActNorm",))
    sym_over2 = build_sym_df(
        df_over2, SYMPTOMS_OVER2,
        flip_cols=("ActNorm",), severity_cols=("HASeverity",),
    )

    COL_CT = "#2166AC"
    COL_YIELD = "#B2182B"

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=False)
    fig.patch.set_facecolor("white")

    def draw_panel(ax, sym_df, title):
        ax.set_facecolor("white")
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_alpha(0.3)

        y = np.arange(len(sym_df))
        for i, row in sym_df.iterrows():
            ax.plot(
                [row["tbi_yield"], row["ct_rate"]], [i, i],
                color="#DDDDDD", linewidth=2.5, zorder=1,
            )
        ax.scatter(sym_df["ct_rate"], y, color=COL_CT, s=120, zorder=3,
                   label="CT utilization rate", edgecolors="white", linewidth=0.8)
        ax.scatter(sym_df["tbi_yield"], y, color=COL_YIELD, s=120, zorder=3,
                   label="TBI yield (CT positive)", edgecolors="white", linewidth=0.8)

        for i, row in sym_df.iterrows():
            ax.text(row["ct_rate"] + 1, i, f"{row['ct_rate']:.0f}%",
                    va="center", fontsize=8.5, color=COL_CT, fontweight="bold")
            ax.text(row["tbi_yield"] - 1, i, f"{row['tbi_yield']:.1f}%",
                    va="center", ha="right", fontsize=8.5,
                    color=COL_YIELD, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(sym_df["label"].tolist(), fontsize=10.5)
        ax.set_ylim(-0.5, len(sym_df) - 0.5)
        ax.set_xlabel("Percentage (%)", fontsize=11)
        ax.grid(axis="x", color="#EEEEEE", linewidth=0.8)
        ax.tick_params(left=False)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.legend(loc="lower right", frameon=False, fontsize=9.5)

    draw_panel(axes[0], sym_under2,
               f"Children < 2 years  (n={len(df_under2):,})\nVerbal symptoms not assessed")
    draw_panel(axes[1], sym_over2,
               f"Children ≥ 2 years  (n={len(df_over2):,})\nFull symptom set")

    fig.suptitle(
        "Gap between CT ordering rate and diagnostic yield by symptom\n"
        "Wider gap = symptom drives CT ordering but rarely confirms TBI",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.02,
        "Yield = TBI on CT among CT-imaged patients only.  "
        "Age split follows Kuppermann et al. (2009) CDR structure.",
        ha="center", fontsize=8.5, color="#666666", style="italic",
    )

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = save_path or os.path.join(FIGURES_DIR, "finding1_symptom_yield.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 2: Racial disparities in CT utilization vs clinical severity ────────

def plot_finding2(df_clean, highmech_threshold=3, save_path=None):
    """
    Figure 2 (Finding 3 in report): White-Black gap plot showing CT utilization
    disparity compared to clinical severity indicators.
    """
    df_race = build_race_df(df_clean)
    df_wb = df_race[df_race["RaceLabel"].isin(["White", "Black"])].copy()

    df_wb["LOCSeparate_bin"] = df_wb["LOCSeparate"].apply(
        lambda x: 1 if x in [1, 2] else (0 if x == 0 else np.nan)
    )
    df_wb["HighMech_bin"] = df_wb["High_impact_InjSev"].apply(
        lambda x: 1 if x >= highmech_threshold
        else (0 if x in [1, 2] and x < highmech_threshold else np.nan)
    )
    df_wb["SFxPalp_bin"] = df_wb["SFxPalp"].apply(
        lambda x: 1 if x in [1, 2] else (0 if x == 0 else np.nan)
    )

    sev_vars = {
        "HighMech_bin":    "High-impact mechanism",
        "AMS":             "Altered mental status",
        "LOCSeparate_bin": "Loss of consciousness",
        "NeuroD":          "Neurological deficit",
        "SFxBas":          "Basilar skull fracture signs",
        "Seiz":            "Post-traumatic seizure",
    }

    rows = []
    for var, label in sev_vars.items():
        gap, ci_lo, ci_hi, pval = gap_with_ci(df_wb, var, "RaceLabel")
        _, _, pw = group_stats(df_wb, var, "RaceLabel", "White")
        _, _, pb = group_stats(df_wb, var, "RaceLabel", "Black")
        rows.append(dict(
            label=label, gap=gap, ci_lo=ci_lo, ci_hi=ci_hi, pval=pval,
            p_white=pw * 100 if pw is not np.nan else np.nan,
            p_black=pb * 100 if pb is not np.nan else np.nan,
            kind="Severity",
        ))

    gap_ct, ci_lo_ct, ci_hi_ct, pval_ct = gap_with_ci(df_wb, "CTDone", "RaceLabel")
    _, _, pw_ct = group_stats(df_wb, "CTDone", "RaceLabel", "White")
    _, _, pb_ct = group_stats(df_wb, "CTDone", "RaceLabel", "Black")
    rows.append(dict(
        label="CT utilization rate", gap=gap_ct, ci_lo=ci_lo_ct, ci_hi=ci_hi_ct,
        pval=pval_ct, p_white=pw_ct * 100, p_black=pb_ct * 100, kind="CT",
    ))

    plot_df = pd.concat([
        pd.DataFrame([r for r in rows if r["kind"] == "Severity"]).sort_values("gap"),
        pd.DataFrame([r for r in rows if r["kind"] == "CT"]),
    ], ignore_index=True)

    df_high = df_wb[df_wb["HighMech_bin"] == 1].copy()
    has_strat = len(df_high) > 50
    if has_strat:
        gap_hi, ci_lo_hi, ci_hi_hi, pval_hi = gap_with_ci(
            df_high, "CTDone", "RaceLabel"
        )
        n_white_hi = (df_high["RaceLabel"] == "White").sum()
        n_black_hi = (df_high["RaceLabel"] == "Black").sum()

    n_white = (df_wb["RaceLabel"] == "White").sum()
    n_black = (df_wb["RaceLabel"] == "Black").sum()

    COL_WHITE = "#2166AC"
    COL_BLACK = "#C0392B"
    COL_CT = "#7B2D8B"
    COL_SEV = "#444444"
    COL_GRID = "#EFEFEF"

    fig = plt.figure(figsize=(15.5, 7.2), dpi=160)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[56, 44], wspace=0.36)
    ax_L = fig.add_subplot(gs[0])
    ax_R = fig.add_subplot(gs[1])

    n_rows = len(plot_df)
    y_pos = np.arange(n_rows)

    for ax in [ax_L, ax_R]:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.25)
        ax.spines["bottom"].set_alpha(0.25)
        ax.grid(axis="x", color=COL_GRID, linewidth=0.8, zorder=0)
        ax.tick_params(axis="x", labelsize=9.5)

    for i, row in plot_df.iterrows():
        is_ct = row["kind"] == "CT"
        color = COL_CT if is_ct else COL_SEV
        height = 0.82 if is_ct else 0.65
        alpha = 0.88 if is_ct else 0.62
        ax_L.barh(i, row["gap"], color=color, alpha=alpha,
                  height=height, edgecolor="white", linewidth=0.4, zorder=2)
        err = np.array([[row["gap"] - row["ci_lo"]], [row["ci_hi"] - row["gap"]]])
        ax_L.errorbar(row["gap"], i, xerr=err, fmt="none",
                      color="#111111", capsize=4, capthick=1.4, elinewidth=1.4, zorder=3)
        label_x = row["ci_hi"] + 0.35
        ax_L.text(
            label_x, i,
            f"+{row['gap']:.1f} pp" if row["gap"] >= 0 else f"{row['gap']:.1f} pp",
            va="center", ha="left", fontsize=8,
            color=COL_CT if is_ct else COL_SEV, fontweight="bold",
        )
        if not np.isnan(row["pval"]):
            pv = row["pval"]
            stars = ("***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns")
            ax_L.text(label_x + 1.8, i, stars, va="center", ha="left", fontsize=9,
                      color="#222222", fontweight="bold" if stars != "ns" else "normal")

    ax_L.axvline(0, color="#333333", linewidth=1.2, alpha=0.8, zorder=4)
    ax_L.set_yticks(y_pos)
    ax_L.set_yticklabels(plot_df["label"], fontsize=10.5)
    ax_L.set_xlabel("Percentage-point difference  (White − Black)", fontsize=10.5)

    ct_row = plot_df[plot_df["kind"] == "CT"].iloc[-1]
    ct_y = plot_df[plot_df["kind"] == "CT"].index[-1]
    ax_L.annotate(
        f"+{ct_row['gap']:.1f} pp\nWhite > Black",
        xy=(ct_row["gap"], ct_y),
        xytext=(ct_row["gap"] + 2.2, ct_y - 0.85),
        fontsize=8.5, color=COL_CT, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COL_CT, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COL_CT, alpha=0.9, lw=0.9),
    )

    sep_y = n_rows - 1.5
    ax_L.axhline(sep_y, color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=1)
    ax_L.text(0.7, sep_y - 0.22,
              "Clinical severity indicators below, CT use above.",
              transform=ax_L.get_yaxis_transform(),
              fontsize=7.5, color="#999999", va="top", ha="center", style="italic")
    ax_L.text(0.0, -0.11,
              "Whiskers = 95% CI (Wald).   *** p<0.001   ** p<0.01   * p<0.05   ns = not significant",
              transform=ax_L.transAxes, fontsize=7.8, color="#666666", va="top")

    bar_h = 0.30
    offset = 0.19
    for i, row in plot_df.iterrows():
        ax_R.barh(i + offset, row["p_white"], height=bar_h,
                  color=COL_WHITE, alpha=0.82, edgecolor="white", zorder=2)
        ax_R.barh(i - offset, row["p_black"], height=bar_h,
                  color=COL_BLACK, alpha=0.82, edgecolor="white", zorder=2)
        ax_R.text(row["p_white"] + 0.4, i + offset, f"{row['p_white']:.1f}%",
                  va="center", fontsize=8.5, color=COL_WHITE, fontweight="bold")
        ax_R.text(row["p_black"] + 0.4, i - offset, f"{row['p_black']:.1f}%",
                  va="center", fontsize=8.5, color=COL_BLACK, fontweight="bold")

    ax_R.axvline(0, color="#333333", linewidth=0.8, alpha=0.6, zorder=4)
    ax_R.axhline(sep_y, color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=1)
    ax_R.set_yticks(y_pos)
    ax_R.set_yticklabels([""] * n_rows)
    ax_R.set_xlabel("Rate within racial group (%)", fontsize=10.5)

    pw = mpatches.Patch(color=COL_WHITE, alpha=0.85, label=f"White  (n={n_white:,})")
    pb = mpatches.Patch(color=COL_BLACK, alpha=0.85, label=f"Black  (n={n_black:,})")
    ax_R.legend(handles=[pw, pb], loc="lower right", frameon=False, fontsize=9.5)

    if has_strat:
        p_hi = f"p = {'<0.001' if pval_hi < 0.001 else f'{pval_hi:.3f}'}"
        strat_txt = (
            "Severity-stratified check\n"
            "(high-mechanism patients only)\n\n"
            f"CT gap = +{gap_hi:.1f} pp  (White − Black)\n"
            f"95% CI [{ci_lo_hi:.1f}, {ci_hi_hi:.1f}]   {p_hi}\n"
            f"White n={n_white_hi:,}  ·  Black n={n_black_hi:,}\n\n"
            "CT gap persists even after\n"
            "   restricting to highest-severity\n"
            "   injury mechanism."
        )
        ax_R.text(0.99, 0.13, strat_txt, transform=ax_R.transAxes,
                  fontsize=8, ha="right", va="bottom", color="#333333", linespacing=1.45,
                  bbox=dict(boxstyle="round,pad=0.55", fc="#FDF6FF",
                            ec=COL_CT, alpha=0.92, lw=1.0))

    fig.suptitle(
        "Black children receive fewer CT scans despite similar clinical severity",
        fontsize=14.5, fontweight="bold", y=0.98, color="#111111",
    )
    ax_L.set_title("White − Black gap (pp) with 95% CI", fontsize=10.5, pad=7, color="#444444")
    ax_R.set_title("Absolute rates by racial group", fontsize=10.5, pad=7, color="#444444")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = save_path or os.path.join(FIGURES_DIR, "finding2_equity.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 3: CT use vs yield by race (stacked bars) ─────────────────────────

def plot_finding3(df_clean, save_path=None):
    """
    Figure 3 (Finding 2 in report): Stacked horizontal bar chart showing
    CT use and TBI yield per 100 children by race.
    """
    df_race = build_race_df(df_clean)
    n_dropped = len(df_clean) - len(df_race)

    df_race["PosCT_clean"] = 0
    mask_ct = df_race["CTDone"] == 1
    df_race.loc[mask_ct, "PosCT_clean"] = (
        df_race.loc[mask_ct, "PosCT"].replace(92, np.nan).fillna(0).astype(int)
    )

    summary = df_race.groupby("RaceLabel").agg(
        n=("CTDone", "size"), ct=("CTDone", "sum"), ct_pos=("PosCT_clean", "sum")
    )
    summary["ct_neg"] = summary["ct"] - summary["ct_pos"]
    summary["no_ct"] = summary["n"] - summary["ct"]
    summary["no_ct_100"] = summary["no_ct"] / summary["n"] * 100
    summary["ct_neg_100"] = summary["ct_neg"] / summary["n"] * 100
    summary["ct_pos_100"] = summary["ct_pos"] / summary["n"] * 100
    summary["ct_rate"] = summary["ct"] / summary["n"]
    summary = summary.sort_values("ct_rate")

    ct_pos_counts = summary["ct_pos"].values
    ct_neg_counts = summary["ct_neg"].values
    _, p_yield, _, _ = chi2_contingency(
        np.array([ct_pos_counts, ct_neg_counts]), correction=False
    )

    c_no = "#D9F2F0"
    c_neg = "#5AB4AC"
    c_pos = "#00441B"

    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(summary))
    labels = summary.index.tolist()

    ax.barh(y, summary["no_ct_100"].to_numpy(), color=c_no,
            edgecolor="white", height=0.62, label="No CT")
    ax.barh(y, summary["ct_neg_100"].to_numpy(),
            left=summary["no_ct_100"].to_numpy(),
            color=c_neg, edgecolor="white", height=0.62, label="CT done, negative")
    ax.barh(y, summary["ct_pos_100"].to_numpy(),
            left=(summary["no_ct_100"] + summary["ct_neg_100"]).to_numpy(),
            color=c_pos, edgecolor="white", height=0.62,
            label="CT done, positive (TBI on CT)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Out of 100 children in each group")
    ax.set_title("CT use vs CT yield by Race (per 100 children)")

    for i, grp in enumerate(labels):
        ct_rate = summary.loc[grp, "ct_rate"] * 100
        ct_yield = (
            (summary.loc[grp, "ct_pos"] / summary.loc[grp, "ct"]) * 100
            if summary.loc[grp, "ct"] > 0 else np.nan
        )
        n_grp = int(summary.loc[grp, "n"])
        lo, hi = wilson_ci(summary.loc[grp, "ct_pos"], summary.loc[grp, "ct"])
        ax.text(101, i,
                f"CT: {ct_rate:.1f}%  |  Yield: {ct_yield:.1f}% [{lo:.1f}–{hi:.1f}]  |  n={n_grp}",
                va="center", fontsize=9.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)

    plt.tight_layout()
    p_str = "< 0.001" if p_yield < 0.001 else f"= {p_yield:.3f}"
    ax.text(0.01, -0.25,
            f"Chi-square test for yield differences across groups: p {p_str}",
            transform=ax.transAxes, fontsize=8.5, color="#555555")
    ax.text(0.01, -0.30,
            f"Note: {n_dropped:,} patients excluded due to missing race data.",
            transform=ax.transAxes, fontsize=8, color="#888888", style="italic")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = save_path or os.path.join(FIGURES_DIR, "finding3_race_bars.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 5: Model stability under cleaning perturbations ────────────────────

def plot_stability_model(results, save_path=None):
    """
    Figure 5: Bar chart comparing sensitivity, specificity, CT rate, and NPV
    across three cleaning conditions (original, 10% threshold, RR filter off).
    """
    METRICS = ["sensitivity", "specificity", "ct_rate", "npv"]
    conditions = {
        "Original\n(2%, RR on)":  ["cdr_orig_pecarn_all", "lr_orig_pecarn_all",  "rf_orig_pecarn_all"],
        "10% threshold\n(RR on)": ["cdr_10pct_pecarn_all", "lr_10pct_pecarn_all", "rf_10pct_pecarn_all"],
        "2% threshold\n(RR off)": ["cdr_norr_pecarn_all",  "lr_norr_pecarn_all",  "rf_norr_pecarn_all"],
    }
    model_labels = ["CDR", "LR", "RF"]
    x = np.arange(len(model_labels))
    bar_width = 0.25
    cond_colors = ["#555555", "#F4A582", "#92C5DE"]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4), sharey=False)
    fig.suptitle(
        "Model Stability: Original vs Cleaning Perturbations\n(PECARN features, all ages)",
        fontsize=11, fontweight="bold",
    )

    for ax, metric in zip(axes, METRICS):
        for ci, (cond_label, keys) in enumerate(conditions.items()):
            vals = [
                results[k].get(metric, float("nan")) if k in results else float("nan")
                for k in keys
            ]
            ax.bar(x + ci * bar_width, vals, width=bar_width,
                   label=cond_label, color=cond_colors[ci], alpha=0.88)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_title(metric.upper(), fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="grey", linewidth=0.4, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric == METRICS[0]:
            ax.set_ylabel("Score")
            ax.legend(fontsize=7, frameon=True, edgecolor="#AAAAAA",
                      facecolor="white", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = save_path or os.path.join(FIGURES_DIR, "stability_model_cleaning.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Figure 6: Sensitivity by model and age group ─────────────────────────────

def plot_stability_sensitivity(results, save_path=None):
    """
    Figure 6: Bar chart of sensitivity by model (CDR, LR, RF) and age group
    (all ages / under 2 / 2 and older), with Kuppermann validation benchmarks.
    """
    age_groups = ["all", "under2", "2plus"]
    age_labels = ["All ages", "Under 2", "2 and older"]
    models = [
        ("cdr_pecarn", "CDR", "#555555"),
        ("lr_pecarn",  "LR",  "#2166AC"),
        ("rf_pecarn",  "RF",  "#B2182B"),
    ]

    x = np.arange(len(age_groups))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for mi, (mkey, mlbl, mcol) in enumerate(models):
        vals = [results.get(f"{mkey}_{ag}", {}).get("sensitivity", np.nan)
                for ag in age_groups]
        bars = ax.bar(x + mi * bar_width, vals, width=bar_width,
                      label=mlbl, color=mcol, alpha=0.82)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.008, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=8,
                        color=mcol, fontweight="bold")

    ax.axhline(0.968, color="#E67E22", linewidth=1.4, linestyle="--", zorder=3,
               label="Kuppermann validation: 96.8% (aged 2+)")
    ax.axhline(1.00, color="#27AE60", linewidth=1.4, linestyle=":", zorder=3,
               label="Kuppermann validation: 100% (under 2)")

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(age_labels, fontsize=11)
    ax.set_ylabel("Sensitivity (ciTBI caught)", fontsize=11)
    ax.set_ylim(0.6, 1.08)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.7)
    ax.legend(frameon=True, edgecolor="#AAAAAA", facecolor="white",
              framealpha=0.9, fontsize=9.5)
    ax.set_title(
        "Sensitivity by model and age group\n"
        "Sensitivity = proportion of ciTBI cases correctly flagged for CT",
        fontsize=12, fontweight="bold", pad=10,
    )
    fig.text(
        0.5, -0.04,
        "CDR = Kuppermann et al. clinical decision rule.  "
        "LR = logistic regression.  RF = random forest.  "
        "Reference lines show Kuppermann et al. (2009) validation sensitivities: "
        "96.8% for children aged 2+ and 100% for children under 2.",
        ha="center", fontsize=8, color="#666666", style="italic",
    )

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = save_path or os.path.join(FIGURES_DIR, "stability_sensitivity.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


# ── Stability figure 4: Before/after finding 2 (racial disparity perturbation) ─

def plot_stability_finding2(df_orig, df_pert, save_path=None):
    """
    Figure 4: Save before/after White-Black gap plots as two separate PNGs —
    original (threshold=3) and perturbed (threshold=2).
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    out_orig = os.path.join(FIGURES_DIR, "stability_finding2_before.png")
    out_pert = os.path.join(FIGURES_DIR, "stability_finding2_after.png")

    plot_finding2(df_orig, highmech_threshold=3, save_path=out_orig)
    plot_finding2(df_pert, highmech_threshold=2, save_path=out_pert)

    print("Saved: stability_finding2_before.png and stability_finding2_after.png")


# ── Main: run all figures ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from clean import clean_data
    from models import run_all_models

    df_raw = pd.read_csv(os.path.join("..", "data", "TBI PUD 10-08-2013.csv"))

    # EDA figures (Findings 1, 2, 3)
    df_eda = clean_data(df_raw.copy(), mode="eda", run_validation=False)
    plot_finding1(df_eda)
    plot_finding2(df_eda, highmech_threshold=3)
    plot_finding3(df_eda)

    # Stability figure 4: racial disparity before/after perturbation
    df_eda_pert = clean_data(df_raw.copy(), mode="eda", run_validation=False,
                             missing_threshold=0.05)
    plot_stability_finding2(df_eda, df_eda_pert)

    # Model figures (Figures 5 & 6)
    df_model = clean_data(df_raw.copy(), mode="model", run_validation=False)

    df_2pct_rr   = clean_data(df_raw.copy(), mode="model", run_validation=False,
                               missing_threshold=0.02, use_rr_filter=True)
    df_10pct_rr  = clean_data(df_raw.copy(), mode="model", run_validation=False,
                               missing_threshold=0.10, use_rr_filter=True)
    df_2pct_norr = clean_data(df_raw.copy(), mode="model", run_validation=False,
                               missing_threshold=0.02, use_rr_filter=False)

    results = {}
    for m in ["cdr", "lr", "rf"]:
        results[f"{m}_orig_pecarn_all"]  = run_all_models(df_2pct_rr,   model_name=m)
        results[f"{m}_10pct_pecarn_all"] = run_all_models(df_10pct_rr,  model_name=m)
        results[f"{m}_norr_pecarn_all"]  = run_all_models(df_2pct_norr, model_name=m)
        results[f"{m}_pecarn_all"]       = run_all_models(df_model, model_name=m,
                                                          age_group="all")
        results[f"{m}_pecarn_under2"]    = run_all_models(df_model, model_name=m,
                                                          age_group="under2")
        results[f"{m}_pecarn_2plus"]     = run_all_models(df_model, model_name=m,
                                                          age_group="2plus")

    plot_stability_model(results)
    plot_stability_sensitivity(results)

    print("\nAll figures saved to ../figures/")