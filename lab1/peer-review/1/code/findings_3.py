import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io_utils import *
from clean import *


# -----------------------------
# Config
# -----------------------------
FIG_DIR = "../figs"
OUT_DIR = "../data"
FIG_NAME = "finding3_headache_incremental_value_by_signcount.png"
TABLE_NAME = "finding3_headache_incremental_value_summary.csv"

MISSING_CODES = {90, 91, 92, -1}


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def to_binary(series):
    """
    Coerce to numeric, keep only {0,1}. Treat 90/91/92/-1/NaN as missing.
    """
    s = to_numeric(series)
    s = s.replace({90: np.nan, 91: np.nan, 92: np.nan, -1: np.nan})
    s = s.where(s.isin([0, 1]), np.nan)
    return s


def wald_ci(p, n, z=1.96):
    if n <= 0 or np.isnan(p):
        return np.nan, np.nan
    se = np.sqrt(max(p * (1 - p) / n, 0.0))
    lo = max(0.0, p - z * se)
    hi = min(1.0, p + z * se)
    return lo, hi


def compute_sign_count(df, sign_cols):
    """
    Count how many signs are present (==1), per row.
    Missing signs treated as 0 for counting to avoid dropping many rows.
    """
    mats = []
    used = []
    for c in sign_cols:
        if c in df.columns:
            mats.append(to_binary(df[c]).fillna(0.0))
            used.append(c)
    if len(mats) == 0:
        raise ValueError("None of the sign columns exist in the dataframe.")
    X = pd.concat(mats, axis=1)
    return X.sum(axis=1).astype(int), used


def bin_count(k):
    return k if k <= 3 else 4


# -----------------------------
# Core computation
# -----------------------------
def compute_headache_incremental(df, sign_cols, headache_col="headache", ct_col="ct_planned", outcome_col="citbi", min_n=50):
    """
    For each sign-count bin k in {0,1,2,3,4+}, compare:
      - CT planned rate between headache=0 vs headache=1
      - ciTBI rate between headache=0 vs headache=1
    with 95% CI and subgroup sample sizes.

    Returns a tidy summary table.
    """
    if headache_col not in df.columns:
        raise ValueError(f"Column not found: {headache_col}")

    ct = to_binary(df[ct_col])
    y = to_binary(df[outcome_col])
    h = to_binary(df[headache_col])

    cnt, used_signs = compute_sign_count(df, sign_cols)
    cnt_bin = cnt.map(bin_count)

    tmp = pd.DataFrame(
        {
            "count_bin": cnt_bin,
            "headache": h,
            "ct": ct,
            "y": y,
        }
    )

    # Keep rows where key outcomes are observed
    tmp = tmp[tmp["ct"].notna() & tmp["y"].notna() & tmp["headache"].notna()].copy()

    rows = []
    for k in [0, 1, 2, 3, 4]:
        for hv in [0, 1]:
            sub = tmp[(tmp["count_bin"] == k) & (tmp["headache"] == hv)]
            n = int(sub.shape[0])
            if n < min_n:
                rows.append(
                    {
                        "count_bin": k,
                        "label": f"{k}" if k < 4 else "4+",
                        "headache": hv,
                        "n": n,
                        "ct_rate": np.nan,
                        "ct_ci_lo": np.nan,
                        "ct_ci_hi": np.nan,
                        "citbi_rate": np.nan,
                        "citbi_ci_lo": np.nan,
                        "citbi_ci_hi": np.nan,
                    }
                )
                continue

            p_ct = float(sub["ct"].mean())
            p_y = float(sub["y"].mean())

            ct_lo, ct_hi = wald_ci(p_ct, n)
            y_lo, y_hi = wald_ci(p_y, n)

            rows.append(
                {
                    "count_bin": k,
                    "label": f"{k}" if k < 4 else "4+",
                    "headache": hv,
                    "n": n,
                    "ct_rate": p_ct,
                    "ct_ci_lo": ct_lo,
                    "ct_ci_hi": ct_hi,
                    "citbi_rate": p_y,
                    "citbi_ci_lo": y_lo,
                    "citbi_ci_hi": y_hi,
                }
            )

    out = pd.DataFrame(rows)
    out.attrs["used_signs"] = used_signs

    # Optional: compute within-bin differences (headache=1 minus headache=0)
    diffs = []
    for k in [0, 1, 2, 3, 4]:
        a = out[(out["count_bin"] == k) & (out["headache"] == 0)].iloc[0]
        b = out[(out["count_bin"] == k) & (out["headache"] == 1)].iloc[0]
        diffs.append(
            {
                "count_bin": k,
                "label": f"{k}" if k < 4 else "4+",
                "delta_ct_rate_h1_minus_h0": (b["ct_rate"] - a["ct_rate"]) if np.isfinite(a["ct_rate"]) and np.isfinite(b["ct_rate"]) else np.nan,
                "delta_citbi_rate_h1_minus_h0": (b["citbi_rate"] - a["citbi_rate"]) if np.isfinite(a["citbi_rate"]) and np.isfinite(b["citbi_rate"]) else np.nan,
                "n_h0": int(a["n"]),
                "n_h1": int(b["n"]),
            }
        )
    diff_df = pd.DataFrame(diffs)

    return out, diff_df


# -----------------------------
# Plot
# -----------------------------
def plot_headache_incremental(summary, title=None):
    """
    Two-panel plot:
      Top: CT planned rate by sign-count bin, headache=0 vs 1
      Bottom: ciTBI rate by sign-count bin, headache=0 vs 1
    With 95% CI. Also prints n (h=0/h=1) under x-axis per bin.
    """
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    blue = "#1f77b4"     # headache=0
    orange = "#ff7f0e"   # headache=1
    gray = "#4d4d4d"

    # Build wide per bin for plotting convenience
    bins = [0, 1, 2, 3, 4]
    labels = ["0", "1", "2", "3", "4+"]

    def get_row(k, hv):
        return summary[(summary["count_bin"] == k) & (summary["headache"] == hv)].iloc[0]

    x = np.arange(len(bins))
    dx = 0.13  # horizontal offset for the two groups

    # Extract arrays for CT + ciTBI
    ct0 = np.array([get_row(k, 0)["ct_rate"] for k in bins], dtype=float)
    ct1 = np.array([get_row(k, 1)["ct_rate"] for k in bins], dtype=float)
    ct0_lo = np.array([get_row(k, 0)["ct_ci_lo"] for k in bins], dtype=float)
    ct0_hi = np.array([get_row(k, 0)["ct_ci_hi"] for k in bins], dtype=float)
    ct1_lo = np.array([get_row(k, 1)["ct_ci_lo"] for k in bins], dtype=float)
    ct1_hi = np.array([get_row(k, 1)["ct_ci_hi"] for k in bins], dtype=float)

    y0 = np.array([get_row(k, 0)["citbi_rate"] for k in bins], dtype=float)
    y1 = np.array([get_row(k, 1)["citbi_rate"] for k in bins], dtype=float)
    y0_lo = np.array([get_row(k, 0)["citbi_ci_lo"] for k in bins], dtype=float)
    y0_hi = np.array([get_row(k, 0)["citbi_ci_hi"] for k in bins], dtype=float)
    y1_lo = np.array([get_row(k, 1)["citbi_ci_lo"] for k in bins], dtype=float)
    y1_hi = np.array([get_row(k, 1)["citbi_ci_hi"] for k in bins], dtype=float)

    n0 = np.array([int(get_row(k, 0)["n"]) for k in bins], dtype=int)
    n1 = np.array([int(get_row(k, 1)["n"]) for k in bins], dtype=int)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.8, 8.2), sharex=True)

    # ---- Panel 1: CT planned ----
    ax1.errorbar(
        x - dx,
        ct0,
        yerr=[ct0 - ct0_lo, ct0_hi - ct0],
        fmt="o-",
        color=blue,
        capsize=4,
        linewidth=2.0,
        markersize=6.5,
        label="Headache = 0",
    )
    ax1.errorbar(
        x + dx,
        ct1,
        yerr=[ct1 - ct1_lo, ct1_hi - ct1],
        fmt="o-",
        color=orange,
        capsize=4,
        linewidth=2.0,
        markersize=6.5,
        label="Headache = 1",
    )
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("CT planned rate (95% CI)")
    ax1.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax1.legend(loc="upper left", frameon=False)

    # ---- Panel 2: ciTBI ----
    ax2.errorbar(
        x - dx,
        y0,
        yerr=[y0 - y0_lo, y0_hi - y0],
        fmt="o-",
        color=blue,
        capsize=4,
        linewidth=2.0,
        markersize=6.5,
        label="Headache = 0",
    )
    ax2.errorbar(
        x + dx,
        y1,
        yerr=[y1 - y1_lo, y1_hi - y1],
        fmt="o-",
        color=orange,
        capsize=4,
        linewidth=2.0,
        markersize=6.5,
        label="Headache = 1",
    )

    # ciTBI is usually low; set y-lim based on CI high
    ymax = np.nanmax(np.r_[y0_hi, y1_hi])
    if not np.isfinite(ymax):
        ymax = 0.25
    ax2.set_ylim(0, min(1.0, max(0.12, ymax * 1.35)))
    ax2.set_ylabel("ciTBI rate (95% CI)")
    ax2.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Number of clinical signs present (binned)")

    # n annotations below x-axis: show n(h=0)/n(h=1) for each bin
    for i in range(len(x)):
        ax2.text(
            x[i],
            -0.22,
            f"n0={n0[i]}\nn1={n1[i]}",
            ha="center",
            va="top",
            fontsize=9.5,
            color=gray,
            transform=ax2.get_xaxis_transform(),
            clip_on=False,
        )

    if title is None:
        title = "Finding 3: Headache increases CT planning more than it increases observed ciTBI risk, within symptom-count strata"
    fig.suptitle(title, y=0.98)

    # Reserve space for the n text
    fig.subplots_adjust(bottom=0.22, top=0.92, hspace=0.20)
    return fig


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(FIG_DIR)
    ensure_dir(OUT_DIR)

    df = load_raw_data()
    df = clean_data(df)  # ensure cleaned for consistent processing; you can skip if already cleaned and saved

    # Use the same sign set as Finding 2 for consistency.
    # IMPORTANT: We exclude 'headache' from the count so that we measure headache's incremental value.
    sign_cols_for_count = [
        "loss_of_consciousness",
        "vomiting",
        "ams",
        "palpable_skull_fracture",
        "basilar_skull_fracture_signs",
        "posttraumatic_seizure",
        "scalp_hematoma",
        "neuro_deficit",
        # do NOT include "headache"
    ]

    summary, diffs = compute_headache_incremental(
        df=df,
        sign_cols=sign_cols_for_count,
        headache_col="headache",
        ct_col="ct_planned",
        outcome_col="citbi",
        min_n=50,  # you can lower to 30 if some bins get sparse
    )

    # Export tables
    summary.to_csv(os.path.join(OUT_DIR, TABLE_NAME), index=False)
    diffs.to_csv(os.path.join(OUT_DIR, "finding3_headache_incremental_deltas.csv"), index=False)

    # Plot
    fig = plot_headache_incremental(summary)
    fig.savefig(os.path.join(FIG_DIR, FIG_NAME), dpi=300, bbox_inches="tight")
    plt.close(fig)

    used_signs = summary.attrs.get("used_signs", [])
    print("Saved figure:", os.path.join(FIG_DIR, FIG_NAME))
    print("Saved table:", os.path.join(OUT_DIR, TABLE_NAME))
    print("Saved deltas:", os.path.join(OUT_DIR, "finding3_headache_incremental_deltas.csv"))
    print("Signs used in count (excluding headache):", used_signs)


if __name__ == "__main__":
    main()