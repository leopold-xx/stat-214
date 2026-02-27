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
FIG_NAME = "finding2_sign_count_vs_ct_and_citbi.png"
TABLE_NAME = "finding2_sign_count_summary.csv"

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


def sign_count(df, sign_cols):
    """
    Count how many signs are present (==1), per row.
    Missing signs are treated as 0 *for counting only* to avoid dropping lots of rows.
    If you prefer strict counting (drop rows with any missing sign), tell me and I'll swap it.
    """
    mat = []
    for c in sign_cols:
        if c not in df.columns:
            continue
        s = to_binary(df[c])
        mat.append(s.fillna(0.0))
    if len(mat) == 0:
        raise ValueError("None of the sign columns exist in the dataframe.")
    X = pd.concat(mat, axis=1)
    return X.sum(axis=1).astype(int)


# -----------------------------
# Core computation
# -----------------------------
def compute_by_signcount(df, sign_cols, ct_col="ct_planned", outcome_col="citbi"):
    ct = to_binary(df[ct_col])
    y = to_binary(df[outcome_col])

    cnt = sign_count(df, sign_cols)

    tmp = pd.DataFrame({"count": cnt, "ct": ct, "y": y})
    tmp = tmp[tmp["ct"].notna() & tmp["y"].notna()].copy()

    # Bin: 0,1,2,3,4+
    def bin_count(k):
        return k if k <= 3 else 4

    tmp["count_bin"] = tmp["count"].map(bin_count)

    rows = []
    for k in [0, 1, 2, 3, 4]:
        sub = tmp[tmp["count_bin"] == k]
        n = int(sub.shape[0])
        if n == 0:
            rows.append(
                {
                    "count_bin": k,
                    "label": f"{k}" if k < 4 else "4+",
                    "n": 0,
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
    return out


# -----------------------------
# Plot
# -----------------------------
def plot_signcount(out, title=None):
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

    blue = "#1f77b4"
    orange = "#ff7f0e"
    gray = "#4d4d4d"

    x = np.arange(out.shape[0])
    labels = out["label"].tolist()

    fig, ax = plt.subplots(figsize=(11.8, 6.6))

    ax.errorbar(
        x,
        out["ct_rate"],
        yerr=[out["ct_rate"] - out["ct_ci_lo"], out["ct_ci_hi"] - out["ct_rate"]],
        fmt="o-",
        color=blue,
        capsize=4,
        linewidth=2.2,
        markersize=7,
        label="CT planned rate",
    )

    ax.errorbar(
        x,
        out["citbi_rate"],
        yerr=[out["citbi_rate"] - out["citbi_ci_lo"], out["citbi_ci_hi"] - out["citbi_rate"]],
        fmt="o-",
        color=orange,
        capsize=4,
        linewidth=2.2,
        markersize=7,
        label="ciTBI rate",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Give headroom and bottom room
    ax.set_ylim(0, 1.05)

    ax.set_xlabel("Number of clinical signs present (binned)")
    ax.set_ylabel("Proportion (95% CI)")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)

    # More cautious title (see section 2 below)
    if title is None:
        title = "Finding 2: CT planning approaches saturation before ciTBI risk becomes high"
    ax.set_title(title, pad=14)

    # Legend outside right, reserve space
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    # n labels: place them BELOW the x-axis label area, and reserve bottom margin
    for i in range(out.shape[0]):
        ax.text(
            x[i],
            -0.16,  # lower than before
            f"n={int(out.loc[i, 'n'])}",
            ha="center",
            va="top",
            fontsize=10,
            color=gray,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

    # Reserve margins explicitly (tight_layout alone is not enough here)
    fig.subplots_adjust(bottom=0.24, right=0.82, top=0.90)

    return fig


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(FIG_DIR)
    ensure_dir(OUT_DIR)

    df = load_raw_data()
    df = clean_data(df)

    # Choose a reasonable “sign set” that exists in your rename_map
    sign_cols = [
        "loss_of_consciousness",
        "vomiting",
        "headache",
        "ams",
        "palpable_skull_fracture",
        "basilar_skull_fracture_signs",
        "posttraumatic_seizure",
        "scalp_hematoma",
        "neuro_deficit",
    ]

    out = compute_by_signcount(
        df=df,
        sign_cols=sign_cols,
        ct_col="ct_planned",
        outcome_col="citbi",
    )

    out.to_csv(os.path.join(OUT_DIR, TABLE_NAME), index=False)

    fig = plot_signcount(out)
    fig.savefig(os.path.join(FIG_DIR, FIG_NAME), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved figure:", os.path.join(FIG_DIR, FIG_NAME))
    print("Saved table:", os.path.join(OUT_DIR, TABLE_NAME))


if __name__ == "__main__":
    main()