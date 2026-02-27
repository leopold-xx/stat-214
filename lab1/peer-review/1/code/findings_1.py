import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io_utils import *
from clean import *

# -----------------------------
# Config: paths
# -----------------------------
FIG_DIR = "../figs"
OUT_DIR = "../data"
FIG_NAME = "finding1_ct_vs_citbi.png"
TABLE_NAME = "finding1_ct_vs_citbi_summary.csv"


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_binary(series):
    """
    Coerce to numeric, keep only {0,1}. Treat 90/91/92/-1 as missing.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace({90: np.nan, 91: np.nan, 92: np.nan, -1: np.nan})
    s = s.where(s.isin([0, 1]), np.nan)
    return s


def wald_ci(p, n, z=1.96):
    """
    Wald CI for a proportion. Returns (lo, hi) clipped to [0,1].
    """
    if n <= 0 or np.isnan(p):
        return np.nan, np.nan
    se = np.sqrt(max(p * (1 - p) / n, 0.0))
    lo = max(0.0, p - z * se)
    hi = min(1.0, p + z * se)
    return lo, hi


def compute_rates(df, sign_cols, ct_col="ct_planned", outcome_col="citbi", min_n=50):
    """
    For each sign col, compute:
      P(CT=1 | sign=1), P(ciTBI=1 | sign=1), with 95% CI
    """
    ct = to_binary(df[ct_col])
    y = to_binary(df[outcome_col])

    rows = []
    for col in sign_cols:
        if col not in df.columns:
            continue

        s = to_binary(df[col])
        mask = (s == 1) & ct.notna() & y.notna()

        ct_sub = ct[mask]
        y_sub = y[mask]
        n = int(mask.sum())

        if n < min_n:
            continue

        p_ct = float(ct_sub.mean())
        p_y = float(y_sub.mean())

        ct_lo, ct_hi = wald_ci(p_ct, n)
        y_lo, y_hi = wald_ci(p_y, n)

        rows.append(
            {
                "sign": col,
                "n_sign1": n,
                "ct_rate": p_ct,
                "ct_ci_lo": ct_lo,
                "ct_ci_hi": ct_hi,
                "citbi_rate": p_y,
                "citbi_ci_lo": y_lo,
                "citbi_ci_hi": y_hi,
                "gap_ct_minus_citbi": p_ct - p_y,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("gap_ct_minus_citbi", ascending=True).reset_index(drop=True)
    return out


def plot_dumbbell(out, sign_label_map=None, xlim=(0, 1)):
    """
    Dumbbell plot comparing CT planned rate vs ciTBI rate (conditioned on sign=1).
    """
    if sign_label_map is None:
        sign_label_map = {}

    # --- style: keep it clean, journal-ish ---
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    labels = [sign_label_map.get(s, s) for s in out["sign"].tolist()]
    y = np.arange(out.shape[0])

    fig_h = max(4.5, 0.45 * out.shape[0] + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    # Use colorblind-friendly-ish choices (no red/green pairing)
    color_ct = "#1f77b4"    # blue
    color_y = "#ff7f0e"     # orange
    color_link = "#7f7f7f"  # gray

    # CIs
    ax.hlines(y=y, xmin=out["ct_ci_lo"], xmax=out["ct_ci_hi"], color=color_ct, linewidth=2.2, alpha=0.95)
    ax.hlines(y=y, xmin=out["citbi_ci_lo"], xmax=out["citbi_ci_hi"], color=color_y, linewidth=2.2, alpha=0.95)

    # Points
    ax.plot(out["ct_rate"], y, "o", color=color_ct, markersize=6, label="CT planned rate")
    ax.plot(out["citbi_rate"], y, "o", color=color_y, markersize=6, label="ciTBI rate")

    # Connectors
    for i in range(out.shape[0]):
        ax.plot(
            [out.loc[i, "citbi_rate"], out.loc[i, "ct_rate"]],
            [y[i], y[i]],
            color=color_link,
            linewidth=1.6,
            alpha=0.6,
            zorder=0,
        )

    # Right-side n annotation (helpful for trust)
    for i in range(out.shape[0]):
        ax.text(
            xlim[1] + 0.02,
            y[i],
            f"n={int(out.loc[i, 'n_sign1'])}",
            va="center",
            ha="left",
            fontsize=9,
            color="#444444",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(xlim[0], xlim[1] + 0.14)  # leave room for n labels
    ax.set_xlabel("Rate (95% CI) among sign-positive patients")
    ax.set_ylabel(None)

    ax.set_title(
        "Finding 1: CT planning vs observed ciTBI risk by clinical sign\n\n"
        "Blue = CT planned rate     Orange = ciTBI rate     (both with 95% CI)",
        pad=12
    )

    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.35)

    fig.tight_layout()
    return fig


def main():
    ensure_dir(FIG_DIR)
    ensure_dir(OUT_DIR)

    df_cleaned = load_raw_data()
    df_cleaned = clean_data_without_dropping(df_cleaned)

    # --- choose signs for Finding 1 ---
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

    sign_label_map = {
        "loss_of_consciousness": "Loss of consciousness",
        "vomiting": "Vomiting",
        "headache": "Headache",
        "ams": "Altered mental status",
        "palpable_skull_fracture": "Palpable skull fracture",
        "basilar_skull_fracture_signs": "Basilar skull fracture signs",
        "posttraumatic_seizure": "Post-traumatic seizure",
        "scalp_hematoma": "Scalp hematoma",
        "neuro_deficit": "Neurological deficit",
    }

    out = compute_rates(
        df=df_cleaned,
        sign_cols=sign_cols,
        ct_col="ct_planned",
        outcome_col="citbi",
        min_n=50,
    )

    # export table
    table_path = os.path.join(OUT_DIR, TABLE_NAME)
    out.to_csv(table_path, index=False)

    # plot
    fig = plot_dumbbell(
        out,
        sign_label_map=sign_label_map,
        xlim=(0, 1),
    )

    fig_path = os.path.join(FIG_DIR, FIG_NAME)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved figure:", fig_path)
    print("Saved table:", table_path)
    print("Rows in summary:", out.shape[0])


if __name__ == "__main__":
    main()