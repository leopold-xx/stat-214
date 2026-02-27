"""
plots.py

Publication-quality plotting helpers for Lab 1.
Single-column report, with optional side-by-side (half-width) figures.
All figures are saved under figs/ as both PDF and PNG.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from io_utils import fig_path, ensure_dir, get_figs_dir


# Theme

def get_palette():
    """
    Colorblind-friendly palette.
    """
    return {
        "blue": "#1F77B4",
        "orange": "#FF7F0E",
        "purple": "#6A3D9A",
        "gray": "#6B6B6B",
        "black": "#111111",
    }


def set_plot_style():
    """
    Set a restrained, journal-like style.
    Explicitly controls fonts, sizes, lines, grids, and color cycle.
    """
    pal = get_palette()

    mpl.rcParams.update({
        # Size / resolution
        "figure.dpi": 120,
        "savefig.dpi": 300,

        # Fonts (serif for "journal feel", with safe fallbacks)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # Axes aesthetics
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.12,
        "grid.linewidth": 0.6,
        "grid.color": pal["gray"],

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,

        # Legend
        "legend.frameon": False,

        # Explicit color cycle (avoid default)
        "axes.prop_cycle": mpl.cycler(color=[pal["blue"], pal["orange"], pal["purple"], pal["gray"]]),
    })


def get_figsize(layout="single"):
    """
    Standard figure sizes for a single-column report.
    layout:
      - "single": full width
      - "half": for side-by-side figures
      - "tall": for plots that need more vertical room
    """
    if layout == "single":
        return (6.4, 4.2)
    if layout == "half":
        return (3.15, 2.6)
    if layout == "tall":
        return (6.4, 5.2)
    raise ValueError("layout must be one of: single, half, tall")


def _finalize_ax(ax, title=None, xlabel=None, ylabel=None):
    if title is not None:
        ax.set_title(title, pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=8)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=8)

    ax.set_axisbelow(True)
    ax.spines["left"].set_alpha(0.9)
    ax.spines["bottom"].set_alpha(0.9)


def save_figure(fig, filename_base):
    """
    Save both PDF and PNG to figs/ with tight bounding box.
    filename_base: no extension, e.g. "eda_missingness"
    """
    ensure_dir(get_figs_dir())

    pdf_path = fig_path(filename_base + ".pdf")
    png_path = fig_path(filename_base + ".png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, png_path


def _check_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# EDA plots

def plot_outcome_distribution(df, outcome_col, filename_base="eda_outcome_distribution", layout="single"):
    """
    Bar plot of outcome counts (0/1).
    Use neutral + one highlight color (no red/green).
    """
    set_plot_style()
    pal = get_palette()
    _check_columns(df, [outcome_col])

    counts = df[outcome_col].value_counts(dropna=False).sort_index()
    x = [str(i) for i in counts.index.tolist()]
    y = counts.values

    fig, ax = plt.subplots(figsize=get_figsize(layout))
    colors = [pal["gray"] if xi == "0" else pal["blue"] for xi in x]
    ax.bar(x, y, color=colors, edgecolor=pal["black"], linewidth=0.6)

    _finalize_ax(ax, title="Outcome distribution", xlabel=outcome_col, ylabel="Count")
    return save_figure(fig, filename_base)


def plot_missingness_by_column(df, top_k=30, filename_base="eda_missingness_by_column", layout="single"):
    """
    Missing rate by column (top_k). Explicit styling, readable labels.
    """
    set_plot_style()
    pal = get_palette()

    missing_rate = df.isna().mean().sort_values(ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=get_figsize(layout))
    ax.bar(
        range(len(missing_rate)),
        missing_rate.values,
        color=pal["gray"],
        edgecolor=pal["black"],
        linewidth=0.4
    )

    ax.set_xticks(range(len(missing_rate)))
    ax.set_xticklabels(missing_rate.index.tolist(), rotation=60, ha="right")

    _finalize_ax(ax, title=f"Missingness by column (top {top_k})", xlabel="Variable", ylabel="Missing rate")
    return save_figure(fig, filename_base)


def plot_numeric_distribution(df, col, bins=30, filename_base=None, layout="single"):
    """
    Histogram for a numeric column.
    """
    set_plot_style()
    pal = get_palette()
    _check_columns(df, [col])

    data = df[col].dropna().values
    if filename_base is None:
        filename_base = f"eda_hist_{col}"

    fig, ax = plt.subplots(figsize=get_figsize(layout))
    ax.hist(data, bins=bins, color=pal["orange"], edgecolor=pal["black"], linewidth=0.5)

    _finalize_ax(ax, title=f"Distribution of {col}", xlabel=col, ylabel="Count")
    return save_figure(fig, filename_base)


def plot_rate_by_group(df, outcome_col, group_col, filename_base=None, layout="single"):
    """
    Dot plot of outcome rate by group (paper-friendly, low ink).
    """
    set_plot_style()
    pal = get_palette()
    _check_columns(df, [outcome_col, group_col])

    g = df[[outcome_col, group_col]].dropna()
    summary = g.groupby(group_col)[outcome_col].mean().sort_values()

    if filename_base is None:
        filename_base = f"eda_rate_{outcome_col}_by_{group_col}"

    fig, ax = plt.subplots(figsize=get_figsize(layout))
    ax.plot(summary.values, range(len(summary)), marker="o", linestyle="None", color=pal["blue"])
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([str(i) for i in summary.index.tolist()])

    _finalize_ax(ax, title=f"Rate of {outcome_col} by {group_col}", xlabel=f"Mean({outcome_col})", ylabel=group_col)
    return save_figure(fig, filename_base)


# Model evaluation plots

def plot_threshold_tradeoff(metrics_df, filename_base="model_threshold_tradeoff", layout="single"):
    """
    Sensitivity vs specificity across thresholds.
    metrics_df columns: threshold, sensitivity, specificity
    """
    set_plot_style()
    pal = get_palette()
    _check_columns(metrics_df, ["threshold", "sensitivity", "specificity"])

    df = metrics_df.sort_values("threshold")

    fig, ax = plt.subplots(figsize=get_figsize(layout))
    ax.plot(df["threshold"], df["sensitivity"], label="Sensitivity", color=pal["blue"])
    ax.plot(df["threshold"], df["specificity"], label="Specificity", color=pal["orange"])

    _finalize_ax(ax, title="Threshold tradeoff", xlabel="Threshold", ylabel="Value")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower center", ncol=2)
    return save_figure(fig, filename_base)


def plot_roc_curve(roc_df, filename_base="model_roc_curve", title="ROC curve", layout="single"):
    """
    ROC curve from DataFrame columns: fpr, tpr.
    """
    set_plot_style()
    pal = get_palette()
    _check_columns(roc_df, ["fpr", "tpr"])

    fig, ax = plt.subplots(figsize=get_figsize(layout))
    ax.plot(roc_df["fpr"], roc_df["tpr"], color=pal["blue"], label="Model")
    ax.plot([0, 1], [0, 1], color=pal["gray"], linewidth=1.2, linestyle="--", label="Chance")

    _finalize_ax(ax, title=title, xlabel="False positive rate", ylabel="True positive rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right")
    return save_figure(fig, filename_base)


def plot_confusion_matrix(cm, filename_base="model_confusion_matrix", title="Confusion matrix", layout="half"):
    """
    Confusion matrix heatmap (single-hue colormap; no red/green).
    cm: dict with TP, FP, TN, FN
    """
    set_plot_style()
    pal = get_palette()

    for k in ["TP", "FP", "TN", "FN"]:
        if k not in cm:
            raise ValueError(f"Missing key in cm: {k}")

    mat = np.array([[cm["TN"], cm["FP"]],
                    [cm["FN"], cm["TP"]]])

    fig, ax = plt.subplots(figsize=get_figsize(layout))
    im = ax.imshow(mat, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title, pad=10)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color=pal["black"])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_figure(fig, filename_base)


# Stability plots

def plot_stability_comparison(stability_df, filename_base="stability_comparison", layout="single"):
    """
    Compare metric values under different settings.
    Expected columns: setting, metric_name, metric_value
    """
    set_plot_style()
    _check_columns(stability_df, ["setting", "metric_name", "metric_value"])

    pivot = stability_df.pivot_table(
        index="metric_name",
        columns="setting",
        values="metric_value",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=get_figsize(layout))
    pivot.plot(kind="bar", ax=ax, width=0.85)

    _finalize_ax(ax, title="Stability comparison", xlabel="Metric", ylabel="Value")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="Setting", loc="upper right")
    return save_figure(fig, filename_base)
