"""
io_utils.py

Centralized I/O helpers.
"""

from pathlib import Path
import pandas as pd


# Paths


def get_project_root():
    """Return lab1/ root."""
    return Path(__file__).resolve().parents[1]


def get_data_dir():
    """Return lab1/data/."""
    return get_project_root() / "data"


def get_figs_dir():
    """Return lab1/figs/."""
    return get_project_root() / "figs"


def ensure_dir(path):
    """Create directory if missing."""
    Path(path).mkdir(parents=True, exist_ok=True)


# Default Filenames


def get_filenames():
    """
    Central place to manage default filenames.
    """
    return {
        "raw_csv": "TBI PUD 10-08-2013.csv",
        "doc_xlsx": "TBI PUD Documentation 10-08-2013.xlsx",
        "cleaned_csv": "cleaned.csv",
        "metrics_csv": "metrics.csv",
        "stability_csv": "stability_results.csv",
    }


# Raw dataset


def load_raw_data(path=None):
    """
    Load raw dataset from data/.
    """
    names = get_filenames()
    file_path = Path(path) if path else (get_data_dir() / names["raw_csv"])

    if not file_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {file_path}")

    return pd.read_csv(file_path)


# Cleaned dataset


def save_clean_data(df, filename=None, index=False):
    """
    Save cleaned dataset as CSV under data/.
    """
    names = get_filenames()
    fname = filename if filename else names["cleaned_csv"]

    file_path = get_data_dir() / fname
    df.to_csv(file_path, index=index)

    return file_path


def load_clean_data(filename=None):
    """
    Load cleaned dataset from data/.
    """
    names = get_filenames()
    fname = filename if filename else names["cleaned_csv"]

    file_path = get_data_dir() / fname

    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {file_path}.")

    return pd.read_csv(file_path)


# General tables


def save_table(df, filename, index=False):
    """
    Save any DataFrame under data/ as CSV.
    """
    file_path = get_data_dir() / filename
    df.to_csv(file_path, index=index)
    return file_path


def load_table(filename):
    """
    Load a CSV table from data/.
    """
    file_path = get_data_dir() / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


# Metrics


def save_metrics(df, filename=None):
    """
    Save metrics table under data/.
    """
    names = get_filenames()
    fname = filename if filename else names["metrics_csv"]
    return save_table(df, fname, index=False)


def load_metrics(filename=None):
    """
    Load metrics table from data/.
    """
    names = get_filenames()
    fname = filename if filename else names["metrics_csv"]
    return load_table(fname)


# Stability results


def save_stability_results(df, filename=None):
    """
    Save stability analysis results under data/.
    """
    names = get_filenames()
    fname = filename if filename else names["stability_csv"]
    return save_table(df, fname, index=False)


def load_stability_results(filename=None):
    """
    Load stability results from data/.
    """
    names = get_filenames()
    fname = filename if filename else names["stability_csv"]
    return load_table(fname)


# Figures


def fig_path(filename):
    """
    Return path under figs/ for saving figures.
    """
    ensure_dir(get_figs_dir())
    return get_figs_dir() / filename
