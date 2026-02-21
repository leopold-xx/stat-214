#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Layout:
#   lab1/
#     code/ (run.sh, clean.py, models.py, figures.py, environment.yaml)
#     data/ (raw csv, cleaned_data.csv)
#     model_output/
#     figs/
#     report/ (lab1.ipynb, lab1.pdf)
# ------------------------------------------------------------

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CODE_DIR}/.." && pwd)"

DATA_DIR="${ROOT_DIR}/data"
MODEL_OUT_DIR="${ROOT_DIR}/model_output"
FIGS_DIR="${ROOT_DIR}/figs"
REPORT_DIR="${ROOT_DIR}/report"
REPORT_IPYNB="${REPORT_DIR}/lab1.ipynb"
REPORT_PDF="${REPORT_DIR}/lab1.pdf"

CLEANED_CSV="${DATA_DIR}/cleaned_data.csv"

ENV_NAME="stat214"
ENV_YAML="${CODE_DIR}/environment.yaml"

mkdir -p "${DATA_DIR}" "${MODEL_OUT_DIR}" "${FIGS_DIR}" "${REPORT_DIR}"

echo "[run] ROOT_DIR=${ROOT_DIR}"

# ------------------------------------------------------------
# 0) Conda: ensure env matches environment.yaml, then activate
# ------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "[ERROR] conda not found in PATH."
  echo "        Open Anaconda/Miniconda terminal OR add conda to PATH."
  exit 1
fi

if [[ ! -f "${ENV_YAML}" ]]; then
  echo "[ERROR] ${ENV_YAML} not found."
  exit 1
fi

# Create or update env from YAML
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[run] updating env '${ENV_NAME}' from ${ENV_YAML}"
  conda env update -n "${ENV_NAME}" -f "${ENV_YAML}" --prune
else
  echo "[run] creating env '${ENV_NAME}' from ${ENV_YAML}"
  conda env create -n "${ENV_NAME}" -f "${ENV_YAML}"
fi

echo "[run] activating conda env: ${ENV_NAME}"
conda activate "${ENV_NAME}"

# Optional: if your YAML uses only pip deps, ensure pip itself is OK
python -m pip install --quiet --upgrade pip

# ------------------------------------------------------------
# 1) Find raw CSV in data/ (exclude cleaned_data.csv)
# ------------------------------------------------------------
RAW_CSV="$(find "${DATA_DIR}" -maxdepth 1 -type f -name "*.csv" ! -name "cleaned_data.csv" -print | head -n 1 || true)"
if [[ -z "${RAW_CSV}" ]]; then
  echo "[ERROR] No raw CSV found in ${DATA_DIR}"
  echo "        Put the original dataset CSV into data/ (any .csv, excluding cleaned_data.csv)."
  conda deactivate
  exit 1
fi
echo "[run] RAW_CSV=${RAW_CSV}"

# ------------------------------------------------------------
# 2) Run clean.py -> data/cleaned_data.csv
# ------------------------------------------------------------
echo "[run] running clean.py -> ${CLEANED_CSV}"
python - <<PY
import sys
from pathlib import Path

ROOT = Path(r"${ROOT_DIR}")
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from clean import clean_data

raw_csv = Path(r"${RAW_CSV}")
out_csv = Path(r"${CLEANED_CSV}")

df_clean = clean_data(
    raw_csv,
    drop={"02": True},
    set_na={"07": True, "08": True},
)

out_csv.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(out_csv, index=False)
print(f"[clean] wrote {out_csv} rows={len(df_clean)} cols={df_clean.shape[1]}")
PY

# ------------------------------------------------------------
# 3) Run model.py -> model_output/
# ------------------------------------------------------------
echo "[run] running models.py -> ${MODEL_OUT_DIR}"
python "${CODE_DIR}/models.py" \
  --clean_csv "${CLEANED_CSV}" \
  --out_dir "${MODEL_OUT_DIR}"

if [[ ! -f "${MODEL_OUT_DIR}/model_outputs.npz" ]]; then
  echo "[ERROR] model_outputs.npz not found in ${MODEL_OUT_DIR}"
  echo "        models.py likely failed or wrote to a different filename."
  conda deactivate
  exit 1
fi

# ------------------------------------------------------------
# 4) Copy model artifacts into figs/ (figures.py expects them in out_dir)
# ------------------------------------------------------------
echo "[run] syncing model artifacts into ${FIGS_DIR}"
cp -f "${MODEL_OUT_DIR}/model_outputs.npz" "${FIGS_DIR}/model_outputs.npz"
for f in perm_importance.csv metrics.json confusion_matrices.csv stability_metrics.csv stability_rates.csv; do
  if [[ -f "${MODEL_OUT_DIR}/${f}" ]]; then
    cp -f "${MODEL_OUT_DIR}/${f}" "${FIGS_DIR}/${f}"
  fi
done

# ------------------------------------------------------------
# 5) Run figures.py -> figs/
# ------------------------------------------------------------
echo "[run] running figures.py -> ${FIGS_DIR}"
python "${CODE_DIR}/figures.py" \
  --clean_csv "${CLEANED_CSV}" \
  --out_dir "${FIGS_DIR}"

# ------------------------------------------------------------
# 6) Render report/lab1.ipynb -> report/lab1.pdf via Quarto
# ------------------------------------------------------------
# if [[ -f "${REPORT_IPYNB}" ]]; then
#   if command -v quarto >/dev/null 2>&1; then
#     echo "[run] quarto render ${REPORT_IPYNB} -> ${REPORT_PDF}"
#     (cd "${REPORT_DIR}" && quarto render "lab1.ipynb" --to pdf)
#     if [[ -f "${REPORT_PDF}" ]]; then
#       echo "[run] report generated: ${REPORT_PDF}"
#     else
#       echo "[WARN] quarto finished but ${REPORT_PDF} not found (check quarto output)."
#     fi
#   else
#     echo "[WARN] quarto not found in PATH; skip rendering PDF."
#   fi
# else
#   echo "[WARN] ${REPORT_IPYNB} not found; skip rendering PDF."
# fi

# ------------------------------------------------------------
# 7) Deactivate conda env
# ------------------------------------------------------------
echo "[run] deactivating conda env"
conda deactivate

echo "[run] DONE"
echo "  cleaned: ${CLEANED_CSV}"
echo "  model  : ${MODEL_OUT_DIR}"
echo "  figs   : ${FIGS_DIR}"
echo "  report : ${REPORT_DIR}"
