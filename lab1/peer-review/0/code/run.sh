#!/bin/bash

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stat214

# Run scripts
python clean.py
python models.py
python figures.py

# Deactivate conda environment
conda deactivate