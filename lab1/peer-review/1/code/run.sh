#!/bin/bash

conda activate stat214

jupyter nbconvert --to notebook --execute --inplace data_cleaning_pipeline.ipynb
jupyter nbconvert --to notebook --execute --inplace data_exploration_pipeline.ipynb
python findings_1.py
python findings_2.py
python findings_3.py
jupyter nbconvert --to notebook --execute --inplace model_pipeline.ipynb
