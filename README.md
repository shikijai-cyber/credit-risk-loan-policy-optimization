# Credit Risk and Loan Policy Optimisation

## Overview
This project analyses loan-level credit risk data and evaluates policy rules for lending decisions. It combines data cleaning, feature preparation, clustering, predictive modelling, and risk-adjusted policy evaluation.

## Features
- Cleans loan application and outcome data
- Removes leakage and post-decision variables
- Handles missing values, outliers, and categorical encoding
- Builds classification and segmentation workflows
- Evaluates loan approval policies against risk and profitability trade-offs
- Includes methodological reports and data-cleaning notes

## Project structure
```text
src/           Main Python analysis script
notebooks/     Exploratory and modelling notebook
docs/          Reports and cleaning notes
data/raw/      Source CSV placeholder
requirements.txt
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Place the source file in `data/raw/`:
```text
early_2012_2013_loan_sample_with_outcome.csv
```

Run the script:
```bash
python src/loan_policy_analysis.py
```

Or open the notebook:
```bash
jupyter notebook notebooks/loan_policy_midterm_analysis.ipynb
```

## Technologies used
Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Pyomo, Matplotlib, seaborn.

## Portfolio note
The raw loan CSV was not present in the uploaded archive, so the cleaned repository includes a placeholder and documents the expected filename.
