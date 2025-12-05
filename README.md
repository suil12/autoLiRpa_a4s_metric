# Certified Robust Accuracy Metric – Setup & Usage

This guide explains how to set up the project, run the certified robustness metric and see results.

github repo: github.com/suil12/autoLiRpa_a4s_metric
---

## 1. Environment setup

From the project root (`a4s-eval`):

cd /path/to/a4s-eval

Install the package in editable mode inside the uv-managed environment
uv run pip install -e .


This makes `a4s_eval` importable (fixes `ModuleNotFoundError: No module named 'a4s_eval'`).

---

## 2. Running tests for the metric

From the project root:

uv run pytest tests/metrics/model_metrics/test_certified_robust_accuracy.py

This will run unit tests on your `certified_robust_accuracy` implementation.

If you see `ModuleNotFoundError: a4s_eval`, ensure:

"u"v run pip install -e ." has been run

Or again ensure to add to `tests/conftest.py`:
"
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(file), ".."))
if ROOT not in sys.path:
sys.path.insert(0, ROOT)
"

## 3. Running the metric script

Use the evaluation script:

uv run python scripts/eval_certified_robustness.py \
  --dataset-path data/dataset.csv \
  --target-col target \
  --eps 0.05 \
  --device cpu

for a4s_loan_clean and lending_loan_data we have "charged_off" as column target, for breast_cancer_clean we have "target"


What it does:

- Loads the CSV dataset.
- Builds an A4S `DataShape` from the dataframe.
- Trains a PyTorch model.
- Runs the `certified_robust_accuracy` metric (using auto_LiRPA).
- Prints:
  - `certified_robust_accuracy`
  - `nominal_accuracy`
  - `robustness_ratio`
  - `avg_margin`
  - `avg_certified_margin`
  - `eps`
- Saves:
  - `results.json` (raw measures)
  - `measure_<dataset>_<eps>.csv` (for notebooks)

Example: Lending Club cleaned data at different ε:

uv run python scripts/eval_certified_robustness.py \
--dataset-path data/lending_loan_data_clean.csv \
--target-col charged_off \
--eps 0.005 \
--device cpu

uv run python scripts/eval_certified_robustness.py \
--dataset-path data/lending_loan_data_clean.csv \
--target-col charged_off \
--eps 0.05 \
--device cpu

uv run python scripts/eval_certified_robustness.py \
--dataset-path data/lending_loan_data_clean.csv \
--target-col charged_off \
--eps 0.5 \
--device cpu

---

## 4. Saving measure CSVs for analysis and notebook

Example output filenames:

- `measure_a4s_loan_clean_0_005.csv`
- `measure_a4s_loan_clean_0_05.csv`
- `measure_a4s_loan_clean_0_5.csv`

Place these files in the same folder as your analysis notebook (or adjust paths accordingly).

---
