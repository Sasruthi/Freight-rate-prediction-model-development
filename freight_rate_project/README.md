# Freight Rate Prediction

Predicts `posted_rate` ($) for truckload freight based on lane, distance,
equipment, weight, and date.

## Setup

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.12. Requires: pandas, numpy, scikit-learn, lightgbm,
xgboost, catboost, matplotlib, joblib.

## Project structure

```
data/
  train_test.csv                       # 48,000 labeled loads (Jan-Oct 2025)
  validation.csv                       # 12,000 unlabeled loads to predict (Nov-Dec 2025)
  validation_predictions_template.csv  # load_id template to fill
  december_chart_inputs.csv            # 31-row fixed-lane December sweep (predicted_rate filled in place)
src/
  features.py   # shared preprocessing / feature engineering (used by train + predict)
  train.py       # trains & compares 6 models on a time-based holdout, saves the winner
  predict.py     # loads the saved model, writes validation_predictions.csv and fills december_chart_inputs.csv
models/          # saved model + preprocessing artifacts (created by train.py)
reports/
  model_comparison.csv        # holdout metrics for all 6 candidate models
  figures/                     # comparison chart, actual-vs-predicted scatter, December chart
Freight_Rate_Modeling_Report.docx   # full write-up: EDA findings, split rationale, model selection
score.py         # provided scorer (validates output format, renders the December chart)
```

## Run everything

```bash
# 1. Train & compare models (writes models/*.joblib and reports/model_comparison.csv)
python src/train.py

# 2. Generate final predictions
python src/predict.py
#    -> writes validation_predictions.csv (12,000 rows)
#    -> fills data/december_chart_inputs.csv predicted_rate column in place

# 3. Validate output format + render the December chart
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv \
                 --december-predictions data/december_chart_inputs.csv
#    -> scorer_results/candidate_december.png
```

## Approach summary

- **Split**: time-based (last 15% of days held out), not random — the model
  must extrapolate 1-3 months past the training window, and a random split
  would overstate accuracy by letting the model train on days adjacent to
  its test days. See the report for full rationale.
- **Models compared**: Linear Regression, Ridge, Random Forest, XGBoost,
  LightGBM, CatBoost — all on identical engineered features, with early
  stopping tuned for the boosted trees so the comparison is fair.
- **Selected model**: Linear Regression (best RMSE/MAE/MAPE on the holdout;
  full comparison table in `reports/model_comparison.csv` and the report).
- **Handling missing features**: `december_chart_inputs.csv` lacks
  coordinates, `market_index`, and `quote_signal` entirely. `src/features.py`
  treats every such column as optional: present-and-used when available,
  safely imputed with training-set medians (numeric) or `category` dtype
  (categoricals) when absent, with explicit `*_missing` indicator flags.

Full details — EDA findings, data-quality issues, preprocessing steps, and
model-selection reasoning — are in `Freight_Rate_Modeling_Report.docx`.
