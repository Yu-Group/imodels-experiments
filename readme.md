![](docs/logo_saps.png)

# Experiments

This directory contains files pertaining to experimental rule-based / interpretable models.

Follow these steps to benchmark a new model:

1. Write the sklearn-compliant model (init, fit, predict, predict_proba for classifiers) and add it somewhere in imodels
2. Update configs - create a new folder mimicking an existing folder (e.g. `config.saps`)
   1. Select which datasets you want by modifying `datasets.py` (datasets will be downloaded locally automatically later)
   2. Select which models you want by editing a list similar to `models.py`
3. run `01_run_comparisons.py`
    - pass the appropriate cmdline args (e.g. model, dataset, config)
4. run `02_aggregate_comparisons.py` (which just combines pkls into a `combined.pkl` file across datasets)
5. look at the results in notebooks
