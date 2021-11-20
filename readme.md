![](docs/logo_saps.png)

# Experiments

This directory contains files pertaining to experimental rule-based / interpretable models.

Follow these steps to benchmark a new model:

1. Write the sklearn-compliant model and put it in the `models` folder
2. Select which datasets you want by modifying `config.datasets` (datasets will be downloaded locally)
3. Select which models you want by editing a list similar to `config.saps.models`
4. run `01_run_comparisons.py` then `02_aggregate_comparisons.py` (which just combines pkls into a `combined.pkl` file across datasets)
5. look at the results in notebooks
