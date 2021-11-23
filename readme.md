<h1 align="center"> imodels🔍 experiments</h1>
<p align="center"> Scripts for easily comparing different experimental aspects of the <a href="https://github.com/csinva/imodels">imodels package.</a>
</p>



# Experimental models

| ![](docs/logo_saps.png)               | Tree shrinkage 🌱 | Stable rules                 |
| ------------------------------------- | ---------------- | ---------------------------- |
| Greedily learn a concise sum of trees | Shrunk trees     | Learn a set of stable models |



# Documentation

Follow these steps to benchmark a new model:

1. Write the sklearn-compliant model (init, fit, predict, predict_proba for classifiers) and add it somewhere in imodels
2. Update configs - create a new folder mimicking an existing folder (e.g. `config.saps`)
   1. Select which datasets you want by modifying `datasets.py` (datasets will be downloaded locally automatically later)
   2. Select which models you want by editing a list similar to `models.py`
3. run `01_run_comparisons.py`
    - pass the appropriate cmdline args (e.g. model, dataset, config)
    - alternatively, to parallelize over a slurm cluster, run `01_submit_comparisons.py`
    - example command: `python 01_run_comparisons.py --config saps --classification_or_regression regression`
4. run `02_aggregate_comparisons.py` (which just combines the output of `01_run_comparisons.py` into a `combined.pkl` file across datasets) for plotting
5. look at the results in notebooks
