<p align="center">
	<img align="center" width=75% src="https://yu-group.github.io/imodels-experiments/docs/logo_experiments.svg?sanitize=True"> </img> 	 <br/>
	Scripts for easily comparing different experimental aspects of the <a href="https://github.com/csinva/imodels">imodels package.</a>
</p>



# Experimental models

| ![](docs/logo_saps.png)               | Tree shrinkage 🌱 | Stable rules                 |
| ------------------------------------- | ---------------- | ---------------------------- |
| Greedily learn a concise sum of trees | Shrunk trees ([demo](https://yu-group.github.io/imodels-experiments/notebooks/shrinkage/demo_main.html))    | Learn a set of stable models |



# Documentation

Follow these steps to benchmark a new model:

1. Write the sklearn-compliant model (init, fit, predict, predict_proba for classifiers) and add it somewhere in imodels
2. Update configs - create a new folder mimicking an existing folder (e.g. `config.saps`)
   1. Select which datasets you want by modifying `datasets.py` (datasets will be downloaded locally automatically later)
   2. Select which models you want by editing a list similar to `models.py`
3. run `01_fit_models.py`
    - pass the appropriate cmdline args (e.g. model, dataset, config)
    - example command: `python 01_fit_models.py --config saps --classification_or_regression regression --split_seed 0`
    - running everything: loop over `split_seed` + `classification_or_regression`
    - alternatively, to parallelize over a slurm cluster, run `01_submit_fitting.py` with the appropriate loops
4. run `02_aggregate_results.py` (which just combines the output of `01_run_comparisons.py` into a `combined.pkl` file across datasets) for plotting
5. look at the results in the `notebooks` folder


## Config
- Note that any hyperparameters not passed in ModelConfig cannot be aggregated over seeds later on


# Testing

Tests are run via `pytest`

