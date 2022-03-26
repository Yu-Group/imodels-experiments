<p align="center">
	<img align="center" width=75% src="https://yu-group.github.io/imodels-experiments/logo_experiments.svg?sanitize=True"> </img> 	 <br/>
	Scripts for easily comparing different experimental aspects of the <a href="https://github.com/csinva/imodels">imodels package.</a>
</p>



# Experimental models

- FIGS <img align="center" style="height:30px;" src="https://yu-group.github.io/imodels-experiments/figs/logo_figs.svg?sanitize=True"> </img> - greedily learn a concise sum of trees
- Hierarchical tree shrinkage 🌱
  - [demo](https://yu-group.github.io/imodels-experiments/notebooks/shrinkage/demo_main.html)
- Stable rules - finding a stable set of rules across different models


# Documentation

Follow these steps to benchmark a new (supervised) model. If you want to benchmark something like feature importance or unsupervised learning, you will have to make more substantial changes (mostly in `01_fit_models.py`)

1. Write the sklearn-compliant model (init, fit, predict, predict_proba for classifiers) and add it somewhere in a local folder or in `imodels`
2. Update configs - create a new folder mimicking an existing folder (e.g. `config.shrinkage`)
   1. Select which datasets you want by modifying `datasets.py` (datasets will be downloaded locally automatically later)
   2. Select which models you want by editing a list similar to `models.py`
3. run `01_fit_models.py`
    - pass the appropriate cmdline args (e.g. model, dataset, config)
    - example command: `python 01_fit_models.py --config shrinkage --classification_or_regression regression --split_seed 0 --interactions_off`
      - `--interactions_off` avoids potential errors when calculating interactions
    - another example command: `python 01_fit_models.py --config shrinkage --classification_or_regression regression --model randomforest --split_seed 0`
    - running everything: loop over `split_seed` + `classification_or_regression`
    - alternatively, to parallelize over a slurm cluster, run `01_submit_fitting.py` with the appropriate loops
4. run `02_aggregate_results.py` (which just combines the output of `01_run_comparisons.py` into a `combined.pkl` file across datasets) for plotting
5. put scripts/notebooks into a subdirectory of the `notebooks` folder (e.g. `notebooks/shrinkage`)


## Config
- Note that any hyperparameters not passed in `ModelConfig ` cannot be aggregated over seeds later on


# Testing

Tests are run via `pytest`

