<p align="center">
	<img align="center" width=75% src="https://yu-group.github.io/imodels-experiments/logo_experiments.svg?sanitize=True"> </img> 	 <br/>
	Scripts for easily comparing different aspects of the <a href="https://github.com/csinva/imodels">imodels package.</a> Contains code to reproduce <a href="https://arxiv.org/abs/2201.11931">FIGS</a> + <a href="https://arxiv.org/abs/2202.00858">Hierarchical shrinkage</a>
</p>

# Documentation

Follow these steps to benchmark a new (supervised) model. If you want to benchmark something like feature importance or unsupervised learning, you will have to make more substantial changes (mostly in `01_fit_models.py`)

1. Write the sklearn-compliant model (init, fit, predict, predict_proba for classifiers) and add it somewhere in a local folder or in `imodels`
2. Update configs - create a new folder mimicking an existing folder (e.g. `config.interactions`)
   1. Select which datasets you want by modifying `datasets.py` (datasets will be downloaded locally automatically later)
   2. Select which models you want by editing a list similar to `models.py`
3. run `01_fit_models.py`
    - pass the appropriate cmdline args (e.g. model, dataset, config)
    - example command: `python 01_fit_models.py --config interactions --classification_or_regression classification --split_seed 0`
    - another ex.: `python 01_fit_models.py --config interactions --classification_or_regression classification --model randomforest --split_seed 0`
    - running everything: loop over `split_seed` + `classification_or_regression`
    - alternatively, to parallelize over a slurm cluster, run `01_submit_fitting.py` with the appropriate loops
4. run `02_aggregate_results.py` (which just combines the output of `01_run_comparisons.py` into a `combined.pkl` file across datasets) for plotting
5. put scripts/notebooks into a subdirectory of the `notebooks` folder (e.g. `notebooks/interactions`)


## Config
- When running multiple seeds, we want to aggregate over all keys that are not the split_seed
  - If a hyperparameter is not passed in `ModelConfig` (e.g. because we are using parial), it cannot be aggregated over seeds later on
    - The `extra_aggregate_keys={'max_leaf_nodes': n}` is a workaround for this (see configs with `partial` to understand how it works) 


## Testing

Tests are run via `pytest`

# Experimental methods

- **P-FIGS** - pooling data across heterogenous groups for FIGS
- **Stable rules** - finding a stable set of rules across different models


# Working methods

### FIGS: Fast interpretable greedy-tree sums

[📄 Paper](https://arxiv.org/abs/2201.11931), [🔗 Post](https://demos.csinva.io/figs/), [📌 Citation](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=fast+interpretable+greedy-tree+sums&oq=fast#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3ADnPVL74Rop0J%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)

Fast Interpretable Greedy-Tree Sums (FIGS) is an algorithm for fitting concise rule-based models. Specifically, FIGS generalizes CART to simultaneously grow a flexible number of trees in a summation. The total number of splits across all the trees can be restricted by a pre-specified threshold, keeping the model interpretable. Experiments across a wide array of real-world datasets show that FIGS achieves state-of-the-art prediction performance when restricted to just a few splits (e.g. less than 20).

<p align="center">
	<img src="https://demos.csinva.io/figs/diabetes_figs.svg?sanitize=True" width="50%">
</p>  
<p align="center">	
	<i>Example FIGS model. FIGS learns a sum of trees with a flexible number of trees; to make its prediction, it sums the result from each tree.</i>
</p>

### Hierarchical shrinkage: post-hoc regularization for tree-based methods

[📄 Paper](https://arxiv.org/abs/2202.00858), [🔗 Post](https://demos.csinva.io/shrinkage/), [📌 Citation](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=hierarchical+shrinkage+singh&btnG=&oq=hierar#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3Azc6gtLx-aL4J%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)

Hierarchical shrinkage is an extremely fast post-hoc regularization method which works on any decision tree (or tree-based ensemble, such as Random Forest). It does not modify the tree structure, and instead regularizes the tree by shrinking the prediction over each node towards the sample means of its ancestors (using a single regularization parameter). Experiments over a wide variety of datasets show that hierarchical shrinkage substantially increases the predictive performance of individual decision trees and decision-tree ensembles.
