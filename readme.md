<p align="center">
	<img align="center" width=75% src="https://yu-group.github.io/imodels-experiments/logo_experiments.svg?sanitize=True"> </img> 	 <br/>
	Scripts for easily comparing different aspects of the <a href="https://github.com/csinva/imodels">imodels package.</a> Contains code to reproduce <a href="https://arxiv.org/abs/2201.11931">FIGS</a> + <a href="https://arxiv.org/abs/2202.00858">Hierarchical shrinkage</a> + <a href="https://arxiv.org/abs/2205.15135">G-FIGS</a> + <a href="https://arxiv.org/pdf/2307.01932.pdf">MDI+</a>.
</p>

# Documentation

Follow these steps to benchmark a new (supervised) model. 

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


Note: If you want to benchmark feature importances, go to [feature_importance/](https://github.com/Yu-Group/imodels-experiments/tree/master/feature_importance). For benchmarking other tasks such as unsupervised learning, you will have to make more substantial changes (mostly in `01_fit_models.py`).

## Config
- When running multiple seeds, we want to aggregate over all keys that are not the split_seed
  - If a hyperparameter is not passed in `ModelConfig` (e.g. because we are using parial), it cannot be aggregated over seeds later on
    - The `extra_aggregate_keys={'max_leaf_nodes': n}` is a workaround for this (see configs with `partial` to understand how it works) 


## Testing

Tests are run via `pytest`

# Experimental methods

- **Stable rules** - finding a stable set of rules across different models


# Working methods

### FIGS: Fast interpretable greedy-tree sums

[📄 Paper](https://arxiv.org/abs/2201.11931), [🔗 Post](https://csinva.io/imodels/figs.html), [📌 Citation](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=fast+interpretable+greedy-tree+sums&oq=fast#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3ADnPVL74Rop0J%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)

Fast Interpretable Greedy-Tree Sums (FIGS) is an algorithm for fitting concise rule-based models. Specifically, FIGS generalizes CART to simultaneously grow a flexible number of trees in a summation. The total number of splits across all the trees can be restricted by a pre-specified threshold, keeping the model interpretable. Experiments across a wide array of real-world datasets show that FIGS achieves state-of-the-art prediction performance when restricted to just a few splits (e.g. less than 20).

<p align="center">
	<img src="https://demos.csinva.io/figs/diabetes_figs.svg?sanitize=True" width="50%">
</p>  
<p align="center">	
	<i>Example FIGS model. FIGS learns a sum of trees with a flexible number of trees; to make its prediction, it sums the result from each tree.</i>
</p>

### Hierarchical shrinkage: post-hoc regularization for tree-based methods

[📄 Paper](https://arxiv.org/abs/2202.00858) (ICML 2022), [🔗 Post](https://csinva.io/imodels/shrinkage.html), [📌 Citation](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=hierarchical+shrinkage+singh&btnG=&oq=hierar#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3Azc6gtLx-aL4J%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)

Hierarchical shrinkage is an extremely fast post-hoc regularization method which works on any decision tree (or tree-based ensemble, such as Random Forest). It does not modify the tree structure, and instead regularizes the tree by shrinking the prediction over each node towards the sample means of its ancestors (using a single regularization parameter). Experiments over a wide variety of datasets show that hierarchical shrinkage substantially increases the predictive performance of individual decision trees and decision-tree ensembles.

<p align="center">
	<img src="https://demos.csinva.io/shrinkage/shrinkage_intro.svg?sanitize=True" width="75%">
</p>  
<p align="center">	
	<i>HS Example. HS appplies post-hoc regularization to any decision tree by shrinking each node towards its parent.</i>
</p>

### G-FIGS: Group Probability-Weighted Tree Sums for Interpretable Modeling of Heterogeneous Data


[📄 Paper](https://arxiv.org/abs/2202.00858)

Machine learning in high-stakes domains, such as healthcare, faces two critical challenges: (1) generalizing to diverse data distributions given limited training data while (2) maintaining interpretability. To address these challenges, G-FIGS effectively pools data across diverse groups to output a concise, rule-based model. Given distinct groups of instances in a dataset (e.g., medical patients grouped by age or treatment site), G-FIGS first estimates group membership probabilities for each instance. Then, it uses these estimates as instance weights in FIGS (Tan et al. 2022), to grow a set of decision trees whose values sum to the final prediction. G-FIGS achieves state-of-the-art prediction performance on important clinical datasets; e.g., holding the level of sensitivity fixed at 92%, G-FIGS increases specificity for identifying cervical spine injury by up to 10% over CART and up to 3% over FIGS alone, with larger gains at higher sensitivity levels. By keeping the total number of rules below 16 in FIGS, the final models remain interpretable, and we find that their rules match medical domain expertise. All code, data, and models are released on Github.

<p align="center">
	<img src="https://demos.csinva.io/figs/gfigs_intro.svg?sanitize=True" width="90%">
</p>  
<p align="center">	
	<i>G-FIGS 2-step process explained.</i>
</p>


### MDI+: A Flexible Random Forest-Based Feature Importance Framework

[📄 Paper](https://arxiv.org/pdf/2307.01932.pdf), [📌 Citation](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=MDI%2B%3A+A+Flexible+Random+Forest-Based+Feature+Importance+Framework&btnG=#d=gs_cit&t=1690399844081&u=%2Fscholar%3Fq%3Dinfo%3Axc0LcHXE_lUJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)

MDI+ is a novel feature importance framework, which generalizes the popular mean decrease in impurity (MDI) importance score for random forests. At its core, MDI+ expands upon a recently discovered connection between linear regression and decision trees. In doing so, MDI+ enables practitioners to (1) tailor the feature importance computation to the data/problem structure and (2) incorporate additional features or knowledge to mitigate known biases of decision trees. In both real data case studies and extensive real-data-inspired simulations, MDI+ outperforms commonly used feature importance measures (e.g., MDI, permutation-based scores, and TreeSHAP) by substantional margins.
