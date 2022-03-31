# Feature Importance Simulations Pipeline

**Main python script:** `01_run_simulations.py`

- Arguments:
  - `--nreps`: Number of replicates.
  - `--model`: Name of prediction model to run. Default (`None`) uses all models specified in `models.py` config file.
  - `--fi_model`: Name of feature importance estimator to run. Default (`None`) uses all feature importance estimators specified in `models.py` config file.
  - `--config`: Name of config folder (and title of the simulation experiment). 
  - `--alpha`: Rejection probability threshold to use in evaluating metrics like TP, FP, sensitivity, specificitiy, etc. Default is 0.05.
  - `--ignore_cache`: Whether or not to ignore cached results.
  - `--verbose`: Whether or not to print messages.
  - `--parallel_id`: ID for parallelization.
  - `--split_seed`: Seed for data splitting.
  - `--results_path`: Path to save results. Default is `./results`.
  - `--create_rmd`: Whether or not to create R Markdown output file with results summary.
  - `--show_vars`: Max number of features to show in rejection probability plots in the R Markdown. Default (`None`) is to show all variables.
- Example usage in command line: 
```
python 01_run_simulations.py --nreps 100 --config test --split_seed 331 --ignore_cache --create_rmd
```

## Creating the config files

For a starter template, see the `sim_config/test` folder. There are two necessary files:

- `dgp.py`: Script specifying the data-generating process under study. The following variables must be provided:
  - `X_DGP`: Function to generate X data.
  - `X_PARAMS_DICT`: Dictionary of named arguments to pass into the `X_DGP` function.
  - `Y_DGP`: Function to generate y data.
  - `Y_PARAMS_DICT`: Dictionary of named arguments to pass into the `Y_DGP` function.
  - `VARY_PARAM_NAME`: Name of argument (typically in `X_DGP` or `Y_DGP`) to vary across. Note that it is also possible to vary across an argument in an `ESTIMATOR` in very basic simulation setups. See the `test` config folder for an example.
  - `VARY_PARAM_VALS`: Dictionary of named arguments for the `VARY_PARAM_NAME` to take on in the simulation experiment. Note that the value can be any python object, but make sure to keep the key simple for naming and plotting purposes. 
- `models.py`: Script specifying the prediction methods and feature importance estimators under study. The following variables must be provided:
  - `ESTIMATORS`: List of prediction methods to fit. Elements should be of class `ModelConfig`.
    - Note that the class passed into `ModelConfig` should have a `fit` method.
    - Additional arguments to pass to the model class can be specified in a dictionary using the `other_params` arguments in `ModelConfig().
  - `FI_ESTIMATORS`: List of feature importance methods to fit. Elements should be of class `FIModelConfig`.
    - Note that the function passed into `FIModelConfig` should take in the arguments `X`, `y`, `fit` at a minimum. For examples, see `nonlinear_significance/scripts/methods.py`.
    - Additional arguments to pass to the feature importance function can be specified in a dictionary using the `other_params` argument in `FIModelConfig()`.
    - Pair up a prediction model and feature importance estimator by using the same `model_type` ID in both.
    - If a feature importance estimator requires sample splitting (outside of the feature importance function call), use the `splitting_strategy` argument to specify the type of splitting strategy (e.g., `train-test`).

## Simulation Scenarios

- `boolean_linear_dgp`:
  - Entries in X are iid boolean 0-1 with equal probabilities
  - y = linear(X) + error
- `normal_ar1_linear_dgp`:
  - Samples/rows in X are iid normal with mean 0 and covariance Sigma (p x p) where Sigma = AR1(rho) (i.e., toeplitz)
  - y = linear(X) + error
- `normal_block_cor_indep_signal_linear_dgp`:
  - Samples/rows in X are iid normal with mean 0 and covariance Sigma (p x p) where Sigma = block diagonal (and constant correlation within each block)
  - y = linear(X) + error
  - Signal features are chosen to all be part of the same correlated block
- `normal_block_cor_linear_dgp`:
  - X and y are the same as `normal_block_cor_indep_signal_linear_dgp`
  - Samples/rows in X are iid normal with mean 0 and covariance Sigma (p x p) where Sigma is "block diagonal" but only the first (top-leftmost) block has non-zero correlation.
  - y = linear(X) + error
  - One signal feature is chosen from the correlated block. All other signal features are chosen from the independent block.
- `normal_linear_dgp`:
  - Entries in X are iid normal with mean 0
  - y = linear(X) + error
- `normal_lss_dgp`:
  - Entries in X are iid normal with mean 0
  - y = LSS(X) + error with disjoint interaction terms
- `normal_poly_dgp`:
  - Entries in X are iid normal with mean 0
  - y = additive_polynomial_interactions(X) + error with disjoint interaction terms