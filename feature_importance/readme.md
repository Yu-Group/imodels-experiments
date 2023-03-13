# Feature Importance Simulations Pipeline

This is a basic template to run extensive simulation studies for benchmarking a new feature importance method. The main python script is `01_run_importance_simulations.py` (which is largely based on `../01_fit_models.py`). To run these simulations, follow the three main steps below:

1. Take the feature importance method of interest, and wrap it in a function that has the following structure:
  - Inputs:
    - `X`: A data frame of covariates/design matrix.
    - `y`: Response vector. [Note that this argument is required even if it is not used by the feature importance method.]
    - `fit`: A fitted estimator (e.g., a fitted RandomForestRegressor).
    - Additional input arguments are allowed, but `X`, `y`, and `fit` are required at a minimum.
  - Output: 
    - A data frame with at least the columns `var` and `importance`, containing the variable ID and the importance scores respectively. Additional columns are also permitted.
  - For examples of this feature importance wrapper, see `scripts/competing_methods.py`.
2. Update configuration files (in `fi_config/`) to set the data-generating process(es), prediction models, and feature importance estimators to run in the simulation. See below for additional information and examples of these configuration files.
3. Run `01_run_importance_simulations.py` and pass in the appropriate commandline arguments. See below for additional information and examples.
  - If `--create_rmd` is passed in as an argument in step 3, this will automatically generate an html document with some basic visualization summaries of the results. These results are rendered using R Markdown via `rmd/simulation_results.Rmd` and are saved in the results folder that was specified in step 3.

Notes:
  - To apply the feature importance method to real data (or any setting where the true support/signal features are unknown), one can use `02_run_importance_real_data.py` instead of `01_run_importance_simulations.py`.
  - To evaluate the prediction accuracy of the model fits, see `03_run_prediction_simulations.py` and `04_run_prediction_real_data.py` for simulated and real data, respectively.

Additional details for steps 2 and 3 are provided below.


## Creating the config files (step 2)

For a starter template, see the `fi_config/test` folder. There are two necessary files:

- `dgp.py`: Script specifying the data-generating process under study. The following variables must be provided:
  - `X_DGP`: Function to generate X data.
  - `X_PARAMS_DICT`: Dictionary of named arguments to pass into the `X_DGP` function.
  - `Y_DGP`: Function to generate y data.
  - `Y_PARAMS_DICT`: Dictionary of named arguments to pass into the `Y_DGP` function.
  - `VARY_PARAM_NAME`: Name of argument (typically in `X_DGP` or `Y_DGP`) to vary across. Note that it is also possible to vary across an argument in an `ESTIMATOR` in very basic simulation setups. This can also be a vector of parameters to vary over in a grid.
  - `VARY_PARAM_VALS`: Dictionary of named arguments for the `VARY_PARAM_NAME` to take on in the simulation experiment. Note that the value can be any python object, but make sure to keep the key simple for naming and plotting purposes. 
- `models.py`: Script specifying the prediction methods and feature importance estimators under study. The following variables must be provided:
  - `ESTIMATORS`: List of prediction methods to fit. Elements should be of class `ModelConfig`.
    - Note that the class passed into `ModelConfig` should have a `fit` method (e.g., like sklearn models).
    - Additional arguments to pass to the model class can be specified in a dictionary using the `other_params` arguments in `ModelConfig().
  - `FI_ESTIMATORS`: List of feature importance methods to fit. Elements should be of class `FIModelConfig`.
    - Note that the function passed into `FIModelConfig` should take in the arguments `X`, `y`, `fit` at a minimum. For examples, see `scripts/competing_methods.py`.
    - Additional arguments to pass to the feature importance function can be specified in a dictionary using the `other_params` argument in `FIModelConfig()`.
    - Pair up a prediction model and feature importance estimator by using the same `model_type` ID in both.
    - Note that by default, higher values from the feature importance method are assumed to indicate higher importance. If higher values indicate lower importance, set `ascending=False` in `FIModelConfig()`.
    - If a feature importance estimator requires sample splitting (outside of the feature importance function call), use the `splitting_strategy` argument to specify the type of splitting strategy (e.g., `train-test`).

For an example of the fi_config files used for real data case studies, see the `fi_config/gmdi/real_data_case_study/ccle_rnaseq_regression-/` folder. Like before, there are two necessary files: `dgp.py` and `models.py`. `models.py` follows the same structure and requirements as above. `dgp.py` is a script that defines two variables, `X_PATH` and `Y_PATH`, specifying the file paths of the covariate/feature data `X` and the response data `y`, respectively.


## Running the simulations (step 3)

**For running simulations to evaluate feature importance rankings:** `01_run_importance_simulations.py`

- Command Line Arguments:
  - Simulation settings:
    - `--nreps`: Number of replicates.
    - `--model`: Name of prediction model to run. Default (`None`) uses all models specified in `models.py` config file.
    - `--fi_model`: Name of feature importance estimator to run. Default (`None`) uses all feature importance estimators specified in `models.py` config file.
    - `--config`: Name of fi_config folder (and title of the simulation experiment). 
    - `--omit_vars`: (Optional) Comma-separated string of feature indices to omit (as unobserved variables). More specifically, these features may be used in generating the response *y* but are omitted from the *X* used in training/evaluating the prediction model and feature importance estimator.
  - Computational settings:
    - `--nosave_cols`: (Optional) Comma-separated string of column names to omit in the output file (to avoid potential errors when saving to pickle).
    - `--ignore_cache`: Whether or not to ignore cached results.
    - `--verbose`: Whether or not to print messages.
    - `--parallel`: Whether or not to run replicates in parallel.
    - `--parallel_id`: ID for parallelization.
    - `--n_cores`: Number of cores if running in parallel. Default uses all available cores.
    - `--split_seed`: Seed for data splitting.
    - `--results_path`: Path to save results. Default is `./results`.
  - R Markdown output options:
    - `--create_rmd`: Whether or not to output R Markdown-generated html file with summary of results.
    - `--show_vars`: Max number of features to show in rejection probability plots in the R Markdown. Default (`None`) is to show all variables.
- Example usage in command line: 
```
python 01_run_importance_simulations.py --nreps 100 --config test --split_seed 331 --ignore_cache --create_rmd
```

**For running feature importance methods on real data:** `02_run_importance_real_data.py`

- Command Line Arguments:
  - Simulation settings:
    - `--nreps`: Number of replicates (or times to run method on the given data).
    - `--model`: Name of prediction model to run. Default (`None`) uses all models specified in `models.py` config file.
    - `--fi_model`: Name of feature importance estimator to run. Default (`None`) uses all feature importance estimators specified in `models.py` config file.
    - `--config`: Name of fi_config folder (and title of the simulation experiment). 
    - `--response_idx`: (Optional) Name of response column to use if response data *y* is a matrix or multi-task. If not provided, independent regression/classification problems are fitted for every column of *y* separately. If *y* is not a matrix, this argument should be ignored and is unused.
  - Computational settings:
    - `--nosave_cols`: (Optional) Comma-separated string of column names to omit in the output file (to avoid potential errors when saving to pickle).
    - `--ignore_cache`: Whether or not to ignore cached results.
    - `--verbose`: Whether or not to print messages.
    - `--parallel`: Whether or not to run replicates in parallel.
    - `--parallel_id`: ID for parallelization.
    - `--n_cores`: Number of cores if running in parallel. Default uses all available cores.
    - `--split_seed`: Seed for data splitting.
    - `--results_path`: Path to save results. Default is `./results`.
- Example usage in command line: 
```
python 02_run_simulations.py --nreps 1 --config test --split_seed 331 --ignore_cache
```

**For running simulations to evaluate prediction accuracy of the model fits:** `03_run_prediction_simulations.py`

- Command Line Arguments:
  - Simulation settings:
    - `--nreps`: Number of replicates.
    - `--mode`: One of 'regression', 'binary_classification', or 'binary_classification'.
    - `--model`: Name of prediction model to run. Default (`None`) uses all models specified in `models.py` config file.
    - `--config`: Name of fi_config folder (and title of the simulation experiment). 
    - `--omit_vars`: (Optional) Comma-separated string of feature indices to omit (as unobserved variables). More specifically, these features may be used in generating the response *y* but are omitted from the *X* used in training/evaluating the prediction model and feature importance estimator.
  - Computational settings:
    - `--nosave_cols`: (Optional) Comma-separated string of column names to omit in the output file (to avoid potential errors when saving to pickle).
    - `--ignore_cache`: Whether or not to ignore cached results.
    - `splitting_strategy`: One of 'train-test', 'train-tune-test', 'train-test-lowdata', 'train-tune-test-lowdata', indicating how to split the data into training and test.
    - `--verbose`: Whether or not to print messages.
    - `--parallel`: Whether or not to run replicates in parallel.
    - `--parallel_id`: ID for parallelization.
    - `--n_cores`: Number of cores if running in parallel. Default uses all available cores.
    - `--split_seed`: Seed for data splitting.
    - `--results_path`: Path to save results. Default is `./results`.
- Example usage in command line: 
```
python 03_run_prediction_simulations.py --nreps 100 --config gmdi.glm_metric_choices_sims.regression_prediction_sims.enhancer_linear_dgp --mode regression --split_seed 331 --ignore_cache --nosave_cols prediction_model
```

**For running prediction methods on real data:** `04_run_prediction_real_data.py`

- Command Line Arguments:
  - Simulation settings:
    - `--nreps`: Number of replicates.
    - `--mode`: One of 'regression', 'binary_classification', or 'binary_classification'.
    - `--model`: Name of prediction model to run. Default (`None`) uses all models specified in `models.py` config file.
    - `--config`: Name of fi_config folder (and title of the simulation experiment). 
    - `--response_idx`: (Optional) Name of response column to use if response data *y* is a matrix or multi-task. If not provided, independent regression/classification problems are fitted for every column of *y* separately. If *y* is not a matrix, this argument should be ignored and is unused.
    - `--subsample_n`: (Optional) Integer indicating max number of samples to use in training prediction model. If None, no subsampling occurs.
  - Computational settings:
    - `--nosave_cols`: (Optional) Comma-separated string of column names to omit in the output file (to avoid potential errors when saving to pickle).
    - `--ignore_cache`: Whether or not to ignore cached results.
    - `splitting_strategy`: One of 'train-test', 'train-tune-test', 'train-test-lowdata', 'train-tune-test-lowdata', indicating how to split the data into training and test.
    - `--verbose`: Whether or not to print messages.
    - `--parallel`: Whether or not to run replicates in parallel.
    - `--parallel_id`: ID for parallelization.
    - `--n_cores`: Number of cores if running in parallel. Default uses all available cores.
    - `--split_seed`: Seed for data splitting.
    - `--results_path`: Path to save results. Default is `./results`.
- Example usage in command line: 
```
python 04_run_prediction_real_data.py --nreps 10 --config gmdi.prediction_sims.enhancer_classification- --mode binary_classification --split_seed 331 --ignore_cache --nosave_cols prediction_model
```