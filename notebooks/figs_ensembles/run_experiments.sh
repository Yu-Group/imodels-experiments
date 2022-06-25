python 01_fit_models.py --config figs_ensembles --classification_or_regression classification --model xgboost --dataset heart --split_seed 0

python 01_fit_models.py --config figs_ensembles --classification_or_regression classification --model randomforest --dataset heart --split_seed 0

python 01_fit_models.py --config figs_ensembles --classification_or_regression classification --model figs --dataset heart --split_seed 0

python3 01_fit_models.py --config figs_ensembles --classification_or_regression classification --model baggingfigs --dataset heart --split_seed 0

python 01_fit_models.py --config figs_ensembles --classification_or_regression classification --dataset heart --split_seed 0
python 01_fit_models.py --config figs_ensembles --classification_or_regression classification --dataset sonar --split_seed 0
python 01_fit_models.py --config figs_ensembles --classification_or_regression regression --dataset diabetes-regr --split_seed 0

python3 01_fit_models.py --config figs_ensembles --classification_or_regression classification --model rffigs --dataset heart --split_seed 0