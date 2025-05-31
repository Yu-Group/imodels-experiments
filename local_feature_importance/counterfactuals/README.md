# Counterfactual Experiments

## Appendix G

To reproduce the results in Appendix G of the LMDI+ paper, run the `simulation.ipynb` file.

## Section 7

To reproduce the results in Section 7 of the LMDI+ paper, follow these steps:
- Run `get_data.ipynb`, which will download the six classification datasets (downsampled to 2000 rows) to a `data` folder.
- Run `knn-runner.sh`. This file runs through the `knn.py` script for each dataset, finding the closest counterfactual to each test sample and its corresponding distance, writing the results to a `results` folder.
- Run `knn-results.ipynb`. This notebook compiles the information in the `results` folder into what we present in the paper.
