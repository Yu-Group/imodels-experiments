# Subgroup Experiments

To reproduce the results in Section 8 and Appendix H:
- Run `investigation-runner.sh` in the `get-values` subdirectory. This script reads the miami housing data in `data_openml` and saves the local feature importance scores in a new `lfi-values` folder.
- In the `compile-results` subdirectory, run `compile-runner.sh`, which takes the LFI scores in `lfi-values` and clusters on them, saving cluster labels and errors to a new `cluster-results` subdirectory.
- Still in `compile-results`, run the `case-study.ipynb` notebook, producing the figures displayed in both Section 8 and Appendix H.