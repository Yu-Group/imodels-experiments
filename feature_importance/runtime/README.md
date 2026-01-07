# Runtime Experiments

To reproduce the results in Appendix J:
- Run the `get_data.ipynb` notebook to download the data from OpenML into a `data` subdirectory.
- Run `regression-runner.sh`. This script runs a variety of datasets in the `data` subdirectory and saves their runtimes in a new `results` folder.
- Run the `runtime.ipynb` notebook, producing the tables displayed in Appendix J.