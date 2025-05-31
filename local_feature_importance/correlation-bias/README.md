# Correlation Bias Results - Section 4.2

To reproduce the results in Section 4.2 of the LMDI+ paper, do the following:
- Run `correlation-runner.sh`, which simulates data and obtains feature rankings for a range of different correlation strengths. Results are written to a `results` folder.
- Run `correlation.ipynb`. This notebook compiles the information in the `results` folder into what we present in the paper.