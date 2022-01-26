import subprocess

import numpy as np


def main():
    for seed in range(9):
        for reg in np.arange(0, 0.0051, 0.001):
            cmd = f"sbatch -p jsteinhardt -C manycore viz_godst.sh {seed} {reg}"
            subprocess.call(cmd, shell=True)