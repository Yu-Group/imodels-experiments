import subprocess

import numpy as np


def main():
    for seed in range(9):
        for reg in np.arange(0, 0.0051, 0.001):
            cmd = f"sbatch -p jsteinhardt -C xlmem viz_godst.sh {seed} {reg}"
            subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
