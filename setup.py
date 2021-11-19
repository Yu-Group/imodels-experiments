import setuptools
from os import path

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required_pypi = [
    'numpy',
    'pandas',
    'scipy',
]

setuptools.setup(
    name="experiments",
    version="0.0.1",
    author="Chandan Singh, Keyan Nasseri, and others",
    author_email="chandan_singh@berkeley.edu",
    description="Experiments with various interpretable models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/imodels",
    packages=setuptools.find_packages(
        exclude=['tests', 'tests.*', '*.test.*']
    ),
    install_requires=required_pypi,
    extras_require={
        'dev': [
            'dvu',
            'gdown',
            'jupyter',
            'jupytext',
            'matplotlib',
            'pytest',
            'pytest-cov',
            'slurmpy',
            'tqdm',
            'pmlb',
        ]
    },
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)