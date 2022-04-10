# EuroSciPy 2019 - scikit-learn tutorial

Some intro [slides](https://docs.google.com/presentation/d/1xPf8vN9-pwZkAq28gbghJ9IWaNO4w-ZKO59193BWs34/edit?usp=sharing)

## Running the tutorial locally

### Dependencies

The tutorials will require the following packages:

* python
* jupyter
* scikit-learn
* pandas
* matplotlib
* seaborn

### Local install

We provide both `requirements.txt` and `environment.yml` to install packages.

You can install the packages using `pip`:

```
$ pip install -r requirements.txt
```

You can create an `sklearn-tutorial` conda environment executing:

```
$ conda env create -f environment.yml
```

and later activate the environment:

```
$ conda activate sklearn-tutorial
```

You might also only update your current environment using:

```
$ conda env update --prefix ./env --file environment.yml  --prune
```
