# PyConDE & PyData Berlin 2022 - scikit-learn tutorial

Some intro [slides](https://docs.google.com/presentation/d/1xPf8vN9-pwZkAq28gbghJ9IWaNO4w-ZKO59193BWs34/edit?usp=sharing)

## Follow the tutorial online

- Launch an online notebook environment using: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/glemaitre/pydata_berlin_2022_scikit_learn_tutorial/main)

You need an internet connection but you will not have to install any package
locally.

## Running the tutorial locally

### Dependencies

The tutorials will require the following packages:

* python
* jupyter
* scikit-learn
* pandas
* matplotlib
* seaborn
* shap

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
