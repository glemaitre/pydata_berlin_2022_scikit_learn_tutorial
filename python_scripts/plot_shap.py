# %% [markdown]
#
# # Exploring the SHAP library
#
# In this example, we use the "Current Population Survey" dataset, already
# used in the interpretation of linear models.

# %%
import sklearn

sklearn.set_config(display="diagram")

# %%
from sklearn.datasets import fetch_openml

survey = fetch_openml(data_id=534, as_frame=True)
survey.frame.head()

# %% [markdown]
#
# The aim is to predict the wage of a person based on set of features such as
# age, experience, education, etc.
#
# We will define a predictive model that uses a gradient-boosting as predictor.
# Beforehand, the categorical data will be encoded using an
# `OrdinalEncoder`. These categorical columns are
# defined by the "category" data type reported by pandas.

# %%
survey.frame.dtypes

# %% [markdown]
#
# We reproduce the same experiment setting than with the linear models example:
# we will use a single train-test split.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    survey.data, survey.target, random_state=0
)

# %% [markdown]
#
# Let's first define the preprocessing pipeline for the encoding of categorical
# features.

# %%
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector, make_column_transformer

categorical_columns = make_column_selector(dtype_include="category")
numerical_columns = make_column_selector(dtype_exclude="category")
preprocessor = make_column_transformer(
    (
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ),
        categorical_columns,
    ),
    remainder="passthrough",
    verbose_feature_names_out=False,
)

# %% [markdown]
#
# Then, we define entire predictive model composed of the preprocessing and the
# gradient-boosting regressor.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

model = make_pipeline(
    preprocessor,
    HistGradientBoostingRegressor(max_iter=10_000, early_stopping=True, random_state=0),
)

# %% [markdown]
#
# Before to start, we will check the statistical performance of the model.
# We can compare it with the linear models seen in the previous example.

# %%
model.fit(X_train, y_train)

# %%
from sklearn.metrics import mean_absolute_error

print(
    f"MAE on the training set: "
    f"{mean_absolute_error(y_train, model.predict(X_train)):.3f} $/hour"
)
print(
    f"MAE on the training set: "
    f"{mean_absolute_error(y_test, model.predict(X_test)):.3f} $/hour"
)

# %% [markdown]
#
# Now, we use the SHAP library that allows to compute an approximation of the
# Shapley values. However, before using SHAP, we need to preprocess the data
# separately due to limited support of some scikit-learn components.

# %%
import pandas as pd

feature_names = categorical_columns(X_train) + numerical_columns(X_train)
X_train_preprocessed = pd.DataFrame(
    preprocessor.fit_transform(X_train), columns=feature_names
)
X_test_preprocessed = pd.DataFrame(
    preprocessor.transform(X_test), columns=feature_names
)

# %% [markdown]
#
# Now, we use SHAP to get an approximation of the Shapley values for each
# testing sample.

# %%
import shap

explainer = shap.Explainer(model[-1], masker=X_train_preprocessed, feature_perturbation="interventional")
shap_values = explainer(X_test_preprocessed)
shap_values.shape == X_test_preprocessed.shape

# %% [markdown]
#
# By inspecting `shap_values`, we observe that we get a feature attributions
# for each data point of the testing set. We can as well see a repeated
# information called `base_values`:

# %%
shap_values

# %% [markdown]
#
# Indeed, this base value represents the mean prediction and SHAP is
# attributing feature values to explain the difference of each prediction
# of the testing set in regards with the base value.
#
# Let's show the SHAP values decomposition for the first sample of the test
# set. Our model would produce the following value:

# %%
model.predict(X_test.iloc[[0]])

# %% [markdown]
#
# The reported SHAP values for the different features are:

# %%
pd.Series(shap_values[0].values, index=feature_names)

# %% [markdown]
#
# Taking into account the base value, then the model prediction corresponds to
# the following sum:

# %%
shap_values[0].values.sum() + shap_values.base_values[0]

# %% [markdown]
#
# SHAP package comes with handy plotting facilities to visualize the Shapley
# values. Let's start by the `waterfall` plot.

# %%
shap.plots.waterfall(shap_values[0])

# %% [markdown]
#
# It represents the graphical summation of the Shapley values for each
# feature to observe the difference between the expected value and the actual
# prediction. Another inline representation is the `force` plot.

# %%
shap.initjs()
shap.plots.force(
    shap_values.base_values[0],
    shap_values.values[0],
    feature_names=feature_names,
)

# %% [markdown]
#
# We can plot the Shapley values for all samples and encode the color of the
# features values.

# %%
shap.plots.beeswarm(shap_values)

# %% [markdown]
#
# By combining the SHAP values for all samples of the testing set, we can then
# get a global explanation.

# %%
shap.plots.bar(shap_values)

# %% [markdown]
#
# Now, we can make a quick comparison between the Shapley values and the
# permutation importances. For both, we will make a study of uncertainty by
# using multiple permutations.

# %%
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

importances = permutation_importance(model, X_test, y_test, n_jobs=-1)
sorted_idx = importances.importances_mean.argsort()

importances = pd.DataFrame(
    importances.importances[sorted_idx].T, columns=X_test.columns[sorted_idx]
)
importances.plot.box(vert=False, whis=100)
plt.axvline(0, color="k", linestyle="--")
plt.xlabel("Decrease in R2 score")
_ = plt.title("Permutation importances")

# %% [markdown]
#
# We can make use of bootstrap resampling of the test set in order to repeat
# the experiment with a variation of the test dataset.

# %%
import numpy as np

rng = np.random.default_rng(42)
n_bootstrap = 25

all_shap_values = []
for _ in range(n_bootstrap):
    bootstrap_idx = rng.choice(
        np.arange(X_test.shape[0]), size=X_test.shape[0], replace=True
    )
    X_test_bootstrap = X_test.iloc[bootstrap_idx]
    X_test_preprocessed = pd.DataFrame(
        preprocessor.transform(X_test_bootstrap), columns=feature_names
    )
    all_shap_values.append(explainer(X_test_preprocessed))

# %%
shap_values = pd.DataFrame(
    [np.abs(shap_values.values).mean(axis=0) for shap_values in all_shap_values],
    columns=feature_names,
)
sorted_idx = shap_values.mean().sort_values().index

# %%
shap_values[sorted_idx].plot.box(vert=False, whis=10)
plt.xlabel("mean(|SHAP values|)")
_ = plt.title("SHAP values")

# %% [markdown]
#
# Comparing the permutation importance and the SHAP values, we observe a
# difference in the ranking of the features.
#
# ## Bonus point regarding some SHAP internal:

# %%
explainer

# %%
explainer.feature_perturbation

# %%
explainer = shap.Explainer(model[-1])
explainer(X_test_preprocessed)

# %%
explainer.feature_perturbation

# %%
explainer = shap.Explainer(model[-1], feature_perturbation="interventional")
explainer(X_test_preprocessed)

# %%
explainer.feature_perturbation

# %%
X = np.concatenate([
    [[0, 0]] * 400,
    [[0, 1]] * 100,
    [[1, 0]] * 100,
    [[1, 1]] * 400
], axis=0)
X

# %%
y = np.array(
    [0] * 400 + [50] * 100 + [50] * 100 + [100] * 400
)

# %%
from sklearn.tree import DecisionTreeRegressor

tree_1 = DecisionTreeRegressor(random_state=0).fit(X, y)

# %%
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 6))
_ = plot_tree(tree_1)

# %%
tree_2 = DecisionTreeRegressor(random_state=4).fit(X, y)

# %%
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 6))
_ = plot_tree(tree_2)

# %%
X_test = np.array([[1, 1]])
explainer = shap.explainers.Exact(tree_1.predict, X)
explainer(X_test)

# %%
explainer = shap.explainers.Exact(
    tree_1.predict, masker=shap.maskers.Independent(X, max_samples=X.shape[0])
)
explainer(X_test)

# %%
explainer = shap.explainers.Exact(
    tree_2.predict, masker=shap.maskers.Independent(X, max_samples=X.shape[0])
)
explainer(X_test)

# %%
explainer = shap.Explainer(tree_1)
explainer(X_test)

# %%
explainer = shap.Explainer(tree_2)
explainer(X_test)

# %%
explainer

# %%
explainer = shap.Explainer(tree_1, shap.maskers.Independent(X, max_samples=X.shape[0]))
explainer(X_test)

# %%
explainer = shap.Explainer(tree_2, shap.maskers.Independent(X, max_samples=X.shape[0]))
explainer(X_test)

# %%
explainer
