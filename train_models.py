# %% [markdown]
# # Steel plate fault classification: training the five models
#
# ML Assignment 2, Anirudh Gosain (2026AK05290)
#
# This script goes from the raw UCI files to the five saved models:
#
# 1. read the raw data and turn the 7 one-hot fault columns into one label
# 2. split off a stratified 20 % test set and write it to `test_data.csv`
# 3. tune each model with 5-fold cross-validation on the training part only
# 4. score the tuned models once on the test set (Accuracy, AUC, Precision,
#    Recall, F1, MCC)
# 5. save every fitted pipeline to `model/` together with `model_info.json`
#
# Run it with `python train_models.py`. The `# %%` markers also let it be
# opened as a notebook (VS Code / Jupytext) and run cell by cell.

# %%
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from fault_metrics import METRIC_NAMES, TARGET, confusion_table, evaluate

ROOT = Path(__file__).resolve().parent if "__file__" in globals() \
    else Path.cwd()
RAW = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"

SEED = 5290          # last four digits of my BITS ID, fixes every random step
TEST_SHARE = 0.20
CV_FOLDS = 5

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

# %% [markdown]
# ## 1. Loading the raw data
#
# `Faults.NNA` has no header; the 34 column names are in `Faults27x7_var`.
# The first 27 columns are measurements of the defect (geometry, luminosity,
# steel type, plate thickness and derived indices), the last 7 are one-hot
# flags for the fault type. Every row has exactly one flag set, so the
# flags are collapsed into a single 7-class label.

# %%
column_names = [line.strip() for line in
                (RAW / "Faults27x7_var").read_text().splitlines()
                if line.strip()]
raw = pd.read_csv(RAW / "Faults.NNA", sep=r"\s+", header=None,
                  names=column_names)

measurement_cols = column_names[:27]
fault_cols = column_names[27:]

assert (raw[fault_cols].sum(axis=1) == 1).all(), "every row has one fault"

plates = raw[measurement_cols].copy()
plates[TARGET] = raw[fault_cols].idxmax(axis=1)

print("shape of the raw table:", raw.shape)
print("missing values        :", int(raw.isna().sum().sum()))
print("duplicate rows        :", int(raw.duplicated().sum()))
print()
print(plates[TARGET].value_counts().to_frame("plates"))

# %% [markdown]
# ## 2. Features
#
# `TypeOfSteel_A300` and `TypeOfSteel_A400` are a one-hot pair: in every row
# exactly one of them is 1. The second carries no information the first
# does not, and an exactly collinear column is unhelpful for logistic
# regression and Gaussian Naive Bayes, so it is dropped. That leaves 26
# input features, above the minimum of 12 in the brief.

# %%
assert (plates["TypeOfSteel_A300"] + plates["TypeOfSteel_A400"] == 1).all()
features = [c for c in measurement_cols if c != "TypeOfSteel_A400"]
print(len(features), "features used:")
print(", ".join(features))

# %% [markdown]
# Several features are very skewed: `Pixels_Areas`, `X_Perimeter`,
# `Y_Perimeter` and `Sum_of_Luminosity` have a few plates hundreds of times
# larger than the median. That matters for the distance- and
# gradient-based models, so for them the grid search is allowed to choose
# between a plain `StandardScaler` and a Yeo-Johnson `PowerTransformer`,
# which also standardises but first pulls the long tails in.

# %%
skew = plates[features].skew().sort_values(ascending=False)
print("most skewed features:")
print(skew.head(6).round(2).to_string())

# %% [markdown]
# ## 3. Train / test split
#
# A stratified 80 / 20 split keeps the class proportions the same in both
# parts, which matters for the two small classes (Dirtiness has only 55
# plates in total). The test part is written to `test_data.csv`: these are
# the plates the app is demonstrated on, and the models never see them
# during tuning.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    plates[features], plates[TARGET], test_size=TEST_SHARE,
    stratify=plates[TARGET], random_state=SEED)

test_frame = X_test.copy()
test_frame[TARGET] = y_test
test_frame.to_csv(ROOT / "test_data.csv", index=False)

print(f"training plates: {len(X_train)}   test plates: {len(X_test)}")
print(pd.DataFrame({"train": y_train.value_counts(),
                    "test": y_test.value_counts()}))

# %% [markdown]
# ## 4. The five models and their search grids
#
# Each model is a `Pipeline`, so the scaling learnt on the training data is
# saved with the model and applied automatically to whatever the app feeds
# it. Hyper-parameters are tuned with `GridSearchCV` on the training part
# only, using stratified 5-fold cross-validation.
#
# The tuning score is the **Matthews correlation coefficient**. Plain
# accuracy would reward a model that ignores the rare faults, while MCC
# uses the whole confusion matrix and stays low unless every class is
# handled reasonably.
#
# `class_weight="balanced"` is offered to the models that support it; the
# search decides whether up-weighting the rare classes helps.

# %%
scalers = [StandardScaler(), PowerTransformer(method="yeo-johnson")]

# scikit-learn renamed CalibratedClassifierCV's "base_estimator" to
# "estimator" in version 1.2. Asking the class which one it has lets the
# same grid run on the BITS Virtual Lab's Anaconda and on newer installs.
SVM_INNER = ("estimator" if "estimator" in CalibratedClassifierCV().get_params()
             else "base_estimator")

candidates = {
    "Logistic Regression": (
        Pipeline([("scale", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))]),
        {"scale": scalers,
         "clf__C": [0.01, 0.1, 1, 10, 100],
         "clf__class_weight": [None, "balanced"]}),

    "Decision Tree": (
        Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
        {"clf__criterion": ["gini", "entropy"],
         "clf__max_depth": [4, 6, 8, 10, 14, None],
         "clf__min_samples_leaf": [1, 3, 5, 10, 20, 40],
         "clf__class_weight": [None, "balanced"]}),

    "KNN": (
        Pipeline([("scale", StandardScaler()),
                  ("clf", KNeighborsClassifier())]),
        {"scale": scalers,
         "clf__n_neighbors": [3, 5, 7, 9, 13, 17, 25],
         "clf__weights": ["uniform", "distance"],
         "clf__p": [1, 2]}),

    "Naive Bayes": (
        Pipeline([("scale", StandardScaler()),
                  ("clf", GaussianNB())]),
        {"scale": scalers,
         "clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 1.0]}),

    # An SVM gives margins, not probabilities, and AUC needs probabilities.
    # They are added by Platt scaling (a sigmoid fitted on cross-validated
    # margins) through CalibratedClassifierCV. ensemble=False keeps a single
    # SVM trained on all the data. This replaces SVC(probability=True),
    # which scikit-learn 1.9 deprecates.
    "Support Vector Machine": (
        Pipeline([("scale", StandardScaler()),
                  ("clf", CalibratedClassifierCV(
                      SVC(kernel="rbf", random_state=SEED),
                      method="sigmoid", cv=5, ensemble=False))]),
        {"scale": scalers,
         f"clf__{SVM_INNER}__C": [0.3, 1, 3, 10, 30],
         f"clf__{SVM_INNER}__gamma": ["scale", 0.01, 0.03, 0.1],
         f"clf__{SVM_INNER}__class_weight": [None, "balanced"]}),
}

file_names = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

# %% [markdown]
# ## 5. Tuning
#
# The same `StratifiedKFold` object (fixed seed) is used for every model,
# so all five are compared on exactly the same folds.

# %%
mcc_scorer = make_scorer(matthews_corrcoef)
folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

tuned = {}
search_log = {}

for name, (pipeline, grid) in candidates.items():
    started = time.time()
    search = GridSearchCV(pipeline, grid, scoring=mcc_scorer, cv=folds,
                          n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    tuned[name] = search.best_estimator_

    # the chosen settings, written so that they read well in the README
    chosen = {key.replace("clf__", "").replace(f"{SVM_INNER}__", ""): (type(value).__name__
                                         if key == "scale" else value)
              for key, value in search.best_params_.items()}
    search_log[name] = {
        "best_params": chosen,
        "cv_mcc_mean": float(search.best_score_),
        "cv_mcc_std": float(search.cv_results_["std_test_score"]
                            [search.best_index_]),
        "settings_tried": len(search.cv_results_["params"]),
        "seconds": round(time.time() - started, 1),
    }
    print(f"{name:<24} CV MCC = {search.best_score_:.4f} "
          f"(+/- {search_log[name]['cv_mcc_std']:.4f})   "
          f"{len(search.cv_results_['params'])} settings, "
          f"{search_log[name]['seconds']} s")
    print(f"{'':<24} best: {chosen}")

# %% [markdown]
# ## 6. Scoring on the held-out test set
#
# Each tuned pipeline (refitted on the full training part) is scored once on
# the test plates. AUC is the macro one-vs-rest AUC from the predicted
# probabilities; Precision, Recall and F1 are macro averages over the seven
# fault types (see `fault_metrics.py`).

# %%
rows = []
for name, model in tuned.items():
    scores = evaluate(model, X_test, y_test)
    rows.append({"ML Model Name": name, **scores})

comparison = pd.DataFrame(rows).set_index("ML Model Name")
print(comparison.round(4).to_string())

# %% [markdown]
# Confusion matrices on the test set (rows = true fault, columns =
# predicted fault). These show *which* faults each model mixes up, which
# the single-number metrics hide.

# %%
confusions = {}
for name, model in tuned.items():
    table = confusion_table(model, X_test, y_test)
    confusions[name] = table
    print(f"\n{name}")
    print(table.to_string())

# %% [markdown]
# Recall per fault type, i.e. the share of each kind of fault that each
# model catches.

# %%
per_class_recall = pd.DataFrame(
    {name: np.diag(table.values) / table.values.sum(axis=1)
     for name, table in confusions.items()},
    index=confusions["KNN"].index)
print(per_class_recall.round(3).to_string())

# %% [markdown]
# ## 7. Saving everything
#
# * `model/<model>.joblib`: the five fitted pipelines (scaler + classifier)
# * `model/model_info.json`: feature order, class labels, the chosen
#   hyper-parameters, CV scores, test scores and the library versions used
# * `results/`: the comparison table, per-class recall and confusion
#   matrices as CSV files, for the README

# %%
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

for name, model in tuned.items():
    joblib.dump(model, MODEL_DIR / file_names[name], compress=3)

info = {
    "dataset": "Steel Plates Faults (UCI id 198)",
    "target": TARGET,
    "features": features,
    "classes": sorted(plates[TARGET].unique().tolist()),
    "seed": SEED,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "metric_averaging": "macro (Precision, Recall, F1); "
                        "macro one-vs-rest (AUC)",
    "models": {name: {"file": file_names[name], **search_log[name],
                      "test_scores": {m: float(comparison.loc[name, m])
                                      for m in METRIC_NAMES}}
               for name in tuned},
    "versions": {"python": __import__("platform").python_version(),
                 "scikit-learn": sklearn.__version__,
                 "pandas": pd.__version__, "numpy": np.__version__},
}
(MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2))

comparison.round(4).to_csv(RESULTS_DIR / "test_metrics.csv")
per_class_recall.round(4).to_csv(RESULTS_DIR / "per_class_recall.csv")
for name, table in confusions.items():
    table.to_csv(RESULTS_DIR / f"confusion_{file_names[name][:-7]}.csv")

print("saved:", sorted(p.name for p in MODEL_DIR.iterdir()))

# %% [markdown]
# ## 8. Check: the saved files give the same answers
#
# The app will only ever load these files, so reload each one from disk and
# confirm it reproduces the test scores above exactly. Nothing is refitted.

# %%
for name in tuned:
    reloaded = joblib.load(MODEL_DIR / file_names[name])
    again = evaluate(reloaded, X_test, y_test)
    same = all(np.isclose(again[m], comparison.loc[name, m])
               for m in METRIC_NAMES)
    print(f"{name:<24} reloaded model matches: {same}")

# %% [markdown]
# ## 9. Summary
#
# The final results in one place, so the end of the run shows the whole
# outcome: the environment it ran in and the test scores of all five models.

# %%
import platform

print()
print("=" * 78)
print("SUMMARY: Steel Plates Faults, 5 classifiers, held-out test set "
      f"({len(X_test)} plates)")
print(f"python {platform.python_version()} | scikit-learn {sklearn.__version__}"
      f" | host {platform.node()}")
print("=" * 78)
print(comparison.round(4).to_string())
print("-" * 78)
best_model = comparison["MCC"].idxmax()
print(f"Highest test MCC: {best_model} ({comparison.loc[best_model, 'MCC']:.4f})"
      f" | models saved to {MODEL_DIR.name}/")
