"""
Evaluation helpers shared by the training script and the Streamlit app, so
the numbers in the README and the numbers the app shows are computed by the
same code.

The problem has 7 classes, so the per-class scores are combined with a
macro average: every fault type counts the same, however rare it is. With
classes as uneven as these (673 Other_Faults against 55 Dirtiness), a
weighted average would mostly report how well the big classes are handled.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

TARGET = "Fault_Type"

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def auc_one_vs_rest(y_true, proba, classes):
    """Macro one-vs-rest ROC AUC.

    Each class in turn is treated as "positive" against all the others, and
    the 7 AUCs are averaged. A class missing from y_true (possible in a
    small uploaded file) has no AUC, so it is left out of the average
    rather than crashing the calculation.
    """
    y_true = np.asarray(y_true)
    scores = []
    for k, label in enumerate(classes):
        positive = y_true == label
        if positive.all() or not positive.any():
            continue
        scores.append(roc_auc_score(positive, proba[:, k]))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate(model, X, y):
    """All six required metrics for one fitted model on (X, y)."""
    predicted = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, predicted),
        "AUC": auc_one_vs_rest(y, proba, model.classes_),
        "Precision": precision_score(y, predicted, average="macro",
                                     zero_division=0),
        "Recall": recall_score(y, predicted, average="macro",
                               zero_division=0),
        "F1": f1_score(y, predicted, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, predicted),
    }


def confusion_table(model, X, y):
    """Confusion matrix as a labelled DataFrame (rows = true class)."""
    labels = list(model.classes_)
    matrix = confusion_matrix(y, model.predict(X), labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def report_table(model, X, y):
    """sklearn's classification report as a DataFrame."""
    report = classification_report(y, model.predict(X), output_dict=True,
                                   zero_division=0)
    # sklearn stores accuracy as a single number, not a row of four, so it
    # is rebuilt by hand: the value goes under f1-score (where sklearn's own
    # text report prints it) and the support is the number of plates
    accuracy = report.pop("accuracy")
    table = pd.DataFrame(report).T
    table.loc["accuracy"] = [np.nan, np.nan, accuracy, len(y)]
    table = table.loc[[*table.index[:-3], "accuracy", "macro avg",
                       "weighted avg"]]
    table["support"] = table["support"].astype(int)
    return table
