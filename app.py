"""
Steel Plate Fault Inspector - Streamlit front end for ML Assignment 2.

Upload a CSV of steel plates (the 26 measurement columns plus the true
Fault_Type), pick one of the five saved classifiers and see how it does.
The models are only loaded from model/ and used to predict; nothing is
trained or refitted here.

Run locally with:  streamlit run app.py
"""

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from fault_metrics import (METRIC_NAMES, TARGET, confusion_table, evaluate,
                           report_table)

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
SAMPLE_CSV = HERE / "test_data.csv"

# a plain description of each label, so the class names in the results mean
# something to a reader who has never seen the dataset
FAULT_NOTES = {
    "Pastry": "patch-like surface defect",
    "Z_Scratch": "scratch defect, Z type",
    "K_Scatch": "scratch defect, K type (spelt 'Scatch' in the dataset)",
    "Stains": "stain on the surface",
    "Dirtiness": "dirt on the surface",
    "Bumps": "raised bump on the surface",
    "Other_Faults": "any fault outside the six named types",
}

st.set_page_config(page_title="Steel Plate Fault Inspector",
                   page_icon="🔩", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 2rem; }
div[data-testid="stMetric"] {
    background: rgba(70, 110, 160, 0.08);
    border: 1px solid rgba(70, 110, 160, 0.25);
    border-radius: 10px; padding: 10px 14px;
}
.hero { border-left: 5px solid #466ea0; padding: 4px 0 4px 14px;
        margin-bottom: 0.6rem; }
.hero h1 { margin: 0; font-size: 2rem; }
.hero p { margin: 2px 0 0 0; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)


# ---------- loading the saved models (cached, never retrained) ----------

@st.cache_resource
def load_models():
    info = json.loads((MODEL_DIR / "model_info.json").read_text())
    models = {name: joblib.load(MODEL_DIR / details["file"])
              for name, details in info["models"].items()}
    return info, models


@st.cache_data
def read_csv(raw_bytes):
    """Parse CSV bytes. Taking bytes (not the upload object) keeps the
    cache keyed on the file content itself."""
    return pd.read_csv(io.BytesIO(raw_bytes))


INFO, MODELS = load_models()
FEATURES = INFO["features"]


def check_columns(frame):
    """Returns (missing feature columns, whether the target is present)."""
    missing = [c for c in FEATURES if c not in frame.columns]
    return missing, TARGET in frame.columns


def draw_confusion(table, title):
    """Heat map of a confusion matrix with the counts written in."""
    counts = table.values
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(counts, cmap="Blues")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=40,
                  ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("Predicted fault")
    ax.set_ylabel("True fault")
    ax.set_title(title, fontsize=11)
    threshold = counts.max() / 2
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, counts[i, j], ha="center", va="center",
                    color="white" if counts[i, j] > threshold else "black",
                    fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def draw_comparison(scores):
    """Grouped bars: one group per metric, one bar per model."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.16
    x = np.arange(len(METRIC_NAMES))
    colours = ["#466ea0", "#d17b0f", "#4c9a52", "#b8475a", "#7a5ca8"]
    for k, (name, row) in enumerate(scores.iterrows()):
        ax.bar(x + (k - 2) * width, row[METRIC_NAMES].values, width,
               label=name, color=colours[k])
    ax.set_xticks(x, METRIC_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=5, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, 1.15), frameon=False)
    fig.tight_layout()
    return fig


# ---------- sidebar: data and model ----------

with st.sidebar:
    st.header("1 · Test data")
    upload = st.file_uploader("Upload a CSV of steel plates", type="csv",
                              help="Needs the 26 feature columns and the "
                                   f"true label column '{TARGET}'.")
    st.download_button("Download the sample test_data.csv",
                       SAMPLE_CSV.read_bytes(), file_name="test_data.csv",
                       mime="text/csv", width="stretch")
    use_sample = st.toggle("Use the bundled test_data.csv instead",
                           value=upload is None)

    st.header("2 · Model")
    chosen = st.selectbox("Classifier", list(MODELS))
    params = INFO["models"][chosen]["best_params"]
    st.caption("Tuned settings: " +
               ", ".join(f"{k} = {v}" for k, v in params.items()))

    st.divider()
    st.caption("ML Assignment 2 · Anirudh Gosain · 2026AK05290  \n"
               "Data: UCI Steel Plates Faults (id 198)")


# ---------- header ----------

st.markdown("""
<div class="hero">
<h1>🔩 Steel Plate Fault Inspector</h1>
<p>Seven kinds of surface defect on steel plates, five classifiers, one
held-out test set. Upload plates, pick a model, inspect the results.</p>
</div>""", unsafe_allow_html=True)

if upload is not None and not use_sample:
    data = read_csv(upload.getvalue())
    source_name = upload.name
else:
    data = read_csv(SAMPLE_CSV.read_bytes())
    source_name = "test_data.csv (bundled)"

missing, has_target = check_columns(data)
if missing:
    st.error(f"**{source_name}** is missing {len(missing)} required "
             "column(s): " + ", ".join(missing))
    st.stop()

X = data[FEATURES]
st.caption(f"Data: **{source_name}**, {len(data)} plates, "
           f"{len(FEATURES)} features"
           + ("" if has_target else f". No '{TARGET}' column, so only "
              "predictions can be shown."))

tab_eval, tab_compare, tab_predict, tab_about = st.tabs(
    ["📊 Evaluate model", "🏁 Compare all models", "🔎 Predictions",
     "ℹ️ About"])


# ---------- tab 1: one model in detail ----------

with tab_eval:
    if not has_target:
        st.info(f"Add the '{TARGET}' column to the CSV to see metrics.")
    else:
        y = data[TARGET]
        unknown = sorted(set(y) - set(MODELS[chosen].classes_))
        if unknown:
            st.warning("Labels the models were not trained on: "
                       + ", ".join(map(str, unknown)))

        model = MODELS[chosen]
        scores = evaluate(model, X, y)
        st.subheader(f"{chosen}: evaluation metrics")
        columns = st.columns(6)
        for column, metric in zip(columns, METRIC_NAMES):
            column.metric(metric, f"{scores[metric]:.4f}")
        st.caption("Precision, Recall and F1 are macro averages over the "
                   "fault types; AUC is macro one-vs-rest.")

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("**Confusion matrix**")
            figure = draw_confusion(confusion_table(model, X, y),
                                    f"{chosen} on {len(data)} plates")
            st.pyplot(figure)
            plt.close(figure)
        with right:
            st.markdown("**Classification report**")
            report = report_table(model, X, y)
            # written out as text so the empty cells on the accuracy row
            # stay blank instead of showing "None"
            for column in ["precision", "recall", "f1-score"]:
                report[column] = report[column].map(
                    lambda v: "" if pd.isna(v) else f"{v:.3f}")
            st.dataframe(report, width="stretch")


# ---------- tab 2: all five models side by side ----------

with tab_compare:
    if not has_target:
        st.info(f"Add the '{TARGET}' column to the CSV to compare models.")
    else:
        y = data[TARGET]
        table = pd.DataFrame({name: evaluate(model, X, y)
                              for name, model in MODELS.items()}).T
        table.index.name = "ML Model Name"
        st.subheader("All five models on this data")
        st.dataframe(table.style.format("{:.4f}")
                     .highlight_max(axis=0, color="#cfe3cf"),
                     width="stretch")
        st.caption("Green marks the best model for each metric.")
        figure = draw_comparison(table)
        st.pyplot(figure)
        plt.close(figure)

        best = table["MCC"].idxmax()
        st.success(f"Highest MCC on this data: **{best}** "
                   f"({table.loc[best, 'MCC']:.4f}).")


# ---------- tab 3: row-by-row predictions ----------

with tab_predict:
    model = MODELS[chosen]
    proba = model.predict_proba(X)
    result = pd.DataFrame({
        "predicted": model.predict(X),
        "confidence": proba.max(axis=1).round(3),
    })
    if has_target:
        result.insert(0, "true", data[TARGET].values)
        result["correct"] = result["true"] == result["predicted"]
        st.metric("Plates classified correctly",
                  f"{int(result['correct'].sum())} / {len(result)}")
        only_wrong = st.checkbox("Show only the mistakes")
        shown = result[~result["correct"]] if only_wrong else result
    else:
        shown = result
    st.dataframe(shown, width="stretch", height=380)
    st.download_button("Download predictions as CSV",
                       result.to_csv(index=False).encode(),
                       file_name=f"predictions_{chosen.replace(' ', '_')}"
                                 ".csv", mime="text/csv")


# ---------- tab 4: background ----------

with tab_about:
    st.subheader("The fault types")
    st.table(pd.DataFrame({"description": FAULT_NOTES}))

    st.subheader("How the models were built")
    st.markdown(
        f"* {INFO['train_rows']} training plates, {INFO['test_rows']} "
        "held-out test plates (stratified split, seed "
        f"{INFO['seed']}).\n"
        "* Each model is a scikit-learn pipeline (scaling + classifier), "
        "tuned with 5-fold cross-validation using the **Matthews "
        "correlation coefficient** as the score.\n"
        "* The app only loads the saved pipelines from `model/`; uploaded "
        "data is never used for training.")
    summary = pd.DataFrame({
        name: {"cross-validated MCC":
               f"{d['cv_mcc_mean']:.4f} ± {d['cv_mcc_std']:.4f}",
               "settings tried": d["settings_tried"],
               "chosen settings": ", ".join(f"{k}={v}" for k, v in
                                            d["best_params"].items())}
        for name, d in INFO["models"].items()}).T
    st.dataframe(summary, width="stretch")
