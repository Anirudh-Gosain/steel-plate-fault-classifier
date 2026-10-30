# Steel Plate Fault Classification

ML Assignment 2 · M.Tech (AIML) · Anirudh Gosain (2026AK05290)

- **GitHub repository:** https://github.com/Anirudh-Gosain/steel-plate-fault-classifier
- **Live Streamlit app:** https://steel-fault-inspector.streamlit.app/

## a. Problem statement

When steel plates are rolled, surface defects appear that have to be sorted
by type: some are cosmetic, others mean the plate must be reworked or
rejected. The task is to predict the **type of fault** on a plate from 26
measurements of the defect region (its position and size, perimeter,
luminosity, steel type, plate thickness and several shape indices).

It is a **7-class classification problem**, and the classes are very uneven
(673 plates of the largest class, 55 of the smallest). Five classifiers
(Logistic Regression, Decision Tree, k-Nearest Neighbours, Gaussian Naive
Bayes and an RBF Support Vector Machine) are trained on the same data, tuned
the same way and compared on the same held-out test set. The saved models
are then served through a Streamlit app that scores any uploaded test CSV.

## b. Dataset description

| | |
|---|---|
| Source | [UCI Machine Learning Repository: Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) (id 198), donated by Semeion, Rome; CC BY 4.0 |
| Instances | 1,941 steel plates, no missing values, no duplicate rows |
| Features | 27 numeric in the raw file; 26 used (see below) |
| Target | `Fault_Type`, 7 classes |
| Split | stratified 80 / 20 → 1,552 training plates, 389 test plates (`test_data.csv`), seed 5290 |

**Classes** (whole dataset → test set):

| Fault_Type | Plates | In test set |
|---|---:|---:|
| Other_Faults | 673 | 135 |
| Bumps | 402 | 81 |
| K_Scatch | 391 | 78 |
| Z_Scratch | 190 | 38 |
| Pastry | 158 | 32 |
| Stains | 72 | 14 |
| Dirtiness | 55 | 11 |

**Features.** Geometry of the defect (`X_Minimum`, `X_Maximum`, `Y_Minimum`,
`Y_Maximum`, `Pixels_Areas`, `X_Perimeter`, `Y_Perimeter`), luminosity
(`Sum_of_Luminosity`, `Minimum_of_Luminosity`, `Maximum_of_Luminosity`), the
production line (`Length_of_Conveyer`, `TypeOfSteel_A300`,
`Steel_Plate_Thickness`) and derived indices (`Edges_Index`, `Empty_Index`,
`Square_Index`, `Outside_X_Index`, `Edges_X_Index`, `Edges_Y_Index`,
`Outside_Global_Index`, `LogOfAreas`, `Log_X_Index`, `Log_Y_Index`,
`Orientation_Index`, `Luminosity_Index`, `SigmoidOfAreas`).

**Preparation.**

- The raw file has the label as 7 one-hot columns with exactly one set per
  row; they are collapsed into the single `Fault_Type` column.
- `TypeOfSteel_A400` is dropped because it always equals
  1 − `TypeOfSteel_A300`, so it adds no information and is perfectly
  collinear. This leaves 26 features.
- Several size features are extremely skewed (skewness 39 for `Y_Perimeter`,
  22 for `X_Perimeter`, 14 for `Pixels_Areas`). For the scale-sensitive
  models the grid search therefore chose between `StandardScaler` and a
  Yeo-Johnson `PowerTransformer`. It picked `StandardScaler` every time.

## c. GitHub repository link

https://github.com/Anirudh-Gosain/steel-plate-fault-classifier

```
steel-plate-fault-classifier/
├── app.py                  Streamlit app (loads the saved models, never retrains)
├── train_models.py         data preparation, tuning, evaluation, saving
├── fault_metrics.py        the six metrics, confusion matrix, report
│                           (shared by training and the app)
├── requirements.txt
├── README.md
├── test_data.csv           the 389 held-out test plates (26 features + Fault_Type)
├── model/                  the five fitted pipelines (.joblib) + model_info.json
├── results/                test metrics, per-class recall, confusion matrices (CSV)
├── data/raw/               the original UCI files, unchanged
└── .streamlit/config.toml  app theme
```

## d. Models used

Every model is a scikit-learn `Pipeline` (scaler + classifier, no scaler
for the tree), so preprocessing learnt on the training data is saved inside
the model. Hyper-parameters were tuned by `GridSearchCV` with the same
stratified 5-fold split for all five models. The tuning score was the
**Matthews correlation coefficient (MCC)**, because plain accuracy can look
good while the rare faults are ignored. Each tuned model was then scored
**once** on the 389 test plates.

| Model | Tuned settings | CV MCC (5-fold) |
|---|---|---:|
| Logistic Regression | StandardScaler, C = 10 | 0.643 ± 0.020 |
| Decision Tree | entropy, max_depth = 8, min_samples_leaf = 10 | 0.659 ± 0.026 |
| KNN | StandardScaler, k = 9, Manhattan distance (p = 1), distance weighting | 0.692 ± 0.028 |
| Naive Bayes (Gaussian) | StandardScaler, var_smoothing = 0.3 | 0.546 ± 0.027 |
| SVM (RBF) | StandardScaler, C = 3, gamma = 0.1, class_weight = balanced, Platt-scaled probabilities | 0.706 ± 0.029 |

### Comparison table (held-out test set, 389 plates)

Precision, Recall and F1 are **macro averages** over the 7 fault types, so
every type counts equally. AUC is the **macro one-vs-rest** ROC AUC from the
predicted probabilities.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6838 | 0.9234 | 0.6734 | 0.7058 | 0.6876 | 0.5941 |
| Decision Tree | **0.7249** | 0.9200 | **0.7674** | 0.7453 | **0.7526** | **0.6470** |
| KNN | 0.7147 | 0.9229 | 0.7364 | 0.7505 | 0.7345 | 0.6321 |
| Naive Bayes | 0.5733 | 0.9043 | 0.5490 | 0.6825 | 0.5706 | 0.5081 |
| Support Vector Machine | 0.7095 | **0.9238** | 0.7256 | **0.7732** | 0.7448 | 0.6348 |

Recall for each fault type on the test set (share of each fault that is
caught):

| Fault_Type (test plates) | Logistic Reg. | Decision Tree | KNN | Naive Bayes | SVM |
|---|---:|---:|---:|---:|---:|
| Bumps (81) | 0.543 | 0.679 | 0.642 | 0.753 | 0.630 |
| Dirtiness (11) | 0.636 | 0.727 | 0.909 | 0.636 | 0.818 |
| K_Scatch (78) | 0.923 | 0.949 | 0.949 | 0.833 | 0.949 |
| Other_Faults (135) | 0.637 | 0.652 | 0.652 | 0.207 | 0.570 |
| Pastry (32) | 0.438 | 0.500 | 0.313 | 0.531 | 0.656 |
| Stains (14) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Z_Scratch (38) | 0.763 | 0.711 | 0.789 | 0.816 | 0.789 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A sound linear baseline: AUC 0.923 shows it ranks the classes well, but it has the lowest test MCC (0.594) of the four stronger models. Clearly separable faults are easy for it (Stains 100 %, K_Scatch 92 %). It struggles where the classes overlap: only 54 % of Bumps and 44 % of Pastry are caught, and Bumps and Other_Faults are confused 52 times between them. Tuning picked weak regularisation (C = 10), so even with plenty of freedom the linear boundaries cannot separate the overlapping faults. |
| Decision Tree | Best on 4 of the 6 test metrics: accuracy 0.725, precision 0.767, F1 0.753 and MCC 0.647. Limiting it to depth 8 with at least 10 plates per leaf (61 leaves) avoids overfitting. It splits mainly on `Outside_X_Index`, `Length_of_Conveyer`, the steel type and the plate thickness, threshold-type rules that a tree finds naturally. It has the lowest AUC (0.920), because a tree's probabilities are coarse leaf proportions. Its cross-validated MCC (0.659) is also below KNN and SVM, so its first place on this test set is partly luck of the split. |
| KNN | Strong and consistent: CV MCC 0.692, test MCC 0.632. Manhattan distance on standardised features with distance weighting worked best, and k = 9 smooths out noise. It has the best recall on the rare Dirtiness class (10 of 11) and ties for best on K_Scatch. Its weak spot is Pastry (31 %), whose plates sit inside the Bumps / Other_Faults region, where the nearest neighbours mostly belong to the bigger classes. |
| Naive Bayes | Clearly the weakest: MCC 0.508, F1 0.571. Its assumption that features are independent fails here, because many features are near-copies of each other (`Pixels_Areas`, `LogOfAreas`, `SigmoidOfAreas`, perimeters, luminosity sums). The same evidence is counted several times and the probabilities become over-confident. It labels 79 of the 135 Other_Faults test plates as Bumps or Pastry, catching only 21 % of that class. It still finds the small classes reasonably well (Bumps 75 %, Z_Scratch 82 %), so its macro recall (0.683) holds up better than its precision (0.549). |
| Support Vector Machine | The most reliable model overall: the best cross-validated MCC (0.706), the best test AUC (0.924) and the best macro recall (0.773). With balanced class weights it catches the rare and hard classes best: Pastry 66 % (the next best model gets 53 %) and Dirtiness 82 %. The price is lower recall on the large Other_Faults class (57 %), so on this test set its accuracy and MCC end up just below the Decision Tree's. |
| **Overall winner for this dataset** | **Decision Tree** on the held-out test set: highest accuracy, precision, F1 and MCC, and simple threshold rules that can be read and explained. The **SVM** is a very close second (MCC 0.635 against 0.647, a gap smaller than the ±0.03 spread seen across CV folds). It is the safer choice if catching rare faults matters most, as it has the best CV MCC, AUC and macro recall. Every model's main source of error is the same: telling **Bumps** and **Other_Faults** apart, two broad classes whose measurements overlap. |

## Streamlit app

The app (`app.py`) provides:

1. **CSV upload** of test data (or one click to use the bundled
   `test_data.csv`, which can also be downloaded from the sidebar)
2. a **model selection dropdown** for the five classifiers
3. the **six evaluation metrics** for the chosen model
4. the **confusion matrix** and **classification report**
5. a **comparison tab** scoring all five models on the uploaded file, plus
   a tab of row-by-row predictions that can be downloaded

The uploaded CSV must contain the 26 feature columns and the `Fault_Type`
column. The app only **loads** the saved pipelines from `model/` and calls
`predict` / `predict_proba`; no model is retrained on the uploaded data.

## Running it

```bash
pip install -r requirements.txt     # Python 3.12 or newer
python train_models.py              # rebuilds test_data.csv, model/ and results/
streamlit run app.py
```

The models were trained with Python 3.14, scikit-learn 1.9.1, pandas 3.1.0
and numpy 2.5.1. `requirements.txt` pins the same versions so the saved
`.joblib` files load exactly as they were saved.
