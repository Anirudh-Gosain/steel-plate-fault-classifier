# Raw data

Steel Plates Faults, UCI Machine Learning Repository (dataset id 198).
https://archive.ics.uci.edu/dataset/198/steel+plates+faults
Donated by Semeion, Research Center of Sciences of Communication, Rome, Italy.
Licence: CC BY 4.0.

- `Faults.NNA`     1941 rows, tab separated, no header: 27 features followed
                   by 7 one-hot fault columns.
- `Faults27x7_var` the 34 column names, one per line.

The files are kept exactly as downloaded. `train_models.py` reads them
directly, so the whole pipeline can be rebuilt from this folder.
