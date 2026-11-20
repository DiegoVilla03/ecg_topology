# Synthetic demo results

80 synthetic windows per class, 200 beats each, seed 7. Cross-validation: 5 folds x 3 repeats. These numbers describe the generator in `ecgtda/synthetic.py`, not real patients.

## Classification

| task | features | model | accuracy | accuracy_sd | f1_macro |
|---|---|---|---|---|---|
| 3 classes: normal vs CHF-like vs AV block II | HRV baseline | Random forest | 0.983 | 0.016 | 0.983 |
| 3 classes: normal vs CHF-like vs AV block II | HRV baseline | SVM (RBF) | 0.944 | 0.036 | 0.944 |
| 3 classes: normal vs CHF-like vs AV block II | Topological | Random forest | 0.967 | 0.028 | 0.966 |
| 3 classes: normal vs CHF-like vs AV block II | Topological | SVM (RBF) | 0.946 | 0.021 | 0.945 |
| 3 classes: normal vs CHF-like vs AV block II | HRV + topological | Random forest | 0.972 | 0.024 | 0.972 |
| 3 classes: normal vs CHF-like vs AV block II | HRV + topological | SVM (RBF) | 0.962 | 0.031 | 0.962 |
| Negative control: normal vs AV block I | HRV baseline | Random forest | 0.523 | 0.062 | 0.518 |
| Negative control: normal vs AV block I | HRV baseline | SVM (RBF) | 0.59 | 0.063 | 0.583 |
| Negative control: normal vs AV block I | Topological | Random forest | 0.502 | 0.058 | 0.499 |
| Negative control: normal vs AV block I | Topological | SVM (RBF) | 0.519 | 0.053 | 0.511 |
| Negative control: normal vs AV block I | HRV + topological | Random forest | 0.492 | 0.067 | 0.489 |
| Negative control: normal vs AV block I | HRV + topological | SVM (RBF) | 0.531 | 0.065 | 0.525 |

## Paroxysmal AV block: window-level detection (AUC)

Positive window = contains at least one RR interval that spans a blocked P wave.

| score | AUC |
|---|---|
| H0 max persistence (topological) | 1.000 |
| H0 max / median persistence (topological) | 1.000 |
| max RR / median RR (baseline) | 1.000 |
