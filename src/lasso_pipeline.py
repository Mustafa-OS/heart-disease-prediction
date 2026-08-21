"""LASSO (L1-penalised) logistic regression on the cleaned Cleveland dataset.

Transcribed from my code listing in Appendix A of the group report, with two
minimal adaptations so it runs headless from this repository:
  - the input path points at data/heart_cleaned.csv (produced by
    src/prepare_data.py, which reproduces the group's cleaning step);
  - plt.show() calls are replaced by plt.savefig(...) into docs/img/.

Pipeline: 60/20/20 stratified train/validation/test split (seed 42),
standardisation fitted on the training set only, C swept from 0.01 to 10 on
the validation set, final model refitted at the chosen C = 0.3 and evaluated
once on the held-back test set.
"""

from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # render figures without a display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "docs" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "heart_cleaned.csv")
y = df["target"]
# removing target column
X = df.drop(columns=["target"])

# Splitting the Data (60-20-20 -> training - validation - test)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
print("Split sizes - train:", len(X_train), "| validation:", len(X_val), "| test:", len(X_test))

# Standardising Data (scaler fitted on the training set only)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Hyperparameter Tuning - Accuracy
C_values = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10]
accTrainList = []
accValList = []
for C in C_values:
    model = logreg(penalty="l1", solver="liblinear", C=C, max_iter=1000)
    model.fit(X_train, y_train)
    accTrainList.append(accuracy_score(y_train, model.predict(X_train)))
    accValList.append(accuracy_score(y_val, model.predict(X_val)))

plt.figure(figsize=(8, 5))
plt.plot(C_values, accTrainList, label="Training set")
plt.plot(C_values, accValList, label="Validation set")
plt.axvline(0.3, color="green", linestyle="--", label="Selected C = 0.3")
plt.xscale("log")
plt.title("Accuracy vs C")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.savefig(IMG_DIR / "repro_accuracy_vs_c.png", dpi=150, bbox_inches="tight")
plt.close()

# Hyperparameter Tuning - Recall
recTrainList = []
recValList = []
for C in C_values:
    model = logreg(penalty="l1", solver="liblinear", C=C, max_iter=1000)
    model.fit(X_train, y_train)
    recTrainList.append(recall_score(y_train, model.predict(X_train)))
    recValList.append(recall_score(y_val, model.predict(X_val)))

plt.figure(figsize=(8, 5))
plt.plot(C_values, recTrainList, label="Training set")
plt.plot(C_values, recValList, label="Validation set")
plt.axvline(0.3, color="green", linestyle="--", label="Selected C = 0.3")
plt.xscale("log")
plt.title("Recall vs C")
plt.xlabel("C")
plt.ylabel("Recall")
plt.grid(True)
plt.legend()
plt.savefig(IMG_DIR / "repro_recall_vs_c.png", dpi=150, bbox_inches="tight")
plt.close()

# Final Model with C = 0.3 chosen
C = 0.3
final_model = logreg(penalty="l1", solver="liblinear", C=C, max_iter=1000)
final_model.fit(X_train, y_train)

# Performance of Model - Accuracy, Recall and Precision scores
for name, Xset, yset in [("Train", X_train, y_train), ("Validation", X_val, y_val), ("Test", X_test, y_test)]:
    pred = final_model.predict(Xset)
    print(name, accuracy_score(yset, pred), recall_score(yset, pred), precision_score(yset, pred))

# Confusion Matrix
cm = confusion_matrix(y_test, final_model.predict(X_test))
tn, fp, fn, tp = cm.ravel()
print("TP:", tp, "| FN:", fn)
print("FP:", fp, "| TN:", tn)

# List of coefficients, in order of importance
coefficients = pd.Series(final_model.coef_[0], index=X.columns)
print(coefficients.sort_values())
print("Features retained:", (coefficients != 0).sum(),
      "| dropped:", list(coefficients[coefficients == 0].index))
