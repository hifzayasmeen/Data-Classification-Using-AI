"""
DecodeLabs - AI Project 2: Data Classification Using AI
Pipeline: INPUT (Iris + Scaling) -> PROCESS (Split + KNN) -> OUTPUT (Confusion Matrix + F1)
"""

import matplotlib
matplotlib.use("Agg")  # works without a display; remove if running in Jupyter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. INPUT: load and understand the dataset
# ------------------------------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)
print(f"Samples   : {df.shape[0]}")
print(f"Features  : {len(iris.feature_names)} -> {iris.feature_names}")
print(f"Classes   : {list(iris.target_names)}")
print(f"Missing   : {int(df.isnull().sum().sum())}")
print("\nClass balance:")
print(df["species"].value_counts().to_string())
print("\nSummary statistics:")
print(df[iris.feature_names].describe().round(2).to_string())

sns.pairplot(df.drop(columns="target"), hue="species", corner=True)
plt.savefig("01_pairplot.png", dpi=120, bbox_inches="tight")
plt.close()

X = iris.data
y = iris.target

# ------------------------------------------------------------------
# 2. PROCESS: shuffled train/test split (80/20)
#    stratify keeps the class ratio equal in both sets
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=RANDOM_STATE
)
print("\n" + "=" * 60)
print("2. TRAIN / TEST SPLIT")
print("=" * 60)
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ------------------------------------------------------------------
# 3. Feature scaling (Gatekeeper rule)
#    fit ONLY on train data, then transform both -> no data leakage
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print("\n3. SCALING (StandardScaler, fit on train only)")
print(f"Train mean ~ {X_train_s.mean(axis=0).round(2)} | std ~ {X_train_s.std(axis=0).round(2)}")

# ------------------------------------------------------------------
# 4. Choose K (the elbow): compare error rate for K = 1..30
#    Uses 5-fold CV on the training set only - test set stays locked.
# ------------------------------------------------------------------
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
errors = []
for k in k_range:
    acc = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                          X_train_s, y_train, cv=5).mean()
    errors.append(1 - acc)

best_k = k_range[int(np.argmin(errors))]
print("\n" + "=" * 60)
print("4. TUNING K")
print("=" * 60)
print(f"Best K by cross-validation: {best_k} (CV error = {min(errors):.4f})")

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), errors, marker="o")
plt.axvline(best_k, color="orangered", linestyle="--", label=f"Best K = {best_k}")
plt.xlabel("K value")
plt.ylabel("Error rate (5-fold CV)")
plt.title("Choosing K: the elbow")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("02_choose_k.png", dpi=120, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# 5. Train the model: Instantiate -> Fit -> Predict
# ------------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=best_k)   # INSTANTIATE
model.fit(X_train_s, y_train)                      # FIT
predictions = model.predict(X_test_s)              # PREDICT

# ------------------------------------------------------------------
# 6. OUTPUT: validation (accuracy is not enough -> confusion matrix + F1)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. RESULTS ON TEST SET")
print("=" * 60)
print(f"Accuracy      : {accuracy_score(y_test, predictions):.4f}")
print(f"F1 (macro)    : {f1_score(y_test, predictions, average='macro'):.4f}")
print(f"F1 (weighted) : {f1_score(y_test, predictions, average='weighted'):.4f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, display_labels=iris.target_names, cmap="Blues", ax=ax
)
ax.set_title(f"Confusion Matrix (KNN, K={best_k})")
plt.savefig("03_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# 7. Bonus: predict a brand-new flower (the Conclusion's "test with new data")
# ------------------------------------------------------------------
new_flower = pd.DataFrame([[5.9, 3.0, 5.1, 1.8]], columns=iris.feature_names)
pred = model.predict(scaler.transform(new_flower))[0]
print(f"New flower {new_flower.values.tolist()[0]} -> predicted: {iris.target_names[pred]}")
