"""Breast Cancer Diagnosis: end-to-end classification project.
Dataset: Wisconsin Diagnostic Breast Cancer (569 samples, 30 features), bundled with scikit-learn.
Steps: EDA -> preprocessing -> model comparison (5-fold CV) -> tuning -> test evaluation -> feature importance.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RS = 42
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target  # target: 0 = malignant, 1 = benign
print("Shape:", X.shape, "| Missing values:", int(X.isna().sum().sum()))
print("Class balance:\n", y.map({0: "malignant", 1: "benign"}).value_counts())

# ---- EDA ----
plt.figure(figsize=(5, 3.5))
y.map({0: "Malignant", 1: "Benign"}).value_counts().plot.bar(color=["#4A86E8", "#9fc0f5"])
plt.title("Class distribution"); plt.ylabel("Samples"); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150); plt.close()

top = X.corrwith(y).abs().sort_values(ascending=False).head(10).index
plt.figure(figsize=(8, 6))
sns.heatmap(X[top].corr(), annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 7})
plt.title("Correlation of 10 features most related to diagnosis"); plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150); plt.close()

# ---- Split ----
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)

# ---- Model comparison with 5-fold CV (scaling inside the pipeline avoids data leakage) ----
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RS),
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "SVM (RBF)": SVC(probability=True, random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RS),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
rows = []
for name, m in models.items():
    pipe = Pipeline([("scale", StandardScaler()), ("clf", m)])
    s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy")
    rows.append({"Model": name, "CV accuracy": round(s.mean(), 4), "Std": round(s.std(), 4)})
cv_df = pd.DataFrame(rows).sort_values("CV accuracy", ascending=False)
print("\n", cv_df.to_string(index=False))
cv_df.to_csv("model_comparison.csv", index=False)

# ---- Tune the two strongest candidates (macro F1 treats both classes fairly) ----
searches = {
    "Logistic Regression": (LogisticRegression(max_iter=5000), {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    "SVM (RBF)": (SVC(probability=True, random_state=RS), {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]}),
}
tuned = {}
for name, (est, params) in searches.items():
    g = GridSearchCV(Pipeline([("scale", StandardScaler()), ("clf", est)]), params, cv=cv, scoring="f1_macro")
    g.fit(X_tr, y_tr)
    tuned[name] = g
    print(f"Tuned {name}: CV macro-F1 = {g.best_score_:.4f} | {g.best_params_}")
best_name = max(tuned, key=lambda n: tuned[n].best_score_)
grid = tuned[best_name]
best = grid.best_estimator_
print("Selected model:", best_name)

# ---- Final test evaluation ----
pred = best.predict(X_te)
proba = best.predict_proba(X_te)[:, 1]
print("\n", classification_report(y_te, pred, target_names=["malignant", "benign"]))
results = {
    "test_accuracy": round(float((pred == y_te).mean()), 4),
    "roc_auc": round(roc_auc_score(y_te, proba), 4),
    "malignant_recall": round(recall_score(y_te, pred, pos_label=0), 4),
    "f1_weighted": round(f1_score(y_te, pred, average="weighted"), 4),
    "selected_model": best_name,
    "best_params": grid.best_params_,
}
print(results)
json.dump(results, open("results.json", "w"), indent=2)
ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Malignant", "Benign"], cmap="Blues")
plt.title(f"Tuned {best_name} - test set"); plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

# ---- Feature importance (Random Forest) ----
rf = RandomForestClassifier(n_estimators=300, random_state=RS).fit(X_tr, y_tr)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10)
plt.figure(figsize=(7, 4.5)); imp.plot.barh(color="#4A86E8")
plt.title("Top 10 features (Random Forest)"); plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()

# ---- PCA: how many components keep 95% variance? ----
pca = PCA().fit(StandardScaler().fit_transform(X_tr))
k = int((pca.explained_variance_ratio_.cumsum() < 0.95).sum() + 1)
print(f"\nPCA: {k} of 30 components explain 95% of variance")
json.dump({**results, "pca_components_95pct": k, "top_feature": imp.index[-1]}, open("results.json", "w"), indent=2)
