# Breast Cancer Diagnosis – ML Classification

End-to-end machine learning project that predicts whether a tumour is **malignant or benign** from 30 cell-nucleus measurements.

**Dataset:** Wisconsin Diagnostic Breast Cancer (569 samples, 30 numeric features, no missing values) – bundled with scikit-learn.

## Workflow
1. **EDA** – class balance (357 benign / 212 malignant), correlation heatmap.
2. **Preprocessing** – stratified 80/20 split; `StandardScaler` inside a `Pipeline` so no data leaks from the test set.
3. **Model comparison** – 5 models with stratified 5-fold CV.
4. **Tuning** – `GridSearchCV` (macro-F1) on the two strongest models.
5. **Evaluation** – held-out test set, confusion matrix, ROC-AUC.
6. **Interpretation** – Random Forest feature importance and PCA.

## Results
| Model | 5-fold CV accuracy |
|---|---|
| Logistic Regression | 97.8% |
| SVM (RBF) | 96.9% |
| Random Forest | 96.5% |
| Gradient Boosting | 95.2% |
| Decision Tree | 91.7% |

**Tuned Logistic Regression (C=0.1) on the test set:** accuracy **97.4%**, ROC-AUC **0.996**, malignant recall **95.2%**.
Most important feature: *worst perimeter*. 10 of 30 PCA components explain 95% of the variance.

## Run it
```bash
pip install -r requirements.txt
python breast_cancer_classification.py
```
Outputs: charts (`*.png`), `model_comparison.csv`, `results.json`.

## Limitations
Small, clean benchmark dataset – results show a correct workflow, not clinical readiness. In a medical setting, recall on malignant cases matters most.

**Tech:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
