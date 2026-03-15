# 🫀 Heart Disease Classifier
 
A machine learning project that predicts the presence of heart disease using the Cleveland Heart Disease dataset. Built to practice end-to-end sklearn workflows including preprocessing pipelines, model evaluation, feature selection, and hyperparameter tuning.
 
---
 
## 📊 Results
 
| Model | ROC-AUC (5-Fold CV) |
|-------|---------------------|
| Logistic Regression (L2) | 0.919 |
| Logistic Regression (L1) | 0.915 |
| SVM (rbf kernel) | 0.906 |
| Random Forest | 0.912 |
| Gradient Boosting | 0.879 |
| Decision Tree | 0.781 |
 
**Best model: Logistic Regression — 0.919 ROC-AUC**
 
---
 
## 🔍 Key Findings
 
- **Dataset had 723 duplicate rows** (1025 → 302 after deduplication) — caused artificially perfect scores before cleaning
- **Top predictors** of heart disease (by Random Forest importance):
  - Chest pain type (`cp`) — strongest signal
  - Max heart rate achieved (`thalach`)
  - ST depression (`oldpeak`)
  - Number of major vessels (`ca`)
  - Age
- Logistic Regression outperformed tree-based models on this small dataset (302 rows), consistent with the bias-variance tradeoff at low sample sizes
 
---
 
## 🛠️ What's Covered
 
- **Preprocessing** — `StandardScaler`, `OneHotEncoder`, `SimpleImputer` via `ColumnTransformer`
- **Pipelines** — Full `sklearn.Pipeline` to prevent data leakage
- **Evaluation** — `StratifiedKFold`, `cross_validate`, ROC curve, Precision-Recall curve
- **Model comparison** — 6 algorithms benchmarked on the same pipeline
- **Hyperparameter tuning** — `GridSearchCV` on Gradient Boosting
- **Feature selection** — `SelectKBest` (F-test) and `RFE`
- **Dimensionality reduction** — PCA (27 features → 15 components at 95% variance)
 
---
 
## 📁 Dataset
 
Cleveland Heart Disease dataset — 302 patients, 13 features, binary target (1 = disease present).
 
Download from Kaggle and place in `data/heart.csv`:
https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci
 
| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | 0 = female, 1 = male |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure |
| chol | Cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120mg/dl (0/1) |
| restecg | Resting ECG result (0–2) |
| thalach | Max heart rate achieved |
| exang | Exercise induced angina (0/1) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak ST segment (0–2) |
| ca | Number of major vessels (0–3) |
| thal | Thalassemia type (0–3) |
 
---
 
## ⚙️ Stack
 
- Python 3
- scikit-learn
- pandas
- NumPy
- Matplotlib
 
---
 
## 🚀 How to Run
 
```bash
# Clone the repo
git clone https://github.com/Arhamkhannn/heart-disease-pridiction.git
cd heart-disease-pridiction
 
# Install dependencies
pip install scikit-learn pandas numpy matplotlib jupyter
 
# Download dataset and place at data/heart.csv
 
# Launch notebook
jupyter notebook Heart_Disease_Pridiction.ipynb
```
