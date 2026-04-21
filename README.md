# 🏦 Customer Investment Needs — ML Recommendation Engine

> An end-to-end machine learning pipeline for classifying client investment needs and generating personalized Next Best Action (NBA) recommendations in a wealth management context.

Built as a business case module by **Raffaele Zenti** (Co-Founder & Chief AI Officer, Whealthype-AI SpA) at **Politecnico di Milano — Mathematical Engineering (Quantitative Finance Track)**.

---

## 📌 Problem Statement

Given anonymized customer data from a large wealth management company, the goal is to:

1. **Classify** each client's investment need — *Income* (lump-sum, decumulation) or *Accumulation* (regular contributions, wealth building)
2. **Recommend** the most suitable specific investment product based on predicted need and individual risk propensity
3. **Flag** uncertain predictions for human advisor review, in compliance with **MiFID II / IDD** regulatory requirements

The targets are derived via a **revealed preference scheme**: if a trusted advisor sold a client a product satisfying a given need, that client is labelled as having that need. The ML model is thus a *clone of a reliable financial advisor*.

---

## 🗂️ Repository Structure

```
├── 0-DataPreprocessing.ipynb          # EDA, feature engineering, FAMD, GMM clustering
├── Recommendation_Engine_final.ipynb  # Full modeling pipeline, ensemble, recommendations
├── models.py                          # OOP model wrappers + ModelFactory (13 models)
├── metrics.py                         # Custom business metrics (safety scorer, balanced scorer)
├── utilities.py                       # CV pipeline, OOF stacking, entropy utilities, experiment runner
├── plots.py                           # All visualization functions
└── README.md
```

---

## ⚙️ Pipeline Overview

### 1. Preprocessing (`0-DataPreprocessing.ipynb`)
- Log transformation of skewed financial variables (`Wealth`, `Income`)
- Feature engineering: `IncomePerFamilyMember_log`, `Income_Wealth_Ratio_log`, `IncomeDivAge`
- Systematic interaction feature generation and evaluation via signal/collinearity scoring
- **FAMD** (Factor Analysis of Mixed Data) for dimensionality reduction on mixed numerical/categorical features
- **GMM clustering** sweep (k=2–10) with BIC, AIC, and Silhouette evaluation
- Supervised EDA: point-biserial correlations, pairplots, boxplots by target class
- Export of processed feature matrix to `processed_features.xlsx`

### 2. Modeling (`Recommendation_Engine_final.ipynb`)
- Stratified 80/20 train/test split — scaler fit on training data **only** (no leakage)
- **13 model classes** evaluated: Logistic Regression, SGD, KNN, SVM, GaussianNB, BernoulliNB, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost
- **Bayesian hyperparameter tuning** via `BayesSearchCV` / `GridSearchCV` per model
- Stratified k-fold cross-validation with parallel execution (`joblib`)
- Custom three-metric evaluation framework (see Metrics section)
- **Entropy diagnostics**: top-5% highest-entropy training samples flagged; test set split into low/high-entropy subsets for performance analysis
- SHAP values and feature importance for interpretability (XAI)

### 3. Stacking Ensemble
- **Out-of-Fold (OOF) meta-feature generation**: each base model predicts on held-out folds it never trained on → honest, leakage-free meta-features
- **XGBoost meta-learner** trained on the OOF probability matrix
- Base models refitted on full training set for final inference
- Separate ensemble trained for each target (Income, Accumulation)

### 4. Recommendation Engine
- **Entropy guardrail**: predictions in the top 5% uncertainty quantile → `ADVISOR_REVIEW` (MiFID/IDD compliance)
- **Optimized thresholds**: Income=0.35, Accumulation=0.45 (precision/recall trade-off tuned for business objectives)
- **Conflict resolution**: if both needs predicted, higher probability wins
- **Risk-bucket product matching**: client's `RiskPropensity` score mapped to the highest-risk suitable product below their tolerance threshold
- Outputs: `INCOME` · `ACCUMULATION` · `NO_ACTION` · `ADVISOR_REVIEW` + specific product name

---

## 📊 Custom Business Metrics

Standard accuracy is insufficient given class imbalance and asymmetric error costs in financial services. Three metrics are used:

| Metric | Type | Business meaning |
|---|---|---|
| **Recall(class 0) ≥ 0.90** | Hard constraint | Reject models that pass too many unsuitable recommendations |
| **Safety-Constrained Precision** | Primary objective | Quality metric among models that pass the safety floor |
| **Balanced Business Score** (α=0.7) | Secondary objective | Weighted trade-off between coverage (recall) and quality (precision) |

A model that fails the recall floor receives a safety score of **-1** and is treated as unsafe regardless of other metrics.

---

## 🏗️ Code Architecture

Models are implemented via a clean OOP hierarchy:

```python
BaseModel
├── train(X_train, y_train)
├── predict(X)
├── predict_proba(X)
├── evaluate(X_train, y_train, X_test, y_test)   # CV + test
└── tune(X_train, y_train)                        # BayesSearchCV / GridSearchCV

ModelFactory.create("xgb")   # Returns any of 13 model instances
```

The `run_experiment()` utility handles the full leakage-safe pipeline:
1. Bayesian tuning on `X_train` only
2. Stratified CV + test evaluation
3. Final fit on full `X_train`
4. Probability extraction for ensemble construction
5. Diagnostic plots (confusion matrix + ROC)

---

## 🧠 Key Design Decisions

**Why separate binary classifiers?**
One model per need rather than a single multi-output classifier — simpler interpretation, independent feature importance, greater robustness, easier maintenance.

**Why OOF for stacking?**
Fitting base models on full training data and predicting back on it produces inflated, optimistic probabilities. The meta-model would learn from dishonest signals that don't exist at test time. OOF forces each base model to predict on data it never saw, replicating test-time conditions during meta-feature construction.

**Why entropy for the advisor guardrail?**
Binary entropy $H = -p\log_2 p - (1-p)\log_2(1-p)$ quantifies prediction uncertainty directly. Routing the top 5% most uncertain predictions to a human advisor is a principled, model-agnostic safety mechanism that maps naturally to MiFID II's "Best Interest" requirement.

**Why FAMD instead of PCA?**
The feature matrix contains one categorical variable (Gender). PCA operates on numerical data only. FAMD handles mixed data natively without requiring manual one-hot encoding, and does not need pre-scaling.

---

## 📦 Dependencies

```
numpy, pandas, scikit-learn, xgboost, lightgbm, catboost
scikit-optimize (BayesSearchCV)
prince (FAMD)
shap
matplotlib, seaborn, scipy
joblib, tabulate, openpyxl
```

---

## 📋 Regulatory Context

The recommendation engine is designed with **MiFID II / IDD** compliance in mind:

- The entropy guardrail ensures that the most uncertain automated recommendations are never delivered directly to clients — they are escalated to a human advisor
- The safety-constrained precision metric with a hard recall floor on class 0 operationalizes the "Best Interest" principle: the model must not miss clients for whom a product is unsuitable
- Risk-bucket product matching ensures that recommended products never exceed the client's stated risk tolerance (suitability requirement)

---

## 👤 Author

**Amirreza Khajouei**
MMF — Mathematical Engineering (Quantitative Finance Track)
Politecnico di Milano

[LinkedIn](https://linkedin.com/in/amirreza-khajouei) · [GitHub](https://github.com/amirKhaju)
