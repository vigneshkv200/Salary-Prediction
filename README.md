# FairPay-Texas: Scalable Public Sector Salary Prediction & Algorithmic Fairness Auditing

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) 
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/) 
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-green.svg)](https://xgboost.readthedocs.io/) 
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-018285.svg)](https://mlflow.org/) 
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blueviolet.svg)](https://github.com/shap/shap)

An end-to-end, production-grade machine learning and statistical auditing pipeline designed to predict annual public sector salaries in Texas (149K+ records). This repository demonstrates advanced data cleaning patterns, feature extraction, a competitive 9-model validation tournament, experiment tracking via **MLflow**, and a rigorous **AI Fairness & Bias Audit** across protected demographic groups.

---

##  Project Architecture & Key Highlights

* **Top Performing Model:** **XGBoost Regressor** (Test $R^2$: **0.9186**, Test MAE: **$3,642.51**).
* **Core Challenge:** Modeling high-cardinality categorical variables (Agencies & Job Titles) while actively protecting the pipeline from target leakage and demographic proxy variables.
* **Strategic Focus:** Ensuring model accountability and transparency by pairing top-tier gradient boosting frameworks with mathematical fairness metrics and SHAP explainability.

---

##  Robust Machine Learning Pipeline Blueprint

```
[Raw Dataset] -> [Chronological Tenure Extractor] -> [Hierarchical Job Parser]
                                                              │
 [MLflow tracking] <- [Leakage-Safe nested 5-Fold CV] <- [Out-Of-Fold Target Encoder]
        │
 [XGBoost Champion Model] -> [Global & Local SHAP Interpretability] -> [Algorithmic Fairness Audit]
```

### 1. Feature Engineering & Leakage Protection
* **Chronological Tenure Analysis**: Computed empirical worker tenure relative to a standardized epoch time to capture job experience without historical drift.
* **Hierarchical Job Title Extraction**: Parsed standard civil service titles using regular expression tokenizers to separate base professional categories from alphanumeric seniority markers (`I` through `VIII`, `NA`).
* **Out-Of-Fold (OOF) Target Encoding**: To process high-cardinality fields (`AGENCY NAME`, `base_title`), we constructed a leakage-safe target encoder running within our nested cross-validation folds. This guarantees zero data leakage from validation/testing subsets into training statistics by using a smoothed posterior probability format.

### 2. Multi-Model Performance Benchmarking
We executed a standardized, nested 5-Fold Cross-Validation evaluation matrix across 9 diverse algorithms to identify the optimal boundary between linear efficiency and ensemble predictive power.

---

##  Model Evaluation Summary

| Algorithm Category | Model | CV RMSE ($) | CV MAE ($) | CV $R^2$ | Test RMSE ($) | Test MAE ($) | Test $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ensemble Regressors** |  **XGBoost** | **7,383.05** | **3,676.54** | **0.9137** | **7,097.19** | **3,642.51** | **0.9186** |
| |  **LightGBM** | 7,380.52 | 3,702.08 | 0.9138 | 7,112.92 | 3,676.68 | 0.9182 
| |  **Random Forest** | 7,766.73 | 3,454.29 | 0.9046 | 7,549.96 | 3,476.88 | 0.9079 |
| |  **Gradient Boosting** | 8,870.30 | 4,561.24 | 0.8755 | 8,504.84 | 4,509.35 | 0.8831 |
| |  **CatBoost** | 9,726.91 | 4,162.06 | 0.8504 | 9,266.56 | 4,060.56 | 0.8612 |
| **Decision Trees** |  **Decision Tree** | 10,467.27 | 4,421.83 | 0.8268 | 9,769.69 | 4,366.88 | 0.8458 |
| **Linear Regressors** |  **Lasso Regression** | 10,276.20 | 5,231.13 | 0.8330 | 9,892.27 | 5,161.72 | 0.8419 |
| |  **Ridge Regression** | 10,274.93 | 5,249.10 | 0.8331 | 9,900.41 | 5,183.31 | 0.8416 |
| |  **Linear Regression** | 10,274.87 | 5,249.09 | 0.8331 | 9,900.36 | 5,183.31 | 0.8416 |

---

##  Responsible AI: Interpretability & Fairness Audit

To ensure our model acts as a trustable advisor rather than a predictive "black box", we implemented post-hoc interpretability models and fairness criteria:

* **Explainable AI (XAI)**: Leveraging TreeSHAP (Shapley Additive Explanations), we mathematically verified feature contribution. Professional features (such as parsed `base_title_TE` and seniority `level`) drive **98.50%** of prediction weight. In comparison, direct demographic proxies (Gender and Ethnicity) account for only **1.50%** of cumulative attribution.
* **Ethical AI Auditing**: We mathematically audited prediction residuals across demographic subdivisions to verify group-level error bounds:
  * **Gender Equity Audit**: Minor performance discrepancy (MAE Gap: **$391.79**; Bias Gap: **$187.24**).
  * **Ethnic Equity Audit**: Identified systemic accuracy variance (MAE Gap: **$3,285.30**; Bias Gap: **$911.05**), with performance gaps closely correlating to highly skewed class distributions inside rare demographic categories.

---

