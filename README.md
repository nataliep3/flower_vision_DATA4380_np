![](UTA-DataScience-Logo.png)

# Software Defect Prediction (Kaggle Playground S3E23)

**This repository contains a solution for the Kaggle challenge: [Playground Series - Season 3, Episode 23](https://www.kaggle.com/competitions/playground-series-s3e23), which involves predicting software defects using code metric features.**

---

## Overview

The challenge is to build a binary classification model to predict whether a software file contains a defect (`1`) or not (`0`) using a range of numerical features extracted from code complexity and structure.

Our approach involved exploring and comparing several machine learning models including:
- Logistic Regression (baseline)
- Random Forest (primary model)
- Class balancing
- Feature transformation (log-scaling + standardization)

Our best model — a Random Forest Classifier with class weighting and calibrated preprocessing — achieved a **validation accuracy of 81%** and a **log loss of 0.46**. These results represent a solid baseline and demonstrate the importance of preprocessing and class imbalance strategies.

---

## Summary of Workdone

### Data

- **Type**: Tabular data in `.csv` format.
  - Input: Numeric software metrics (e.g., lines of code, branching, operands)
  - Output: Binary target `defects` column
- **Size**:
  - Training set: 100,000+ rows
  - Test set: 68,000+ rows
- **Split**:
  - 60% Training
  - 20% Validation
  - 20% Test (local)

#### Preprocessing / Clean up

- Dropped clearly uninformative or noisy features (`locCodeAndComment`, `IOBlank`, etc.)
- Applied `np.log1p()` to reduce skewness
- Standardized features using `StandardScaler`
- Handled class imbalance using `class_weight='balanced'` and threshold tuning

#### Data Visualization

- Histograms for each feature by class (`defects = 0` vs `defects = 1`)
- Distribution comparison before and after scaling
- Feature correlation analysis

---

### Problem Formulation

- **Input**: 1D vector of numeric code metrics
- **Output**: Binary label — `defects`: 1 = defective, 0 = clean
- **Model**: RandomForestClassifier (`sklearn`) with calibrated probabilities
- **Loss Function**: `log_loss` (Kaggle metric)
- **Hyperparameters**:
  - `n_estimators = 200`
  - `max_depth = None`
  - `class_weight = 'balanced'`

---

### Training

- Trained using `scikit-learn` on a standard CPU machine (8-core, 16 GB RAM)
- Training time: ~2 minute
- Early stopping not applicable; Random Forest is non-iterative
- Trained and validated on fixed splits with `random_state=42`

---

### Performance Comparison

| Model                        | Accuracy | Log Loss |
|------------------------------|----------|----------|
| Logistic Regression          | 75%      | 0.55     |
| Random Forest (basic)        | 81%      | 0.46     |
| Random Forest (tuned attempt)| 81%      | 0.45     |

---

### Conclusions

- Random Forests performed best with minimal tuning
- Preprocessing (log-scaling and feature selection) made a large impact
- Threshold tuning significantly improved defect recall
- Class weighting helped balance metrics on imbalanced data

---

### Future Work

- Try XGBoost or LightGBM with tuned parameters
- Use SHAP values for better interpretability
- Test ensemble of models to improve log loss
- Explore feature engineering based on domain knowledge

---
