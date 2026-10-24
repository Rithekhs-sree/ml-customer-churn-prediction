# Predicting Customer Churn in Wireless Telecommunications Using Machine Learning

**IT5022 – Fundamentals of Machine Learning – Group Assignment**

| Member | Student ID | GitHub |
|---|---|---|
| Udani | MS26935140 | [uaSen](https://github.com/uaSen) |
| Rithekha | MS26923260 | [Rithekhs-sree](https://github.com/Rithekhs-sree) |

## Overview

Customer churn, where a subscriber ends their service, reduces a telecom provider's recurring revenue and increases the cost of acquiring new customers. This project uses supervised machine learning to predict which customers of a US wireless carrier are likely to churn, so that a retention team can focus on the customers at highest risk.

Two distinct, non-deep-learning algorithms are implemented, tuned and compared on the same data, folds and metrics:

1. **Logistic Regression**: an interpretable, L2-regularised linear model of the log-odds of churn.
2. **Random Forest**: a bagged ensemble of decision trees that can capture non-linear relationships and interactions.

A dummy classifier that gives every customer the same churn score serves as the no-skill baseline.

## Dataset

- **Dataset:** *telecom churn (cell2cell)*, Kaggle, version 2 (2018): <https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom>
- **Original source:** Teradata Center for Customer Relationship Management, Fuqua School of Business, Duke University (Cell2Cell churn-modelling tournament).
- **Variable documentation:** [Cell2Cell: The Churn Game – Database Documentation](https://github.com/leylatulu/Telecom-Churn-Analysis/blob/main/Cell2Cell-Data-Documentation.pdf)

| File | Customers | Columns | Use |
|---|---|---|---|
| `data/cell2celltrain.csv` | 51,047 | 58 | Labelled data (28.8% churn): split 80% development / 20% test |
| `data/cell2cellholdout.csv` | 20,000 | 58 | `Churn` is blank; used only to demonstrate predictions for new customers |

The 56 candidate predictors cover billing, usage, call quality, customer-care and retention contacts, tenure, handset and equipment, household, credit rating and service area. Three retention fields are excluded from the main models because their timing relative to churn cannot be verified, which leaves **53 predictors**.

## Method

- **Data audit and EDA:** schema, missing and `Unknown` values, duplicates, invalid values, outliers and class imbalance. Exploration uses development data only.
- **Leakage-safe preprocessing:** a scikit-learn `Pipeline`/`ColumnTransformer` with structural cleaning, median imputation with missingness indicators, one-hot encoding (rare `ServiceArea` codes grouped), and scaling for Logistic Regression only. Every learned step is refitted inside each training fold.
- **Validation:** 80/20 stratified development/test split and five-fold stratified cross-validation on the development data. The same folds are used for every model.
- **Experiments:** feature engineering and selection, hyperparameter tuning, class weighting versus threshold choice, and robustness to features of uncertain timing.
- **Final evaluation:** both final models are evaluated **once** on the 10,210 test customers, after every decision has been fixed.
- **Main metric:** Average Precision (area under the precision–recall curve), because churners are the minority class. ROC-AUC, precision, recall, F1, MCC, balanced accuracy, accuracy, top-10% precision and the Brier score are reported alongside it.

## Key Results (test set)

| Measure | Dummy baseline | Logistic Regression | Random Forest |
|---|---|---|---|
| Average Precision | 0.288 | 0.372 | **0.428** |
| ROC-AUC | 0.500 | 0.611 | **0.658** |
| F1 (at F1 threshold) | 0.000 | 0.467 | **0.489** |
| MCC (at F1 threshold) | 0.000 | 0.144 | **0.200** |
| Precision in top 10% of customers | 0.288 | 0.422 | **0.515** |

- **Random Forest performs better** on every main measure. Its test AP advantage of +0.057 has a 95% paired bootstrap interval of [+0.040, +0.070].
- The models are most useful as a **ranking tool**. Contacting the top 10% of Random Forest scores reaches churners at 1.8 times the base rate, whereas the F1-optimal threshold would flag about 60% of all customers.
- The most influential features are handset age, monthly usage, change in usage and tenure.
- Limitations and future work are discussed in Section 16 of the notebook.

## Repository Structure

```
ml-customer-churn-prediction/
├── data/
│   ├── cell2celltrain.csv
│   ├── cell2cellholdout.csv
│   └── Cell2Cell_Section15_Holdout_Predictions.csv   # written by Section 15
├── docs/
│   ├── members.txt        # member IDs and emails
│   ├── submission.txt     # dataset, repository and video links
│   └── requirements.txt   # pinned package versions
├── notebook/
│   └── Machine_Learning_Assignment.ipynb             # main notebook
└── README.md
```

## Notebook Contents

| Section | Topic |
|---|---|
| 0–1 | Environment, reproducibility and data loading |
| 2 | Dataset integrity audit and feature dictionary |
| 3 | Experimental design, data partitioning and metrics |
| 4 | Exploratory data analysis (development data only) |
| 5–6 | Leakage-safe preprocessing, feature engineering and feature analysis |
| 7 | Baseline model and shared evaluation code |
| 8–9 | Logistic Regression and Random Forest |
| 10–12 | Hyperparameter tuning, class imbalance and threshold choice, feature validity |
| 13 | Final evaluation on the test set |
| 14 | Model interpretation and error analysis |
| 15 | Predictions for the unlabelled holdout customers |
| 16 | Conclusions, limitations and future work |

## How to Run

Requires Python 3.13 (the package versions are pinned in `docs/requirements.txt`).

```bash
git clone https://github.com/Rithekhs-sree/ml-customer-churn-prediction.git
cd ml-customer-churn-prediction
pip install -r docs/requirements.txt
cd notebook
jupyter notebook Machine_Learning_Assignment.ipynb
```

Run all cells from top to bottom. The notebook reads the data from `../data`, so start Jupyter from the `notebook/` folder. A full run takes about 20–25 minutes on a laptop, mainly because of the Random Forest experiments. A fixed random seed (`RANDOM_STATE = 42`) makes every run give identical results. Section 15 writes the holdout predictions to `data/Cell2Cell_Section15_Holdout_Predictions.csv`.

## Individual Contributions

Each member developed and committed their own notebook sections; the commit history records who did what.

| Member | Notebook sections |
|---|---|
| **Udani** (MS26935140) | 0–2 data loading and integrity audit · 3.6–3.9 split, cross-validation and metrics · 4 exploratory data analysis · 7 baseline model · 8 Logistic Regression · 10.1–10.3 tuning plan and Logistic Regression tuning · 12 feature validity and robustness · 13.6–13.9 test curves, contact lists, bootstrap uncertainty and checks · 14.1–14.5 error analysis and calibration · 14.7 coefficients · 15 holdout predictions · 16.1–16.4, 16.6, 16.7 conclusions, algorithm comparison and limitations |
| **Rithekha** (MS26923260) | 3.1–3.5 prediction objective, target encoding and partitioning strategy · 5 leakage-safe preprocessing · 6 feature engineering and feature analysis · 9 Random Forest · 10.4–10.9 Random Forest tuning · 11 class imbalance and decision threshold · 13.1–13.5 final models and test results · 14.6 subgroup performance · 14.8–14.10 permutation importance, improvement experiments and missingness analysis · 16.5 decision rule · 16.8 future work |
| **Joint** | 16.9 conclusion, notebook review, report and presentation |
