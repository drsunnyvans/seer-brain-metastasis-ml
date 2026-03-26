# Prediction of Brain Metastasis in Lung Cancer Using Machine Learning

This repository contains R code used to evaluate multiple machine learning algorithms for predicting brain metastasis at diagnosis in patients with lung cancer using SEER data.

## Data Source
The study utilized data from the SEER database. The dataset is publicly available; however, the analytic dataset used in this study is not included in this repository. The manuscript details the extraction and preprocessing steps required to construct the analytic dataset.

## Data Processing Notes

The dataset was derived from the SEER database.

- Outcome variable: "SEER Combined Mets at DX-brain (2010+)" (binary classification)
- Categorical variables were one-hot encoded
- Variable names were standardized using R's `make.names()` function
- Records with missing values were excluded prior to analysis
- Class imbalance was addressed using SMOTE

## Included Code
- Data preprocessing and feature engineering
- Class balancing using SMOTE
- Model training and testing with repeated train/test splits
- Algorithms evaluated: GLMnet, kNN, ANN, Random Forest, XGBoost, Naïve Bayes, SVM
- Model evaluation using sensitivity, specificity, precision, F1 score, balanced accuracy, and ROC/AUC
- Ensemble modeling
- SHAP-based model interpretation

## Reproducibility

To reproduce the analysis:

1. Obtain access to the SEER dataset.
2. Construct the analytic dataset as described in the manuscript.
3. Install required R packages (see manuscript for details).
4. Place the analytic dataset in the `/data` directory as `your_dataset.csv` (or `.xlsx`, depending on script).
5. Run scripts in sequence as provided.

## Notes

- Patient-level data are not included due to data-use considerations.
- SEER data can be accessed at: https://seer.cancer.gov/

## Code Availability

All code used for data processing, model development, and evaluation is publicly available in this repository to support reproducibility and independent validation of results.
