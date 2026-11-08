# Machine Learning Assignment – 2

## a. Problem statement

Implement five classification models on one public classification dataset, evaluate them using six required metrics, and demonstrate the saved models through an interactive Streamlit application.

## b. Dataset description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository (also available through `sklearn.datasets`)  
**Instances:** 569  
**Predictor features:** 30  
**Target:** `target` — 0 = malignant, 1 = benign  
**Train/test split:** 80/20 stratified split, `random_state=42`

The dataset satisfies the assignment minimum of 12 features and 500 instances.

## c. Github Repository Link

Replace this placeholder with the actual repository URL after creating the GitHub repository:

`https://github.com/<your-username>/ml-assignment-2-classification`

## d. Models used and comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |\n| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |\n| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |\n| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |\n| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |\n
### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a strong linear baseline. Standardization is used because the feature scales differ substantially. |
| Decision Tree | Captures nonlinear decision boundaries and is easy to interpret, but its performance depends on tree depth and can overfit if left unconstrained. |
| KNN | Uses local neighbourhood structure. Feature scaling is important because distance calculations are sensitive to feature magnitude. |
| Naive Bayes | Fast probabilistic baseline. Gaussian assumptions make it simple and computationally efficient, though the independence assumption can limit performance when features are correlated. |
| Support Vector Machine | The RBF kernel can model nonlinear boundaries. Scaling is essential, and the resulting model gives a strong overall classifier for this dataset. |

### Overall winner for this dataset

Based on the reference test split, **Logistic Regression** is the strongest overall model using the combined F1/AUC/MCC comparison. The exact ranking should be reproduced by running the supplied code in the assignment environment.

## Repository structure

```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── metrics.csv
├── breast_cancer_wisconsin_diagnostic.csv
├── DATASET_INFO.txt
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

## Streamlit features

- CSV test-data upload
- Model-selection dropdown
- Accuracy, AUC, precision, recall, F1 and MCC
- Confusion matrix
- Classification report
- Prediction preview
- Downloadable prediction CSV
- Saved models are loaded; the uploaded test set is not used for retraining

## Deployment

1. Create a GitHub repository.
2. Upload the project files and `model/` directory.
3. Open Streamlit Community Cloud.
4. Sign in with GitHub.
5. Select **New App**.
6. Choose the repository and `main` branch.
7. Select `app.py`.
8. Deploy.
9. Open the generated app URL and verify the CSV upload and model results.

## Academic-integrity note

This repository is a reference implementation prepared from the assignment requirements. Before submission, run it yourself in BITS Virtual Lab, verify the outputs, create your own GitHub commit history, customize the UI/README wording where appropriate, and include your own BITS Lab execution screenshot.
