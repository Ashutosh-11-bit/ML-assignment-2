# Machine Learning Assignment – 2

## a. Problem statement

Implement five classification models on one public classification dataset, evaluate them using the six required metrics (Accuracy, AUC, Precision, Recall, F1 Score, MCC), and demonstrate the saved models through an interactive Streamlit web application. The models must be trained once, saved, and then loaded in the Streamlit app so that evaluation is performed on uploaded test data without retraining.

## b. Dataset description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository (also available via `sklearn.datasets`)  
**Instances:** 569  
**Predictor features:** 30  
**Target:** `target` — 0 = malignant, 1 = benign  
**Train/test split:** 80/20 stratified split with `random_state=42`  
**Train instances:** 455  
**Test instances:** 114  

The dataset satisfies the assignment minimum requirements of at least 12 features and 500 instances.

## c. Github Repository Link

Replace this placeholder with your actual repository URL after creating the GitHub repository:

`https://github.com/<your-username>/ml-assignment-2-classification`

## d. Models used

The following five classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Support Vector Machine (RBF kernel)

### Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|--------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression      | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree            | 0.9211   | 0.9163 | 0.9565    | 0.9167 | 0.9362 | 0.8341 |
| KNN                      | 0.9737   | 0.9884 | 0.9600    | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes              | 0.9386   | 0.9878 | 0.9452    | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine   | 0.9825   | 0.9950 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |

### Observations on model performance

| ML Model Name            | Observation about model performance |
|--------------------------|-------------------------------------|
| Logistic Regression      | Strong linear baseline. Standardization is applied because feature scales differ substantially. Achieves high Accuracy, AUC, F1 and MCC. |
| Decision Tree            | Captures non-linear decision boundaries and is easy to interpret. Performance depends on tree depth; can overfit if left unconstrained. Lowest overall scores among the five models. |
| KNN                      | Uses local neighbourhood structure. Feature scaling is important because distance calculations are sensitive to feature magnitude. Perfect Recall (1.0) on the test set. |
| Naive Bayes              | Fast probabilistic baseline (Gaussian). Computationally efficient, but the independence assumption can limit performance when features are correlated. |
| Support Vector Machine   | RBF kernel models non-linear boundaries. Scaling is essential. Delivers performance almost identical to Logistic Regression and is among the top two models. |

**Overall Winner for this dataset:**  
Logistic Regression (and Support Vector Machine jointly) — highest Accuracy (0.9825), AUC (0.9954 / 0.9950), F1 (0.9861) and MCC (0.9623). Logistic Regression is selected as the overall winner due to its slightly higher AUC while remaining simple and interpretable.

---

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
├── train_models.py
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

## Streamlit app features

- Dataset upload option (CSV) – upload only test data  
- Model selection dropdown  
- Display of all six evaluation metrics  
- Confusion matrix and classification report  
- Saved models are loaded; the uploaded test set is **not** used for retraining  

## Deployment steps (Streamlit Community Cloud)

1. Go to https://streamlit.io/cloud  
2. Sign in using your GitHub account  
3. Click “New App”  
4. Select your repository  
5. Choose branch (usually `main`)  
6. Select `app.py`  
7. Click Deploy  

## Academic integrity note

This is a reference implementation prepared from the assignment requirements. Before final submission:

- Run the complete pipeline yourself on BITS Virtual Lab  
- Create your own GitHub commit history  
- Customize the UI / wording where appropriate  
- Capture and include your own BITS Lab execution screenshot  
- Replace the GitHub and Streamlit links with your own live links  

Any plagiarism (identical structure, variable names, or outputs across submissions) will result in zero marks.