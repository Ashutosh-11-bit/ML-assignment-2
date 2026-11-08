# Machine Learning — Assignment 2
## Complete Reference Solution

**Marks:** 15  
**Submission deadline:** 08-Nov-2026

### 1. Dataset choice

The selected dataset is the **Breast Cancer Wisconsin (Diagnostic)** dataset from the UCI Machine Learning Repository. It contains **569 instances and 30 predictor features**, satisfying the assignment's minimum requirements of 500 instances and 12 features. The target is binary: malignant (0) and benign (1).

### 2. Train/test design

An 80:20 stratified train/test split is used with `random_state=42`. Stratification preserves the class distribution in both sets.

### 3. Models implemented

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine with RBF kernel

Scaling is used for Logistic Regression, KNN and SVM. Decision Tree and Gaussian Naive Bayes are used without a separate scaler.

### 4. Required evaluation metrics

For binary classification:

- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × Precision × Recall / (Precision + Recall)
- AUC = area under the ROC curve, calculated from predicted probabilities
- MCC = (TP×TN − FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]

### 5. Reference results

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |\n| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |\n| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |\n| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |\n| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |\n
### 6. Model-wise observations

**Logistic Regression:** A strong baseline because the transformed feature space permits a linear decision boundary. Standardization prevents large-scale variables from dominating the optimization.

**Decision Tree:** Learns nonlinear if/then splits and is easy to explain. Limiting depth to 5 reduces the risk of an unnecessarily complex tree.

**KNN:** Predicts from nearby training observations. Standardization is important because Euclidean distance is otherwise dominated by variables with larger numeric ranges.

**Naive Bayes:** Gaussian Naive Bayes is computationally light and provides a useful probabilistic baseline. Its conditional-independence assumption is restrictive for correlated medical measurements.

**Support Vector Machine:** The RBF kernel can represent nonlinear class boundaries. Standardization is important for the geometry of the kernel.

### 7. Overall winner

For this reference split, **Logistic Regression** gives the strongest combined performance when considering F1, AUC and MCC. Do not treat this as a universal winner: the conclusion is specific to the selected dataset, preprocessing and random split.

### 8. Streamlit application

The application implements all mandatory frontend features:

- test CSV upload
- model selection
- display of all six metrics
- confusion matrix
- classification report
- results using saved models without retraining on uploaded test data

### 9. GitHub repository

Use the project structure:

```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

The assignment also asks for a GitHub README, test CSV and saved model artifacts. The provided reference project contains these files.

### 10. Deployment procedure

1. Upload the project to a GitHub repository.
2. Verify `requirements.txt` and the `model/` directory are present.
3. Open Streamlit Community Cloud.
4. Create a new app from the GitHub repository.
5. Select branch `main` and entry point `app.py`.
6. Deploy.
7. Test the live URL by uploading `test_data.csv` and switching among all five models.

### 11. Required PDF submission order

The assignment requires a single PDF containing, in order:

1. GitHub repository link
2. Live Streamlit app link
3. BITS Virtual Lab execution screenshot
4. README content

The two URLs and the screenshot cannot be fabricated in this reference solution because they depend on the student's own GitHub deployment and BITS Lab execution. Replace the placeholders in the final submission with the actual links and your own screenshot.
