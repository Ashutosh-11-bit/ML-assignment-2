import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="ML Classification Model Lab",
    page_icon="🧪",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

MODEL_FILES = {
    "Logistic Regression": ROOT / "model" / "logistic_regression.joblib",
    "Decision Tree": ROOT / "model" / "decision_tree.joblib",
    "KNN": ROOT / "model" / "knn.joblib",
    "Naive Bayes": ROOT / "model" / "naive_bayes.joblib",
    "Support Vector Machine": ROOT / "model" / "support_vector_machine.joblib",
}

METRIC_COLS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.title("🧪 Machine Learning Classification Model Lab")
st.caption("Five supervised classifiers evaluated on the uploaded test data.")

with st.sidebar:
    st.header("Controls")
    selected_model = st.selectbox("Select a model", list(MODEL_FILES.keys()))
    uploaded = st.file_uploader(
        "Upload test CSV",
        type=["csv"],
        help="Upload a CSV containing the 30 input features and the target column."
    )
    st.info("The app loads saved models. It does not retrain on uploaded test data.")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_metrics():
    return pd.read_csv(ROOT / "metrics.csv")

def validate_dataset(data):
    model = load_model(MODEL_FILES[selected_model])
    expected = list(getattr(model, "feature_names_in_", []))
    if not expected:
        # Pipeline stores feature_names_in_ on the pipeline.
        expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in data.columns]
    if "target" not in data.columns:
        missing.append("target")
    return expected, missing

if uploaded is None:
    st.subheader("How to use")
    st.markdown(
        """
        1. Upload the provided `test_data.csv`.
        2. Choose one of the five saved classifiers.
        3. View accuracy, AUC, precision, recall, F1 and MCC.
        4. Inspect the confusion matrix and classification report.
        """
    )
    st.subheader("Reference test-set comparison")
    comparison = load_metrics().copy()
    comparison[METRIC_COLS] = comparison[METRIC_COLS].round(4)
    st.dataframe(comparison, use_container_width=True)
    st.stop()

data = pd.read_csv(uploaded)
expected_features, missing = validate_dataset(data)

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

X = data[expected_features]
y = data["target"]

model = load_model(MODEL_FILES[selected_model])
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y, pred),
    "AUC": roc_auc_score(y, prob),
    "Precision": precision_score(y, pred, zero_division=0),
    "Recall": recall_score(y, pred, zero_division=0),
    "F1": f1_score(y, pred, zero_division=0),
    "MCC": matthews_corrcoef(y, pred),
}

st.subheader(f"Results — {selected_model}")
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual malignant", "Actual benign"],
        columns=["Predicted malignant", "Predicted benign"]
    )
    st.dataframe(cm_df, use_container_width=True)

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y, pred, target_names=["malignant", "benign"],
        output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, use_container_width=True)

st.subheader("Prediction Preview")
preview = data.copy()
preview["predicted_target"] = pred
preview["predicted_probability_benign"] = prob
st.dataframe(preview.head(20), use_container_width=True)

st.download_button(
    "Download predictions CSV",
    preview.to_csv(index=False),
    file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
    mime="text/csv"
)
