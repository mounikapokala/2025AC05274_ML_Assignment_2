"""
ML Assignment 2 - Streamlit Web Application
Interactive classification model evaluation dashboard.

Features:
  a. Dataset upload option (CSV)
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ML Classification Dashboard")
st.markdown("### Wine Quality Binary Classification")
st.markdown(
    "Compare **5 ML models** on the Wine Quality dataset. "
    "Upload test data (CSV) and select a model to view predictions and evaluation metrics."
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}


@st.cache_resource
def load_model(model_name):
    """Load a saved model from disk."""
    model_path = os.path.join(MODEL_DIR, MODELS[model_name])
    return joblib.load(model_path)


@st.cache_resource
def load_scaler():
    """Load the saved scaler."""
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


@st.cache_resource
def load_feature_names():
    """Load the saved feature names."""
    return joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all 6 evaluation metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC Score": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

# a. Dataset upload option (CSV)
st.sidebar.subheader("📂 Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV test data file",
    type=["csv"],
    help="Upload a CSV file with the same features as the training data. Must include a 'target' column.",
)

# b. Model selection dropdown
st.sidebar.subheader("🧠 Select Model")
selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(MODELS.keys()),
)

compare_all = st.sidebar.checkbox("📊 Compare All Models", value=False)

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ Uploaded: {uploaded_file.name} ({data.shape[0]} rows)")
else:
    # Use default test data
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.csv")
    if os.path.exists(default_path):
        data = pd.read_csv(default_path)
        st.sidebar.info(f"ℹ️ Using default test_data.csv ({data.shape[0]} rows)")
    else:
        st.error("❌ No test data found. Please upload a CSV file or ensure test_data.csv exists.")
        st.stop()

# Validate data
if "target" not in data.columns:
    st.error("❌ CSV must contain a 'target' column (0 = Bad, 1 = Good).")
    st.stop()

feature_names = load_feature_names()
missing_cols = [c for c in feature_names if c not in data.columns]
if missing_cols:
    st.error(f"❌ Missing columns in uploaded CSV: {missing_cols}")
    st.stop()

X_test = data[feature_names]
y_test = data["target"]

# Scale features
scaler = load_scaler()
X_test_scaled = scaler.transform(X_test)

# ─────────────────────────────────────────────
# Dataset Preview
# ─────────────────────────────────────────────
with st.expander("📋 Dataset Preview", expanded=False):
    st.dataframe(data.head(20), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", data.shape[0])
    col2.metric("Features", len(feature_names))
    col3.metric("Target Balance", f"{y_test.mean():.1%} Good")

# ─────────────────────────────────────────────
# Compare All Models
# ─────────────────────────────────────────────
if compare_all:
    st.header("📊 All Models Comparison")

    all_results = []
    for model_name in MODELS:
        model = load_model(model_name)
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = model.decision_function(X_test_scaled)
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = model_name
        all_results.append(metrics)

    results_df = pd.DataFrame(all_results)
    results_df = results_df[["Model", "Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC"]]

    # Style the dataframe - highlight best values
    st.dataframe(
        results_df.style.highlight_max(
            subset=["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC"],
            color="#90EE90",
        ).format({
            "Accuracy": "{:.4f}",
            "AUC Score": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "MCC": "{:.4f}",
        }),
        use_container_width=True,
    )

    # Bar chart comparison
    st.subheader("📈 Visual Comparison")
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics_to_plot = ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC"]
    x = np.arange(len(metrics_to_plot))
    width = 0.15
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics_to_plot]
        ax.bar(x + i * width, values, width, label=row["Model"])
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metrics_to_plot, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison Across All Metrics")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

# ─────────────────────────────────────────────
# Selected Model Evaluation
# ─────────────────────────────────────────────
st.header(f"🔍 {selected_model} — Detailed Evaluation")

model = load_model(selected_model)
y_pred = model.predict(X_test_scaled)
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
else:
    y_prob = model.decision_function(X_test_scaled)

# c. Display of evaluation metrics
metrics = calculate_metrics(y_test, y_pred, y_prob)

col1, col2, col3 = st.columns(3)
col1.metric("🎯 Accuracy", f"{metrics['Accuracy']:.4f}")
col2.metric("📐 AUC Score", f"{metrics['AUC Score']:.4f}")
col3.metric("🔬 Precision", f"{metrics['Precision']:.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("📡 Recall", f"{metrics['Recall']:.4f}")
col5.metric("⚖️ F1 Score", f"{metrics['F1 Score']:.4f}")
col6.metric("📊 MCC", f"{metrics['MCC']:.4f}")

st.markdown("---")

# d. Confusion matrix and classification report
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🟦 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Bad (0)", "Good (1)"],
        yticklabels=["Bad (0)", "Good (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {selected_model}")
    plt.tight_layout()
    st.pyplot(fig)

with col_right:
    st.subheader("📝 Classification Report")
    report = classification_report(
        y_test, y_pred, target_names=["Bad (0)", "Good (1)"], output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "ML Assignment 2 | BITS Pilani — M.Tech (AIML/DSE) | Wine Quality Classification"
    "</div>",
    unsafe_allow_html=True,
)
