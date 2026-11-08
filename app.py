
import os
import html
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Dry Bean Classification Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. CUSTOM UI STYLING
# ==========================================================

st.markdown("""
<style>

/* Main page layout */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Hide heading anchor icons */
[data-testid="stHeaderActionElements"],
.stHeadingActionElements {
    display: none !important;
}

/* Hero section */
.hero {
    padding: 30px;
    border-radius: 16px;
    background: linear-gradient(120deg, #12372A, #24734D);
    color: white;
    margin-bottom: 24px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #E5F5EA;
    font-size: 15px;
    margin-bottom: 0;
}

/* Section descriptions */
.section-note {
    color: #718096;
    font-size: 14px;
    margin-top: -8px;
    margin-bottom: 18px;
}

/* ========================================================
   METRIC CARDS
   ======================================================== */

div[data-testid="stMetric"] {
    background: #F5F8F6 !important;
    border: 1px solid #E0EAE3;
    padding: 15px;
    border-radius: 12px;
}

/* Force metric heading visibility across Safari and Chrome */
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] span {
    color: #52665A !important;
    -webkit-text-fill-color: #52665A !important;
    opacity: 1 !important;
}

/* Metric values */
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
    color: #174D34 !important;
    -webkit-text-fill-color: #174D34 !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    border: 1px dashed #80A991;
    border-radius: 12px;
    padding: 12px;
}

/* Dataset status banner */
.dataset-status {
    padding: 12px 16px;
    border-radius: 10px;
    margin: 12px 0 18px 0;
    border: 1px solid #B8D9C2;
    background: #EAF5EE;
    color: #174D34;
    line-height: 1.6;
}

.dataset-status strong {
    color: #174D34;
}

.dataset-status.uploaded {
    background: #EDF4FF;
    border-color: #B8CFF2;
    color: #244A7C;
}

.dataset-status.uploaded strong {
    color: #244A7C;
}

/* ========================================================
   SELECTBOX RED BORDER FIX
   ======================================================== */

div[data-testid="stSelectbox"] div[data-focus-within="true"] {
    border-color: #80A991 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Normal dropdown border */
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border-color: #C7D6CC !important;
    box-shadow: none !important;
}

/* Focused dropdown border */
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border-color: #80A991 !important;
    box-shadow: none !important;
}

/* Multiselect styling */
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    border-color: #C7D6CC !important;
    box-shadow: none !important;
}

div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
    border-color: #80A991 !important;
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. PROJECT PATHS AND CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "test_data.csv"

FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRatio",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "Roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]

TARGET = "Class"

REQUIRED_COLUMNS = FEATURES + [TARGET]

MODEL_PATHS = {
    "Logistic Regression":
        BASE_DIR / "model" / "logistic_regression.joblib",

    "Decision Tree Classifier":
        BASE_DIR / "model" / "decision_tree.joblib",

    "K-Nearest Neighbor Classifier":
        BASE_DIR / "model" / "knn.joblib",

    "Naive Bayes Classifier - Gaussian":
        BASE_DIR / "model" / "naive_bayes.joblib",

    "Support Vector Machine":
        BASE_DIR / "model" / "svm.joblib"
}


# ==========================================================
# 4. LOAD SAVED MODELS
# ==========================================================

@st.cache_resource
def load_model(model_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


@st.cache_resource
def load_all_models():

    loaded_models = {}

    for model_name, model_path in MODEL_PATHS.items():

        loaded_models[model_name] = load_model(
            str(model_path)
        )

    return loaded_models


# ==========================================================
# 5. LOAD DEFAULT TEST DATASET
# ==========================================================

@st.cache_data
def load_default_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "The default test_data.csv file was not found."
        )

    return pd.read_csv(DATA_PATH)


# ==========================================================
# 6. VALIDATE TEST DATASET
# ==========================================================

def validate_dataset(data):

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("The dataset is empty.")

    data = data[REQUIRED_COLUMNS].copy()

    for column in FEATURES:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    if data[FEATURES].isnull().any().any():
        raise ValueError(
            "The feature columns contain missing or non-numeric values."
        )

    if not np.isfinite(data[FEATURES].to_numpy()).all():
        raise ValueError(
            "The feature columns contain infinite values."
        )

    if data[TARGET].isnull().any():
        raise ValueError(
            "The target column contains missing values."
        )

    data[TARGET] = data[TARGET].astype(str)

    return data


# ==========================================================
# 7. CALCULATE EVALUATION METRICS
# ==========================================================

def calculate_metrics(model, X, y_actual):

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    model_classes = model.classes_

    if set(y_actual.unique()) != set(model_classes):
        raise ValueError(
            "The test dataset must contain all seven bean classes "
            "to calculate the multiclass AUC score consistently."
        )

    metrics = {
        "Accuracy": accuracy_score(y_actual, y_pred),

        "AUC": roc_auc_score(
            y_actual,
            y_proba,
            multi_class="ovr",
            average="weighted",
            labels=model_classes
        ),

        "Precision": precision_score(
            y_actual,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_actual,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "F1": f1_score(
            y_actual,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_actual,
            y_pred
        )
    }

    return metrics, y_pred


# ==========================================================
# 8. EVALUATE ALL FIVE MODELS
# ==========================================================

@st.cache_data(show_spinner=False)
def evaluate_all_models(test_data):

    X = test_data[FEATURES]
    y_actual = test_data[TARGET]

    loaded_models = load_all_models()

    results = []

    for model_name, model in loaded_models.items():

        metrics, _ = calculate_metrics(
            model,
            X,
            y_actual
        )

        results.append({
            "ML Model Name": model_name,
            **metrics
        })

    return pd.DataFrame(results)


# ==========================================================
# 9. APPLICATION HEADER
# ==========================================================

st.markdown("""
<div class="hero">
    <h1>🌱 Dry Bean Classification</h1>
    <p>
        Explore five machine learning classifiers, compare their
        performance and evaluate dry bean varieties using an
        interactive classification dashboard.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# 10. SIDEBAR
# ==========================================================

st.sidebar.title("🌿 Project Overview")

st.sidebar.markdown("""
**Dataset:** UCI Dry Bean Dataset

**Problem:** Multi-class Classification

**Input Features:** 16

**Target Classes:** 7

**ML Models:** 5

**Evaluation Metrics:** 6
""")

st.sidebar.divider()

st.sidebar.subheader("Available Models")

for model_name in MODEL_PATHS:
    st.sidebar.write(f"• {model_name}")


# ==========================================================
# 11. TEST DATASET
# ==========================================================

st.header("📂 Test Dataset")

st.markdown(
    '<p class="section-note">'
    'The application automatically loads the bundled test_data.csv file. '
    'You can also upload any compatible CSV test dataset.'
    '</p>',
    unsafe_allow_html=True
)


# ==========================================================
# 12. DATASET OVERVIEW AND DOWNLOAD
# ==========================================================

# Keep upload and download controls in the same row.

upload_col, download_col = st.columns([1.35, 1], gap="medium")

with upload_col:

    uploaded_file = st.file_uploader(
        "Upload a test dataset (optional)",
        type=["csv"],
        help=(
            "Upload any CSV file containing "
            "all 16 input features and the Class target column."
        ),
        key="test_dataset_upload"
    )


# Load and validate the selected dataset.

try:

    if uploaded_file is not None:

        raw_data = pd.read_csv(uploaded_file)
        test_data = validate_dataset(raw_data)

        dataset_source = "Uploaded test dataset"
        dataset_filename = uploaded_file.name
        dataset_size_kb = uploaded_file.size / 1024

    else:

        raw_data = load_default_dataset()
        test_data = validate_dataset(raw_data)

        dataset_source = "Default UCI Dry Bean test dataset"
        dataset_filename = "test_data.csv"
        dataset_size_kb = DATA_PATH.stat().st_size / 1024

except Exception as error:

    st.error(f"Unable to load the test dataset: {error}")
    st.stop()


with download_col:

    st.caption("Download the current test dataset")

    st.download_button(
        label="⬇️ Download Sample test_data.csv",
        data=test_data.to_csv(index=False).encode("utf-8"),
        file_name="test_data.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_current_test_dataset"
    )


# Show a status message that changes when a new dataset is uploaded.

if uploaded_file is not None:

    safe_filename = html.escape(dataset_filename)

    st.markdown(
        f"""
        <div class="dataset-status uploaded">
            <strong>✓ Test dataset uploaded successfully</strong><br>
            File: <strong>{safe_filename}</strong><br>
            The dashboard is now using your uploaded dataset to
            evaluate all five saved models.
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="dataset-status">
            <strong>✓ Default test dataset loaded</strong><br>
            Dataset: <strong>test_data.csv</strong><br>
            Upload another compatible CSV file
            to evaluate the models on your own test data.
        </div>
        """,
        unsafe_allow_html=True
    )


# Dataset metrics update whenever the selected dataset changes.

row_count = len(test_data)
feature_count = len(FEATURES)
class_count = test_data[TARGET].nunique()

info1, info2, info3, info4 = st.columns(4)

info1.metric(
    "Test Records",
    f"{row_count:,}"
)

info2.metric(
    "Input Features",
    feature_count
)

info3.metric(
    "Bean Classes",
    class_count
)

info4.metric(
    "Dataset Size",
    f"{dataset_size_kb:.1f} KB"
)


st.caption(
    f"Current dataset: **{dataset_source}**"
)


# Dataset preview

with st.expander("👀 Preview Test Dataset"):

    st.dataframe(
        test_data.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Showing the first 10 records from {row_count:,} "
        "total records, containing 16 numerical features "
        "and the target column Class."
    )


# ==========================================================
# 13. LOAD MODELS AND EVALUATE DATASET
# ==========================================================

try:

    with st.spinner(
        "Loading saved models and evaluating performance..."
    ):

        loaded_models = load_all_models()

        comparison_df = evaluate_all_models(test_data)

except Exception as error:

    st.error(f"Model evaluation failed: {error}")
    st.stop()


# ==========================================================
# 14. EXPLORE AN INDIVIDUAL MODEL
# ==========================================================

st.divider()

st.header("🔍 Explore an Individual Model")

st.markdown(
    '<p class="section-note">'
    'Select a saved classifier to view its evaluation metrics, '
    'confusion matrix and classification report on the current test dataset.'
    '</p>',
    unsafe_allow_html=True
)

selected_model_name = st.selectbox(
    "Choose a classification model",
    list(MODEL_PATHS.keys()),
    key="individual_model_select"
)

selected_model = loaded_models[selected_model_name]

X_test = test_data[FEATURES]
y_actual = test_data[TARGET]

try:

    selected_metrics, y_pred = calculate_metrics(
        selected_model,
        X_test,
        y_actual
    )

except Exception as error:

    st.error(f"Unable to evaluate selected model: {error}")
    st.stop()


# ==========================================================
# 15. SELECTED MODEL EVALUATION METRICS
# ==========================================================

st.subheader(f"Performance: {selected_model_name}")

metric_names = [
    ("Accuracy", "Accuracy"),
    ("AUC", "AUC Score"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("F1", "F1 Score"),
    ("MCC", "MCC Score")
]

metric_cols = st.columns(3)

for index, (key, label) in enumerate(metric_names):

    metric_cols[index % 3].metric(
        label,
        f"{selected_metrics[key]:.4f}"
    )


# ==========================================================
# 16. CONFUSION MATRIX
# ==========================================================

st.subheader("🎯 Confusion Matrix")

class_labels = selected_model.classes_

cm = confusion_matrix(
    y_actual,
    y_pred,
    labels=class_labels
)

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    xticklabels=class_labels,
    yticklabels=class_labels,
    linewidths=0.5,
    linecolor="white",
    ax=ax
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title(f"Confusion Matrix – {selected_model_name}")

plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ==========================================================
# 17. CLASSIFICATION REPORT
# ==========================================================

st.subheader("📋 Classification Report")

report = classification_report(
    y_actual,
    y_pred,
    labels=class_labels,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df = report_df.drop(
    index="accuracy",
    errors="ignore"
)

report_df.index.name = "Class"
report_df = report_df.reset_index()

report_df = report_df.rename(
    columns={
        "precision": "Precision",
        "recall": "Recall",
        "f1-score": "F1-score",
        "support": "Support"
    }
)

st.dataframe(
    report_df.style.format({
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-score": "{:.4f}",
        "Support": "{:.0f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ==========================================================
# 18. MODEL PERFORMANCE OVERVIEW
# ==========================================================

st.divider()

st.header("🏆 Model Performance Overview")

st.markdown(
    '<p class="section-note">'
    'Compare all five saved classification models using the six '
    'evaluation metrics on the same test dataset.'
    '</p>',
    unsafe_allow_html=True
)

best_accuracy_row = comparison_df.loc[
    comparison_df["Accuracy"].idxmax()
]

best_auc_row = comparison_df.loc[
    comparison_df["AUC"].idxmax()
]

winner_col1, winner_col2, winner_col3 = st.columns(3)

winner_col1.metric(
    "Best Accuracy",
    f"{best_accuracy_row['Accuracy']:.4f}",
    best_accuracy_row["ML Model Name"]
)

winner_col2.metric(
    "Best AUC",
    f"{best_auc_row['AUC']:.4f}",
    best_auc_row["ML Model Name"]
)

winner_col3.metric(
    "Models Evaluated",
    len(comparison_df)
)


# ==========================================================
# 19. MODEL COMPARISON TABLE
# ==========================================================

st.subheader("📊 Evaluation Metrics Comparison")

metric_columns = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]

styled_df = (
    comparison_df.style
    .format({
        column: "{:.4f}"
        for column in metric_columns
    })
    .highlight_max(
        subset=metric_columns,
        props="background-color: #E1EDF8; color: #17365D;"
    )
)

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True
)


# ==========================================================
# 20. MODEL PERFORMANCE VISUALIZATION
# ==========================================================

st.divider()

st.header("📈 Model Performance Visualization")

chart_tab1, chart_tab2, chart_tab3 = st.tabs([
    "Metric Comparison",
    "Accuracy Ranking",
    "Dataset Distribution"
])


# Metric Comparison

with chart_tab1:

    st.subheader("Compare Evaluation Metrics")

    selected_chart_metrics = st.multiselect(
        "Select metrics to visualize",
        metric_columns,
        default=["Accuracy", "AUC", "F1", "MCC"],
        key="comparison_metric_select"
    )

    if selected_chart_metrics:

        chart_data = comparison_df.set_index(
            "ML Model Name"
        )[selected_chart_metrics]

        fig, ax = plt.subplots(figsize=(11, 5))

        chart_data.plot(
            kind="bar",
            ax=ax,
            width=0.8
        )

        ax.set_ylabel("Metric Score")
        ax.set_xlabel("Classification Model")
        ax.set_ylim(0, 1.05)
        ax.set_title("Evaluation Metrics Across Five Models")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Metrics")

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    else:

        st.info(
            "Select at least one metric to display the chart."
        )


# Accuracy Ranking

with chart_tab2:

    st.subheader("Model Accuracy Ranking")

    ranking_df = comparison_df.sort_values(
        by="Accuracy",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        ranking_df["ML Model Name"],
        ranking_df["Accuracy"]
    )

    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.05)
    ax.set_title("Classification Model Accuracy")

    for index, value in enumerate(ranking_df["Accuracy"]):

        ax.text(
            value + 0.005,
            index,
            f"{value:.4f}",
            va="center"
        )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# Dataset Distribution

with chart_tab3:

    st.subheader("Dry Bean Class Distribution")

    class_distribution = (
        test_data[TARGET]
        .value_counts()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=class_distribution.values,
        y=class_distribution.index,
        ax=ax
    )

    ax.set_xlabel("Number of Test Records")
    ax.set_ylabel("Bean Class")
    ax.set_title("Distribution of Dry Bean Varieties")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ==========================================================
# 21. PREDICTION SUMMARY
# ==========================================================

st.divider()

st.header("📌 Prediction Summary")

correct_predictions = int(
    (y_actual.to_numpy() == y_pred).sum()
)

incorrect_predictions = (
    len(y_actual) - correct_predictions
)

summary1, summary2, summary3 = st.columns(3)

summary1.metric(
    "Total Predictions",
    f"{len(y_actual):,}"
)

summary2.metric(
    "Correct Predictions",
    f"{correct_predictions:,}"
)

summary3.metric(
    "Incorrect Predictions",
    f"{incorrect_predictions:,}"
)


# ==========================================================
# 22. MODEL INSIGHTS
# ==========================================================

st.divider()

st.header("💡 Model Insights")

best_accuracy_model = comparison_df.loc[
    comparison_df["Accuracy"].idxmax(),
    "ML Model Name"
]

best_auc_model = comparison_df.loc[
    comparison_df["AUC"].idxmax(),
    "ML Model Name"
]

lowest_accuracy_model = comparison_df.loc[
    comparison_df["Accuracy"].idxmin(),
    "ML Model Name"
]

st.markdown(f"""

- 🏆 **Best accuracy:** {best_accuracy_model}

- 🎯 **Best AUC:** {best_auc_model}

- 📉 **Lowest accuracy:** {lowest_accuracy_model}

""")


# ==========================================================
# 23. FOOTER
# ==========================================================

st.divider()

st.markdown("""
<div style="text-align:center; color:#718096; padding:15px;">
    <b>Dry Bean Classification Dashboard</b><br>
    BITS Pilani | M.Tech AIML | Machine Learning Assignment 2<br>
    Five classifiers • Six evaluation metrics • Interactive analysis
</div>
""", unsafe_allow_html=True)
