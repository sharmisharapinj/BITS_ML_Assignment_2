
import os
import streamlit as st


# --------------------------------------------------
# 1. Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)


# --------------------------------------------------
# 2. Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Hide heading anchor icons */
[data-testid="stHeaderActionElements"],
.stHeadingActionElements,
.stMarkdown h1 a,
.stMarkdown h2 a,
.stMarkdown h3 a,
.stMarkdown h4 a {
    display: none !important;
    visibility: hidden !important;
}

/* Remove the red focus border from the model dropdown */
div[data-testid="stSelectbox"] div[data-focus-within="true"] {
    border-color: rgba(49, 51, 63, 0.2) !important;
    box-shadow: none !important;
    outline: none !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. Application Title and Introduction
# --------------------------------------------------

st.title("🌱 Dry Bean Classification")

st.write(
    "This application implements five machine learning "
    "classification models using the Dry Bean dataset."
)

st.write(
    "Upload a test dataset containing the required input features "
    "and actual class labels to generate predictions and evaluate "
    "the selected classification model."
)


# --------------------------------------------------
# 4. Implementation of Classification Models
# --------------------------------------------------

st.header("Implementation of Classification Models")

st.write(
    "The following five machine learning classification models "
    "are implemented and evaluated using the same dataset."
)

st.markdown("""
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian
5. Support Vector Machine
""")


# --------------------------------------------------
# 5. Feature and Model Configuration
# --------------------------------------------------

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

MODEL_PATHS = {
    "1. Logistic Regression": "model/logistic_regression.joblib",
    "2. Decision Tree Classifier": "model/decision_tree.joblib",
    "3. K-Nearest Neighbor Classifier": "model/knn.joblib",
    "4. Naive Bayes Classifier - Gaussian": "model/naive_bayes.joblib",
    "5. Support Vector Machine": "model/svm.joblib"
}


# --------------------------------------------------
# 6. Load Trained Models
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):

    import joblib

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


# --------------------------------------------------
# 7. Load All Five Models Once
# --------------------------------------------------

@st.cache_resource
def load_all_models(model_paths):

    loaded_models = {}

    for model_name, model_path in model_paths:

        loaded_models[model_name] = load_model(model_path)

    return loaded_models


# --------------------------------------------------
# 8. Calculate Evaluation Metrics
# --------------------------------------------------

def calculate_metrics(model, X, y_actual):

    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef
    )

    y_pred = model.predict(X)

    y_proba = model.predict_proba(X)

    accuracy = accuracy_score(
        y_actual,
        y_pred
    )

    auc = roc_auc_score(
        y_actual,
        y_proba,
        multi_class="ovr",
        average="weighted",
        labels=model.classes_
    )

    precision = precision_score(
        y_actual,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_actual,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_actual,
        y_pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_actual,
        y_pred
    )

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    }


# --------------------------------------------------
# 9. Evaluate and Cache All Five Models
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def evaluate_all_models(test_data, model_paths):

    X = test_data[FEATURES]

    y_actual = test_data[TARGET]

    comparison_results = []

    for model_name, model_path in model_paths:

        comparison_model = load_model(model_path)

        model_metrics = calculate_metrics(
            comparison_model,
            X,
            y_actual
        )

        comparison_results.append({
            "ML Model Name": model_name.split(". ", 1)[1],
            "Accuracy": model_metrics["Accuracy"],
            "AUC": model_metrics["AUC"],
            "Precision": model_metrics["Precision"],
            "Recall": model_metrics["Recall"],
            "F1": model_metrics["F1"],
            "MCC": model_metrics["MCC"]
        })

    return comparison_results


# --------------------------------------------------
# 10. Upload Test Dataset
# --------------------------------------------------

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# --------------------------------------------------
# 11. Process Uploaded Test Dataset
# --------------------------------------------------

if uploaded_file is not None:

    import pandas as pd

    # --------------------------------------------------
    # 12. Read Uploaded Dataset
    # --------------------------------------------------

    try:

        test_data = pd.read_csv(uploaded_file)

    except Exception as error:

        st.error(
            f"Unable to read the uploaded CSV file: {error}"
        )
        st.stop()


    # --------------------------------------------------
    # 13. Display Uploaded Dataset
    # --------------------------------------------------

    st.subheader("Uploaded Dataset Preview")

    st.dataframe(
        test_data.head(),
        width="stretch"
    )

    st.write("Dataset Shape:", test_data.shape)


    # --------------------------------------------------
    # 14. Validate Dataset
    # --------------------------------------------------

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in test_data.columns
    ]

    if missing_columns:

        st.error(
            f"Missing required columns: {missing_columns}"
        )
        st.stop()

    if test_data.empty:

        st.error("The uploaded dataset is empty.")
        st.stop()

    if test_data[required_columns].isnull().any().any():

        st.error(
            "The uploaded dataset contains missing values. "
            "Please upload a dataset without missing values."
        )
        st.stop()


    # --------------------------------------------------
    # 15. Prepare Input Features and Actual Target
    # --------------------------------------------------

    X_uploaded = test_data[FEATURES]

    y_actual = test_data[TARGET]


    # --------------------------------------------------
    # 16. Load All Five Trained Models
    # --------------------------------------------------

    model_paths_tuple = tuple(
        MODEL_PATHS.items()
    )

    try:

        with st.spinner(
            "Loading all five trained classification models. Please wait..."
        ):

            loaded_models = load_all_models(
                model_paths_tuple
            )

        st.success(
            "All five classification models are ready!"
        )

    except Exception as error:

        st.error(
            f"Unable to load trained models: {error}"
        )
        st.stop()


    # --------------------------------------------------
    # 17. Select Classification Model
    # --------------------------------------------------

    st.header("Select Classification Model")

    selected_model_name = st.selectbox(
        "Choose a classification model",
        list(MODEL_PATHS.keys())
    )


    # --------------------------------------------------
    # 18. Get Selected Model
    # --------------------------------------------------

    selected_model = loaded_models[selected_model_name]


    # --------------------------------------------------
    # 19. Generate Predictions
    # --------------------------------------------------

    try:

        with st.spinner(
            f"Generating predictions using {selected_model_name}..."
        ):

            y_pred = selected_model.predict(X_uploaded)

        st.success(
            f"Predictions generated using {selected_model_name}!"
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )
        st.stop()


    # --------------------------------------------------
    # 20. Calculate Selected Model Evaluation Metrics
    # --------------------------------------------------

    try:

        with st.spinner("Calculating evaluation metrics..."):

            metrics = calculate_metrics(
                selected_model,
                X_uploaded,
                y_actual
            )

        accuracy = metrics["Accuracy"]
        auc = metrics["AUC"]
        precision = metrics["Precision"]
        recall = metrics["Recall"]
        f1 = metrics["F1"]
        mcc = metrics["MCC"]


        # --------------------------------------------------
        # 21. Evaluation Metrics
        # --------------------------------------------------

        st.header("Evaluation Metrics")

        st.write(
            "For the selected classification model, "
            "the following six evaluation metrics are calculated "
            "as specified in the assignment."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "1. Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "2. AUC Score",
            f"{auc:.4f}"
        )

        col3.metric(
            "3. Precision",
            f"{precision:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "4. Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "5. F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "6. Matthews Correlation Coefficient (MCC Score)",
            f"{mcc:.4f}"
        )


        # --------------------------------------------------
        # 22. Confusion Matrix
        # --------------------------------------------------

        from sklearn.metrics import confusion_matrix

        import matplotlib.pyplot as plt
        import seaborn as sns

        with st.spinner("Generating confusion matrix..."):

            class_labels = selected_model.classes_

            cm = confusion_matrix(
                y_actual,
                y_pred,
                labels=class_labels
            )

            fig, ax = plt.subplots(figsize=(9, 6))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels,
                ax=ax
            )

            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")

            ax.set_title(
                f"Confusion Matrix - {selected_model_name}"
            )

            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            st.header("Confusion Matrix")

            st.pyplot(fig)

            plt.close(fig)


        # --------------------------------------------------
        # 23. Classification Report
        # --------------------------------------------------

        st.header("Classification Report")

        from sklearn.metrics import classification_report

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

        styled_report = (
            report_df.style
            .format({
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-score": "{:.4f}",
                "Support": "{:.0f}"
            })
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("vertical-align", "middle")
                    ]
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),
                        ("vertical-align", "middle")
                    ]
                }
            ])
            .set_properties(**{
                "text-align": "center",
                "vertical-align": "middle"
            })
        )

        st.markdown(
            styled_report.to_html(),
            unsafe_allow_html=True
        )


        # --------------------------------------------------
        # 24. Model Performance Comparison
        # --------------------------------------------------

        st.header("Model Performance Comparison")

        st.write(
            "The following table compares all five classification "
            "models using the uploaded test dataset and the six "
            "evaluation metrics specified in the assignment."
        )

        with st.spinner(
            "Evaluating all five classification models..."
        ):

            comparison_results = evaluate_all_models(
                test_data,
                model_paths_tuple
            )

        st.success(
            "Model comparison completed successfully!"
        )

        comparison_df = pd.DataFrame(
            comparison_results
        )

        styled_comparison = (
            comparison_df.style
            .format({
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}"
            })
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("vertical-align", "middle")
                    ]
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),
                        ("vertical-align", "middle")
                    ]
                }
            ])
            .set_properties(**{
                "text-align": "center",
                "vertical-align": "middle"
            })
        )

        st.markdown(
            styled_comparison.to_html(),
            unsafe_allow_html=True
        )


    except Exception as error:

        st.error(
            f"Unable to calculate evaluation metrics: {error}"
        )


else:

    st.info(
        "Please upload a CSV file to generate predictions "
        "and evaluate the selected classification model."
    )
