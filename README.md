# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop a multiclass classification system to identify different varieties of dry beans using their physical measurements.

Five machine learning classification algorithms are implemented and evaluated using the same dataset. Their performance is compared using six evaluation metrics to identify the best-performing model.

## 2. Dataset Description

**Dataset:** UCI Dry Bean Dataset

**Source:** [UCI Machine Learning Repository – Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean)

The dataset contains physical measurements of dry bean samples belonging to seven different classes:

- SEKER
- BARBUNYA
- BOMBAY
- CALI
- HOROZ
- SIRA
- DERMASON

Dataset details:

| Description | Value |
|---|---:|
| Original records | 13,611 |
| Duplicate records removed | 68 |
| Final records | 13,543 |
| Input features | 16 |
| Target variable | Class |
| Number of classes | 7 |
| Training records | 10,834 |
| Testing records | 2,709 |

The dataset contains 16 numerical input features representing the geometric characteristics of dry beans and one target variable, `Class`.

## 3. Data Preprocessing

The following preprocessing steps were performed:

- Inspected the dataset structure, data types and statistical properties.
- Checked for missing values and duplicate records.
- Removed 68 duplicate records.
- Separated the input features and target variable.
- Split the cleaned dataset into training and testing sets using an 80:20 stratified split.
- Applied StandardScaler to Logistic Regression, K-Nearest Neighbor and Support Vector Machine through preprocessing pipelines.

## 4. Machine Learning Models

The following five classification algorithms were implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine

All models were trained using the training dataset and evaluated using the same testing dataset.

## 5. Evaluation Metrics

The models were evaluated using the following six metrics:

1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC Score)

Weighted averaging was used for Precision, Recall and F1 Score. AUC was calculated using the One-vs-Rest multiclass strategy with weighted averaging.

## 6. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9195 | 0.9935 | 0.9201 | 0.9195 | 0.9197 | 0.9028 |
| Decision Tree | 0.8955 | 0.9357 | 0.8954 | 0.8955 | 0.8953 | 0.8737 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Gaussian Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| SVM | 0.9210 | 0.9886 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |

## 7. Model Observations

- **Logistic Regression:** Achieved the highest AUC Score of 0.9935, indicating strong class discrimination performance.
- **Decision Tree Classifier:** Provided reasonable classification performance but performed below Logistic Regression, KNN and SVM.
- **K-Nearest Neighbor Classifier:** Achieved an Accuracy of 0.9155 and demonstrated competitive classification performance.
- **Gaussian Naive Bayes:** Recorded the lowest Accuracy of 0.7630 among the five models.
- **Support Vector Machine:** Achieved the highest Accuracy, Precision, Recall, F1 Score and MCC Score.

## 8. Overall Conclusion

All five classification models were successfully implemented and evaluated using the same dataset.

Support Vector Machine demonstrated the strongest overall classification performance, achieving the highest Accuracy (0.9210), Precision (0.9214), Recall (0.9210), F1 Score (0.9211) and MCC Score (0.9045).

Logistic Regression achieved the highest AUC Score of 0.9935.

Based on the overall classification metrics, Support Vector Machine was identified as the best-performing model.

## 9. Streamlit Application

An interactive Streamlit application was developed to evaluate the five trained classification models.

Application features:

- Upload a testing dataset in CSV format.
- Select any of the five trained classification models.
- Generate predictions using the selected model.
- View all six evaluation metrics.
- Visualize the confusion matrix.
- View the classification report.
- View the prediction summary.

The application loads previously saved models and evaluates them without performing additional model training.

**GitHub Repository:**  
https://github.com/sharmisharapinj/BITS_ML_Assignment_2

**Live Streamlit Application:**  
[Add the deployed Streamlit application URL here]

## 10. Project Structure

```text
BITS_ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
├── ML_Assignment_2.ipynb
└── .gitignore
```

## 11. Installation and Execution

### Prerequisites

- Python
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the browser. Upload `test_data.csv` to evaluate the classification models.