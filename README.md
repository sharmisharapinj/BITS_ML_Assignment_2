# Machine Learning Assignment 2 – Dry Bean Classification

## a. Problem Statement

The objective of this assignment is to implement five machine learning classification models using a common dataset and evaluate their performance using the specified evaluation metrics.

The models are trained and tested on the selected dataset to identify the best-performing classification algorithm for predicting different varieties of dry beans.

## b. Dataset Description

**Dataset Name:** Dry Bean Dataset

**Dataset Source:** [UCI Machine Learning Repository – Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry)

The Dry Bean dataset contains physical measurements of dry bean samples belonging to seven different classes:

- SEKER
- BARBUNYA
- BOMBAY
- CALI
- HOROZ
- SIRA
- DERMASON

The dataset consists of 16 numerical input features representing the geometric characteristics of dry beans and one target variable, `Class`.

| Dataset Details | Value |
|---|---:|
| Original number of records | 13,611 |
| Duplicate records removed | 68 |
| Final number of records | 13,543 |
| Number of input features | 16 |
| Target variable | Class |
| Number of classes | 7 |
| Training records | 10,834 |
| Testing records | 2,709 |

The duplicate records were removed during preprocessing. The cleaned dataset was divided into training and testing sets using an 80:20 stratified split.

## c. GitHub Repository Link

**GitHub Repository Link:**

https://github.com/sharmisharapinj/BITS_ML_Assignment_2

The repository contains the complete source code, Jupyter Notebook, Streamlit application, requirements file, testing dataset and saved trained models for all five classifiers.

## d. Models Used

The following five machine learning classification models were implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine

### Comparison Table – Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9195 | 0.9935 | 0.9201 | 0.9195 | 0.9197 | 0.9028 |
| Decision Tree | 0.8955 | 0.9357 | 0.8954 | 0.8955 | 0.8953 | 0.8737 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| Support Vector Machine | 0.9210 | 0.9886 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |

### Observations on the Performance of Each Model

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the highest AUC score of 0.9935, indicating strong class discrimination performance. It also demonstrated competitive performance across the other evaluation metrics. |
| Decision Tree | Achieved an accuracy of 0.8955. It provided reasonable classification performance but performed below Logistic Regression, KNN and SVM. |
| KNN | Achieved an accuracy of 0.9155, demonstrating competitive performance across the evaluation metrics. |
| Naive Bayes | Recorded the lowest accuracy of 0.7630 and MCC score of 0.7143 among the five models. |
| Support Vector Machine | Achieved the highest accuracy (0.9210), precision (0.9214), recall (0.9210), F1 score (0.9211) and MCC score (0.9045). |

### Overall, Winner for your dataset?

**Support Vector Machine (SVM)**

Support Vector Machine was identified as the overall winner based on its performance across the evaluation metrics. It achieved the highest Accuracy, Precision, Recall, F1 Score and Matthews Correlation Coefficient among the five models.

Although Logistic Regression achieved the highest AUC score of 0.9935, SVM demonstrated the strongest overall classification performance across the other five metrics.