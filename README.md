# Machine Learning Assignment 2 – Dry Bean Classification

## a. Problem Statement

The objective of this assignment is to develop and evaluate five machine learning classification models using a public dataset. The models are trained to classify dry bean varieties based on their physical measurements.

The performance of each model is evaluated using six metrics: Accuracy, AUC Score, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC). The models are compared to identify the best-performing classifier for the selected dataset.

## b. Dataset Description

The **UCI Dry Bean Dataset** was selected for this classification problem.

- **Dataset Source:** [UCI Machine Learning Repository – Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry)
- **Total records:** 13,611
- **Number of input features:** 16
- **Target variable:** `Class`
- **Number of target classes:** 7

The dataset contains 16 numerical features describing the physical characteristics of dry beans, such as area, perimeter, major and minor axis lengths, eccentricity, compactness and shape factors.

The target variable represents seven dry bean varieties: SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA and DERMASON.

After removing 68 duplicate records, the cleaned dataset contained 13,543 records. The data was divided into training and testing sets using an 80:20 stratified split, resulting in 10,834 training records and 2,709 testing records.

## c. GitHub Repository Link

[https://github.com/sharmisharapinj/BITS_ML_Assignment_2](https://github.com/sharmisharapinj/BITS_ML_Assignment_2)

## d. Models Used

The following five classification models were implemented and evaluated using the same training and testing dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Naive Bayes Classifier – Gaussian
5. Support Vector Machine (SVM)

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
| **Logistic Regression** | Achieved the highest AUC score of 0.9935, indicating excellent class discrimination. It also delivered strong and balanced performance across the other evaluation metrics. |
| **Decision Tree** | Achieved an accuracy of 0.8955. It provided reasonable classification performance but scored lower than Logistic Regression, KNN and SVM on most evaluation metrics. |
| **KNN** | Achieved an accuracy of 0.9155 and showed competitive performance across all evaluation metrics. |
| **Naive Bayes** | Achieved an accuracy of 0.7630 and an MCC score of 0.7143, the lowest among the five models. Its overall classification performance was comparatively lower. |
| **Support Vector Machine** | Achieved the highest accuracy (0.9210), precision (0.9214), recall (0.9210), F1 score (0.9211) and MCC score (0.9045), demonstrating the strongest overall classification performance. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** is the overall winner because it achieved the highest scores in five of the six evaluation metrics. Logistic Regression achieved the highest AUC score of 0.9935. |