
# Machine Learning Assignment 2 – Dry Bean Classification

## a. Problem Statement

The aim of this assignment is to build and compare five machine learning classification models on one public dataset. The dataset used is the UCI Dry Bean dataset, where each bean is described by 16 shape measurements and the task is to predict which of 7 bean varieties it belongs to (a multi-class classification problem).

Each model is evaluated with six metrics: Accuracy, AUC Score, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC). The results are compared to find the best model for this dataset. The trained models are saved and used in a Streamlit web app that evaluates them on a test CSV without retraining.

## b. Dataset Description

| Description | Details |
|---|---|
| Dataset Source | UCI Machine Learning Repository – Dry Bean Dataset |
| Problem Type | Multi-class classification (7 classes) |
| Number of Instances | 13,611 (assignment minimum: 500) |
| Number of Features | 16, all numerical (assignment minimum: 12) |
| Target Column | Class: SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON |
| Duplicates | 68 duplicate records removed, 13,543 records remain |
| Train / Test Split | 80:20 stratified split: 10,834 training and 2,709 testing records |

The 16 features describe the physical shape of each bean, for example area, perimeter, major and minor axis length, aspect ratio, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness and four shape factors. Test-set class sizes are uneven (DERMASON 709 beans, BOMBAY only 104), so MCC and weighted-average precision, recall and F1 are reported along with accuracy.

### Dataset Selection and Preparation

The dataset meets Step 1 of the assignment: the public UCI Dry Bean dataset contains 13,611 instances (minimum 500) and 16 numerical input features (minimum 12), with seven bean varieties as the target classes. Before splitting, 68 duplicate rows were removed, leaving 13,543 records. The 16 feature columns were used as inputs and Class as the target. A stratified 80:20 split produced 10,834 training and 2,709 test records, preserving class proportions.

StandardScaler was applied in pipelines for Logistic Regression, KNN and SVM, because these models are scale-sensitive. Decision Tree and Gaussian Naive Bayes used the original features without scaling. Keeping scaling inside each pipeline ensures that the scaler is fitted on training data and then applied consistently to test data.

## c. GitHub Repository Link

[Machine Learning Assignment 2 – GitHub Repository](https://github.com/sharmisharapinj/BITS_ML_Assignment_2)

The repository contains the following files and folders:

```text
BITS_ML_Assignment_2/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
└── ML_Assignment_2.ipynb
```

It also includes the source code and the required preprocessing artifacts, such as the scalers saved along with the models.

## d. Models Used

All five models were trained on the same training set and tested on the same test set (2,709 records):

- Logistic Regression (scaler used in a pipeline)
- Decision Tree Classifier
- K-Nearest Neighbor Classifier (scaler used in a pipeline)
- Naive Bayes Classifier – Gaussian
- Support Vector Machine (scaler used in a pipeline)

### Comparison Table

The table below compares the five models on the same 2,709 test records using all six evaluation metrics. Accuracy is the share of beans classified correctly, and AUC shows how well a model ranks the seven classes. Precision, Recall and F1 are weighted averages, so the uneven class sizes are taken into account, and MCC gives one balanced score that stays reliable when classes are imbalanced.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9195 | **0.9935** | 0.9201 | 0.9195 | 0.9197 | 0.9028 |
| Decision Tree | 0.8955 | 0.9357 | 0.8954 | 0.8955 | 0.8953 | 0.8737 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| Support Vector Machine | **0.9210** | 0.9886 | **0.9214** | **0.9210** | **0.9211** | **0.9045** |

**Bold values indicate the best value for each metric.** Precision, Recall and F1 are weighted averages over the 7 classes.

Reading the table, SVM leads on Accuracy, Precision, Recall, F1 and MCC, while Logistic Regression has the highest AUC. KNN stays close to both, Decision Tree is a step lower, and Naive Bayes is the weakest on every metric except AUC, where Decision Tree is the lowest.

### Observations on the Performance of Each Model

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Got the highest AUC (0.9935), so it separates the seven classes very well. Accuracy is 0.9195, only 0.0015 below SVM (about 4 beans out of 2,709). A simple linear model works well here once the features are scaled. |
| Decision Tree | Accuracy 0.8955. It is lower than Logistic Regression, KNN and SVM on most metrics and has the lowest AUC (0.9357). A single tree gives rough probabilities and can change a lot with the training data. |
| KNN | Accuracy 0.9155 and MCC 0.8978, which is competitive and close to SVM. KNN depends on distances, so the scaling step in the pipeline is important. |
| Naive Bayes | Lowest accuracy (0.7630) and lowest MCC (0.7143). Its AUC (0.9644) is still decent, so it ranks the classes reasonably but its final predictions are weak. The independence assumption probably does not hold because many bean measurements (area, perimeter, axis lengths) are related. |
| Support Vector Machine | Best accuracy (0.9210), precision (0.9214), recall (0.9210), F1 (0.9211) and MCC (0.9045). Its AUC (0.9886) is second best. |
| Overall Winner for This Dataset | Support Vector Machine (SVM) is the overall winner. It has the highest score on five of the six metrics (Accuracy, Precision, Recall, F1 and MCC). Logistic Regression has the highest AUC (0.9935) and is a close second overall. |

### Why SVM Is the Overall Winner

SVM is the strongest overall choice because it achieves the best results on five metrics: Accuracy (0.9210), weighted Precision (0.9214), Recall (0.9210), F1 (0.9211) and MCC (0.9045). This consistent lead suggests that its predictions are not only accurate but also balanced across the seven bean classes. The RBF kernel can capture non-linear relationships among the shape measurements.

However, the lead is modest: SVM correctly classifies 2,495 of 2,709 test records, only four more than Logistic Regression and 15 more than KNN. Logistic Regression achieves the highest AUC (0.9935), ahead of SVM (0.9886), showing stronger class-ranking performance by that metric. Therefore, SVM is the overall winner when the six metrics are considered together, while Logistic Regression is a close alternative.

## Live Streamlit Application

[Open Dry Bean Classification Dashboard](https://bits-ml-assignment-2-dry-bean.streamlit.app/)

The application is deployed using Streamlit Community Cloud. It provides the following features:

- **Dataset upload option (CSV):** Default test data loads automatically, and another compatible test CSV can be uploaded.
- **Model selection dropdown:** Any of the five saved models can be chosen.
- **Display of evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC.
- **Confusion matrix and classification report:** Displays per-class evaluation results for the selected model.

The app loads the saved models and only evaluates them on the uploaded test data; nothing is retrained.
