# Diagnostic Analysis Classification Platform - End-To-End System Deployment
**Institution:** BITS Pilani (Work Integrated Learning Programmes Division)  
**Course Context:** M.Tech (AIML) - Machine Learning Assignment 2  
**Deployment Platform:** Streamlit Community Cloud  

---

### a. Problem Statement
The objective of this assignment is to develop an automated diagnostic classification system capable of analyzing structural multi-dimensional nuclear features of breast mass cells to determine if a tissue sample is **Malignant** or **Benign**. This solution implements and evaluates five separate foundational machine learning algorithms. By analyzing their outputs across six distinct evaluation metrics, we can identify the optimal pipeline node to deploy for live web-based testing.

### b. Dataset Description
The machine learning pipeline utilizes clinical feature data from the **Breast Cancer Wisconsin (Diagnostic) Dataset**. The dataset satisfies all mandatory rubric constraints:
* **Total Samples:** 569 instances (Assignment Metric Limit: ≥ 500 instances).
* **Total Active Features:** 30 continuous real-valued geometric parameter attributes (Assignment Metric Limit: ≥ 12 features).
* **Target Dimension:** Binary classification target vector (`diagnosis`), where **Malignant** matches to **1** and **Benign** maps to **0**.
* **Measured Parameters:** Calculated dimensional properties of cell nuclei including area error bounds, perimeter averages, worst range attributes, smoothness indices, and maximum concavity points.

### c. Project GitHub Repository Link: https://github.com/Shubham563/medical-breast-cancer-pipeline

### d. Model Performance Matrix Evaluation
The following quantitative matrix demonstrates the exact performance scores achieved by the serialized predictive model binary artifacts on the stratified 20% test partition data match:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9649** | 0.9917 | 0.9750 | 0.9286 | 0.9512 | **0.9245** |
| **Decision Tree** | 0.8860 | 0.8887 | 0.8372 | 0.8571 | 0.8471 | 0.7563 |
| **KNN** | 0.9474 | 0.9798 | 0.9737 | 0.8810 | 0.9250 | 0.8872 |
| **Naive Bayes** | 0.9386 | 0.9848 | 0.9487 | 0.8810 | 0.9136 | 0.8675 |
| **Support Vector Machine** | 0.9474 | **0.9954** | 0.9286 | 0.9286 | 0.9286 | 0.8869 |

### e. Qualitative Diagnostic Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Overall Winner.** Standardizing the features allows the solver to find a highly effective linear hyperplane. It achieves the best balance across Accuracy (96.49%), F1-Score (95.12%), and MCC (0.9245), indicating a reliable model with very low error margins. |
| **Decision Tree** | Suffers from noticeable variance drops. Limiting the tree depth helped control overfitting, but it still struggled with the dense, continuous ranges of the 30 descriptive columns, resulting in the lowest overall metrics. |
| **KNN** | Shows good local boundary classification. However, because it relies on standard coordinate distances, its performance is sensitive to density variations in high-dimensional feature distributions. |
| **Naive Bayes** | Executes with minimal computational overhead. Its accuracy is slightly restricted by the assumption that all features are independent, which doesn't fully hold true for closely related spatial metrics like area, perimeter, and radius. |
| **Support Vector Machine** | Excels at high-dimensional border optimization using RBF kernel expansions. It recorded the highest area under the curve score (**AUC: 0.9954**), demonstrating strong confidence in separating the two classes. |

---
