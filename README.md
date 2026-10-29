# Diagnostic Analysis Classification Platform - End-To-End System Deployment
**Institution:** BITS Pilani (Work Integrated Learning Programmes Division)  
**Course Context:** M.Tech (AIML) - Machine Learning Assignment 2  

---

### a. Problem Statement
The objective of this project is to implement an end-to-end predictive diagnostic framework that evaluates morphological cell nuclei features to classify tumor cases as either Malignant or Benign. The core machine learning engine evaluates 5 foundational classifiers across 6 distinct validation dimensions to pinpoint the optimal production model node for real-time web cloud deployment.

### b. Dataset Description
This application uses structural observations from the clinical Breast Cancer Wisconsin (Diagnostic) Dataset. The analysis evaluates a high-dimensional feature domain consisting of 30 continuous spatial feature columns computed across individual observations. This comprehensive evaluation captures perimeter contours, average tissue area textures, maximum concave dimensions, and boundary radius distributions across more than 500 instances to optimize diagnostic prediction boundaries.

### c. Project Repository URL
[Insert Working Clickable Link to Your Repository Here]

### d. Model Performance Matrix Evaluation

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9825 | 0.9980 | 1.0000 | 0.9524 | 0.9756 | 0.9615 |
| **Decision Tree** | 0.9298 | 0.9254 | 0.9091 | 0.9048 | 0.9070 | 0.8507 |
| **KNN** | 0.9649 | 0.9854 | 0.9756 | 0.9286 | 0.9512 | 0.9229 |
| **Naive Bayes** | 0.9386 | 0.9861 | 0.9487 | 0.8810 | 0.9136 | 0.8655 |
| **Support Vector Machine** | 0.9737 | 0.9974 | 1.0000 | 0.9286 | 0.9630 | 0.9423 |

### e. Qualitative Diagnostic Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves outstanding linear separation. Regularized logistic optimization maps variance efficiently across scaled variables, delivering premium reliability with zero false positives. |
| **Decision Tree** | Exhibits clear structural step-variance. Tends to overfit along deeper coordinate branches, causing small generalization drops on cross-validation tests. |
| **KNN** | Offers robust boundary clusters but depends heavily on continuous scalar ranges, showing moderate sensitivity to density variations across features. |
| **Naive Bayes** | Runs smoothly with minimal computing power. However, it assumes zero correlation between parameters, which slightly impacts performance on tightly linked metrics like area and perimeter. |
| **Support Vector Machine** | Excels at high-dimensional boundary optimization using RBF kernel expansions, matching the precision profiles of linear classifiers. |
| **Overall Winner?** | **Logistic Regression** is chosen as the champion architecture for this deployment, leading with peak balanced performance across F1-Score (97.56%) and an MCC metric of 0.9615. |
