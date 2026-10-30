# =====================================================================
# MACHINE LEARNING (ASSIGNMENT 2)
# PART 2: STREAMLIT APP ENGINE FOR EVALUATION INFRASTRUCTURE
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(page_title="BITS WILP ML Diagnostic Platform", layout="wide", initial_sidebar_state="expanded")

# UI Headers
st.title("🔬 Intelligent On-Demand Diagnosis Classification Hub")
st.caption("Developed for Work Integrated Learning Programmes (WILP) | M.Tech AIML")
st.write("---")

# Sidebar Data Controllers
st.sidebar.header("📁 Processing Controls panel")

# Target Data Ingestion Interface
uploaded_test_resource = st.sidebar.file_uploader(
    "Ingest Target Test Frame File (.CSV)", type=["csv"], 
    help="Upload the 'test_data.csv' containing identical features and targets."
)

# Target Engine Selector Dropdown
selected_classifier_label = st.sidebar.selectbox(
    "Select Target Inference Classifier Pipeline",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

def retrieve_saved_pipeline_node(model_alias):
    # Formulate file target paths pointing safely to internal structures
    base_file_string = f"{model_alias.lower().replace(' ', '_')}_classifier.pkl"
    primary_directory_path = os.path.join("model", base_file_string)
    
    if os.path.exists(primary_directory_path):
        return joblib.load(primary_directory_path)
    elif os.path.exists(base_file_string):
        return joblib.load(base_file_string)
    else:
        st.error(f"Execution Error: Missing model binary file named '{base_file_string}' or inside subfolder 'model/'.")
        return None

# Core Operational Pipeline Logic Block
if uploaded_test_resource is not None:
    # Read test dataframe matrix
    test_eval_df = pd.read_csv(uploaded_test_resource)
    
    if 'diagnosis' not in test_eval_df.columns:
        st.error("Missing Target Frame Exception: The uploaded evaluation CSV data file must explicitly contain a target column named 'diagnosis'")
    else:
        # Isolate features from targets without executing retraining sequences
        X_eval_features = test_eval_df.drop(columns=['diagnosis'])
        y_eval_groundtruth = test_eval_df['diagnosis']
        
        # Load scaling operators if present inside directory structures
        scaler_node_path = os.path.join("model", "scaler.pkl") if os.path.exists(os.path.join("model", "scaler.pkl")) else "scaler.pkl"
        scaler_operator = joblib.load(scaler_node_path) if os.path.exists(scaler_node_path) else None
        
        # Apply scaling mutations on runtime target parameters selectively
        if selected_classifier_label in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            if scaler_operator is not None:
                X_eval_transformed = scaler_operator.transform(X_eval_features)
            else:
                st.warning("Warning: StandardScaler binary missing. Proceeding with raw values.")
                X_eval_transformed = X_eval_features
        else:
            X_eval_transformed = X_eval_features
            
        # Instantiate prediction block without triggering local fit actions
        predictive_model_node = retrieve_saved_pipeline_node(selected_classifier_label)
        
        if predictive_model_node is not None:
            runtime_predictions = predictive_model_node.predict(X_eval_transformed)
            
            if hasattr(predictive_model_node, "predict_proba"):
                runtime_probabilities = predictive_model_node.predict_proba(X_eval_transformed)[:, 1]
            else:
                runtime_probabilities = runtime_predictions
                
            # Compute evaluation framework statistics
            stat_acc  = accuracy_score(y_eval_groundtruth, runtime_predictions)
            stat_auc  = roc_auc_score(y_eval_groundtruth, runtime_probabilities)
            stat_prec = precision_score(y_eval_groundtruth, runtime_predictions)
            stat_rec  = recall_score(y_eval_groundtruth, runtime_predictions)
            stat_f1   = f1_score(y_eval_groundtruth, runtime_predictions)
            stat_mcc  = matthews_corrcoef(y_eval_groundtruth, runtime_predictions)
            
            # Render Core Dashboard Analytics Indicators
            st.write(f"### 🎯 Evaluation Performance Summary Metrics: **{selected_classifier_label}**")
            metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5, metric_col_6 = st.columns(6)
            
            metric_col_1.metric("Accuracy", f"{stat_acc:.4f}")
            metric_col_2.metric("AUC Score", f"{stat_auc:.4f}")
            metric_col_3.metric("Precision", f"{stat_prec:.4f}")
            metric_col_4.metric("Recall", f"{stat_rec:.4f}")
            metric_col_5.metric("F1-Score", f"{stat_f1:.4f}")
            metric_col_6.metric("MCC Score", f"{stat_mcc:.4f}")
            
            st.write("---")
            
            # Visualization section split into twin layouts
            panel_col_left, panel_col_right = st.columns(2)
            
            with panel_col_left:
                st.write("#### 📑 Summary Text Classification Analysis Report")
                raw_report_structure = classification_report(y_eval_groundtruth, runtime_predictions, output_dict=True)
                formatted_report_dataframe = pd.DataFrame(raw_report_structure).transpose()
                st.dataframe(formatted_report_dataframe.style.format(precision=4), use_container_width=True)
                
            with panel_col_right:
                st.write("#### 🗺️ Heatmap Confusion Matrix Presentation")
                computed_confusion_matrix = confusion_matrix(y_eval_groundtruth, runtime_predictions)
                
                chart_figure, chart_axis = plt.subplots(figsize=(6, 4.5))
                sns.heatmap(
                    computed_confusion_matrix, annot=True, fmt='d', cmap='YlGnBu',
                    xticklabels=['Benign (0)', 'Malignant (1)'],
                    yticklabels=['Benign (0)', 'Malignant (1)'], 
                    ax=chart_axis, cbar=True
                )
                plt.ylabel('Observed True Diagnostic Target')
                plt.xlabel('System Assigned Predicted Diagnosis')
                st.pyplot(chart_figure)
else:
    st.info("💡 Ingestion Pipeline Idle. Please upload the generated validation file 'test_data.csv' via the sidebar profile to execute dashboard analysis panels.")
