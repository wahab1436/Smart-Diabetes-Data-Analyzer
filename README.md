Smart Diabetes Data Analyzer
Overview

Smart Diabetes Data Analyzer is a healthcare analytics platform built with Streamlit.
It predicts hospital readmission risk for diabetic patients using an XGBoost-based model and provides explainable insights through SHAP.
The application integrates data preprocessing, model evaluation, and clinical interpretation in one interactive dashboard.

Key Features

End-to-end data preparation and model training

XGBoost classifier optimized with SMOTE for balanced prediction

Model performance metrics including precision, recall, F1-score, and ROC-AUC

Explainability using SHAP with fallback to XGBoost feature importance

Automated generation of clinical insights based on top predictors

Interactive Streamlit dashboard for analysis and visualization

Technology Stack

Language: Python

Libraries: XGBoost, scikit-learn, SHAP, imbalanced-learn, pandas, numpy, plotly, streamlit

How to Run

Clone the repository:

git clone https://github.com/your-username/smart-diabetes-analyzer.git
cd smart-diabetes-analyzer

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run apppp.py
Project Structure
smart-diabetes-analyzer/
│
├── apppp.py                  # Main Streamlit application (includes model and dashboard)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
Model Overview

Algorithm: XGBoost Classifier

Balancing Method: SMOTE

Explainability: SHAP with XGBoost fallback

Evaluation Metrics: Precision, Recall, F1-Score, ROC-AUC

Author

Abdul Wahab
Data Scientist
LinkedIn | Email

Disclaimer

This project is developed for educational and research purposes only.
It is not intended for clinical use or medical decision-making.
