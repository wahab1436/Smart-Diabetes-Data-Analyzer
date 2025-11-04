# Smart Diabetes Data Analyzer

## Overview

Smart Diabetes Data Analyzer is a data-driven healthcare analytics project designed to predict hospital readmission risk among diabetic patients using machine learning and explainable AI.
Developed entirely in Python with Streamlit, this project combines predictive modeling, interpretability, and visualization into a single interactive dashboard that demonstrates how data science can support clinical decision-making.

## Key Highlights

* Predicts the likelihood of hospital readmission for diabetic patients using an **XGBoost classifier**.
* Handles data imbalance with **SMOTE (Synthetic Minority Oversampling Technique)** to ensure fair model learning.
* Implements **SHAP (SHapley Additive Explanations)** for feature interpretability and transparency.
* Provides **model evaluation metrics** including accuracy, precision, recall, F1-score, and ROC-AUC.
* Automatically switches to **XGBoost’s native feature importance** if SHAP values are unavailable.
* Offers **clinical insights and recommendations** derived from the most influential features.
* Built with a **Streamlit dashboard** for seamless interactivity and data exploration.

## Technology Stack

* **Programming Language:** Python
* **Framework:** Streamlit
* **Machine Learning:** XGBoost, scikit-learn, imbalanced-learn (SMOTE)
* **Explainability:** SHAP
* **Data Analysis:** pandas, numpy
* **Visualization:** plotly, matplotlib

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/smart-diabetes-analyzer.git
   cd smart-diabetes-analyzer
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:

   ```bash
   streamlit run apppp.py
   ```

## Project Structure

```
smart-diabetes-analyzer/
│
├── apppp.py                  # Main Streamlit application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Model Overview

* Algorithm: XGBoost Classifier
* Data Balancing: SMOTE
* Explainability: SHAP with XGBoost fallback
* Metrics: Precision, Recall, F1-Score, ROC-AUC

## Outcome

The Smart Diabetes Data Analyzer delivers interpretable machine learning insights that can help healthcare professionals identify high-risk patients, improve discharge planning, and allocate medical resources effectively. It demonstrates how data science can be applied to build transparent, evidence-based decision support systems in healthcare.

## Author

**Abdul Wahab**
Data Scientist\

## License

This project is licensed under the MIT License.
You are free to use, modify, and distribute it with proper attribution.

## Disclaimer

This project is intended for educational and research purposes only.
It should not be used for actual clinical or medical decision-making.
