"""
Smart Diabetes Data Analyzer - Complete Production System
A comprehensive ML-powered healthcare analytics platform

Requirements:
pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn prophet shap matplotlib seaborn plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Forecasting
from prophet import Prophet

# Explainability
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Smart Diabetes Data Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stButton>button {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #cccccc;
        padding: 0.5rem 1rem;
        font-size: 14px;
    }
    .stButton>button:hover {
        background-color: #f0f0f0;
        border-color: #999999;
    }
    h1, h2, h3 {
        font-family: 'Georgia', serif;
        color: #1a1a1a;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

class DataProcessor:
    """Handles all data cleaning, preprocessing, and feature engineering"""
    
    def __init__(self):
        self.numeric_imputer = KNNImputer(n_neighbors=5)
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def detect_column_types(self, df):
        """Automatically detect numeric, categorical, and temporal columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Detect temporal columns
        temporal_cols = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                temporal_cols.append(col)
                if col in categorical_cols:
                    categorical_cols.remove(col)
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'temporal': temporal_cols
        }
    
    def handle_missing_values(self, df, column_types):
        """Advanced missing value imputation"""
        df_clean = df.copy()
        
        # Numeric imputation with KNN
        if column_types['numeric']:
            numeric_data = df_clean[column_types['numeric']]
            df_clean[column_types['numeric']] = self.numeric_imputer.fit_transform(numeric_data)
        
        # Categorical imputation
        if column_types['categorical']:
            categorical_data = df_clean[column_types['categorical']]
            df_clean[column_types['categorical']] = self.categorical_imputer.fit_transform(categorical_data)
        
        return df_clean
    
    def detect_outliers(self, df, numeric_cols):
        """Detect and handle outliers using IsolationForest"""
        if not numeric_cols:
            return df
        
        df_clean = df.copy()
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        
        # Detect outliers
        outlier_predictions = iso_forest.fit_predict(df_clean[numeric_cols].fillna(0))
        
        # Clip outliers to 1st and 99th percentile
        for col in numeric_cols:
            q1 = df_clean[col].quantile(0.01)
            q99 = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(lower=q1, upper=q99)
        
        return df_clean
    
    def engineer_features(self, df):
        """Create derived features for diabetes analysis"""
        df_eng = df.copy()
        
        # Common diabetes dataset features
        if 'num_inpatient' in df.columns and 'num_outpatient' in df.columns and 'num_emergency' in df.columns:
            df_eng['total_visits'] = df_eng['num_inpatient'] + df_eng['num_outpatient'] + df_eng['num_emergency']
        
        # Age midpoint calculation
        if 'age' in df.columns and df['age'].dtype == 'object':
            age_mapping = {
                '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
                '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
                '[80-90)': 85, '[90-100)': 95
            }
            df_eng['age_numeric'] = df_eng['age'].map(age_mapping)
        
        # Comorbidity score
        diagnosis_cols = [col for col in df.columns if 'diag' in col.lower()]
        if diagnosis_cols:
            df_eng['comorbidity_score'] = df_eng[diagnosis_cols].notna().sum(axis=1)
        
        return df_eng
    
    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        return df_encoded
    
    def normalize_features(self, df, numeric_cols):
        """Normalize numeric features"""
        df_normalized = df.copy()
        
        if numeric_cols:
            df_normalized[numeric_cols] = self.scaler.fit_transform(df_normalized[numeric_cols])
        
        return df_normalized
    
    def full_pipeline(self, df):
        """Complete preprocessing pipeline"""
        # Remove duplicates
        df_clean = df.drop_duplicates()
        
        # Detect column types
        column_types = self.detect_column_types(df_clean)
        
        # Handle missing values
        df_clean = self.handle_missing_values(df_clean, column_types)
        
        # Detect and handle outliers
        df_clean = self.detect_outliers(df_clean, column_types['numeric'])
        
        # Feature engineering
        df_clean = self.engineer_features(df_clean)
        
        # Update column types after feature engineering
        column_types = self.detect_column_types(df_clean)
        
        # Encode categorical variables
        df_clean = self.encode_categorical(df_clean, column_types['categorical'])
        
        return df_clean, column_types


# ============================================================================
# MACHINE LEARNING MODEL
# ============================================================================
import numpy as np
import shap
import xgboost as xgb
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)


class DiabetesPredictor:
    """XGBoost-based prediction model for diabetes readmission risk"""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def prepare_data(self, df, target_col='readmitted'):
        """Prepare data for model training"""
        if target_col not in df.columns:
            return None, None, None, None

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if y.dtype == 'object':
            y = (y != 'NO').astype(int)

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        self.feature_names = X.columns.tolist()

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train(self, X_train, y_train):
        """Train XGBoost model with SMOTE"""
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            eval_metric='logloss'
        )
        self.model.fit(X_train_balanced, y_train_balanced)
        self.is_trained = True

        # Fix malformed base_score immediately after training
        booster = self.model.get_booster()
        booster.set_param({'base_score': '0.5'})

        return self.model

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")

        y_pred = self.model.predict(X_test)
        y_pred_proba = (
            self.model.predict_proba(X_test) if hasattr(self.model, "predict_proba") else None
        )

        is_multiclass = len(set(y_test)) > 2

        try:
            roc_auc = (
                roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                if is_multiclass and y_pred_proba is not None
                else roc_auc_score(y_test, y_pred_proba[:, 1])
                if y_pred_proba is not None
                else None
            )
        except Exception:
            roc_auc = None

        metrics = {
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        return metrics

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions.")

        if self.feature_names:
            X = X[self.feature_names]

        predictions = self.model.predict(X)
        probabilities = (
            self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None
        )
        return predictions, probabilities

    def get_shap_values(self, X):
        """Compute SHAP values safely with XGBoost — handles base_score and dtype issues"""
        if not self.is_trained:
            raise RuntimeError("Train the model before explaining it.")

        if self.feature_names:
            X = X[self.feature_names].copy()

        # Ensure numeric data only
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        try:
            booster = self.model.get_booster()
            config = booster.save_config()

            # Patch malformed base_score representation
            if '"base_score": "[3.3333334E-1,3.3333334E-1,3.3333334E-1]"' in config:
                booster.set_param({'base_score': '0.33333334'})

            # Extra safety: fallback default
            if 'base_score' not in config or '[3.333' in config:
                booster.set_param({'base_score': '0.5'})

        except Exception as e:
            print("Could not patch base_score:", e)

        try:
            explainer = shap.TreeExplainer(self.model, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X)

            # Handle list output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Check if SHAP values are valid
            if shap_values is None or not hasattr(shap_values, "shape") or shap_values.size == 0:
                print("SHAP returned empty or invalid results — falling back.")
                return None, None

            return shap_values, explainer

        except Exception as e:
            print(f"SHAP computation failed: {e}")
            return None, None



# ============================================================================
# TIME-SERIES FORECASTING
# ============================================================================

class DiabetesForecaster:
    """Prophet-based time-series forecasting for diabetes metrics"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        
    def prepare_prophet_data(self, df, date_col, value_col):
        """Prepare data in Prophet format (ds, y)"""
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': df[value_col]
        })
        
        return prophet_df.dropna()
    
    def train(self, prophet_df):
        """Train Prophet model"""
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        self.model.fit(prophet_df)
        self.is_trained = True
        
        return self.model
    
    def forecast(self, periods=30):
        """Generate forecast for specified periods"""
        if not self.is_trained:
            return None
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        return forecast
    
    def calculate_mape(self, actual, predicted):
        """Calculate Mean Absolute Percentage Error"""
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_eda_distributions(df, numeric_cols):
    """Create distribution plots for numeric features"""
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols[:9]):
        axes[idx].hist(df[col].dropna(), bins=30, color='#3b82f6', edgecolor='black', alpha=0.7)
        axes[idx].set_title(col, fontsize=12)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df, numeric_cols):
    """Create correlation heatmap"""
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14, pad=20)
    
    return fig


def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, ax=ax,
                xticklabels=['Not Readmitted', 'Readmitted'],
                yticklabels=['Not Readmitted', 'Readmitted'])
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, pad=20)
    
    return fig


def plot_shap_summary(shap_values, X, feature_names):
    """Create SHAP summary plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, plot_type='bar')
    plt.tight_layout()
    
    return fig


def plot_forecast(forecast_df):
    """Create interactive forecast plot"""
    fig = go.Figure()
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#3b82f6', width=2)
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fillcolor='rgba(59, 130, 246, 0.2)',
        fill='tonexty',
        showlegend=False
    ))
    
    fig.update_layout(
        title='30-Day Forecast with Confidence Intervals',
        xaxis_title='Date',
        yaxis_title='Predicted Value',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("<h1 style='text-align: center; font-family: Georgia;'>Smart Diabetes Data Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Advanced ML-Powered Healthcare Analytics Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
        st.session_state.cleaned_data = None
        st.session_state.column_types = None
        st.session_state.predictions = None
        st.session_state.forecast = None
        st.session_state.model_metrics = None
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["Upload & Process", "Exploratory Analysis", "Predictive Modeling", 
         "Time-Series Forecast", "Model Explainability", "Export Results"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Privacy Notice")
    st.sidebar.info("All data is processed in-memory. No files are stored or transmitted externally.")
    st.sidebar.warning("For research and educational use only. Do not upload identifiable patient data.")
    
    # ========================================================================
    # PAGE 1: UPLOAD & PROCESS
    # ========================================================================
    
    if page == "Upload & Process":
        st.header("Data Upload & Preprocessing")
        
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"File uploaded successfully! Shape: {df.shape}")
            
            # Show raw data preview
            with st.expander("View Raw Data"):
                st.dataframe(df.head(20))
            
            # Process button
            if st.button("Run Automated Preprocessing Pipeline"):
                with st.spinner("Processing data..."):
                    processor = DataProcessor()
                    cleaned_df, column_types = processor.full_pipeline(df)
                    
                    # Store in session state
                    st.session_state.cleaned_data = cleaned_df
                    st.session_state.column_types = column_types
                    st.session_state.data_processed = True
                    
                    st.success("Data preprocessing completed!")
                    
                    # Show cleaning summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Records", f"{len(cleaned_df):,}")
                    
                    with col2:
                        st.metric("Numeric Features", len(column_types['numeric']))
                    
                    with col3:
                        st.metric("Categorical Features", len(column_types['categorical']))
                    
                    with col4:
                        st.metric("Temporal Features", len(column_types['temporal']))
                    
                    # Show cleaned data
                    st.subheader("Cleaned Dataset Preview")
                    st.dataframe(cleaned_df.head(20))
    
    # ========================================================================
    # PAGE 2: EXPLORATORY ANALYSIS
    # ========================================================================
    
    elif page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        
        if not st.session_state.data_processed:
            st.warning("Please upload and process data first.")
        else:
            df = st.session_state.cleaned_data
            column_types = st.session_state.column_types
            
            # Descriptive Statistics
            st.subheader("Descriptive Statistics")
            st.dataframe(df[column_types['numeric']].describe())
            
            # Distribution Plots
            st.subheader("Feature Distributions")
            if column_types['numeric']:
                fig_dist = plot_eda_distributions(df, column_types['numeric'][:9])
                st.pyplot(fig_dist)
            
            # Correlation Heatmap
            st.subheader("Correlation Analysis")
            if len(column_types['numeric']) > 1:
                fig_corr = plot_correlation_heatmap(df, column_types['numeric'][:15])
                st.pyplot(fig_corr)
            
            # Download cleaned data
            st.subheader("Download Cleaned Dataset")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="cleaned_diabetes_data.csv",
                mime="text/csv"
            )
    
    # ========================================================================
    # PAGE 3: PREDICTIVE MODELING
    # ========================================================================
    
    elif page == "Predictive Modeling":
        st.header("XGBoost Readmission Risk Prediction")
        
        if not st.session_state.data_processed:
            st.warning("Please upload and process data first.")
        else:
            df = st.session_state.cleaned_data
            
            # Check for target column
            target_options = [col for col in df.columns if 'readmit' in col.lower() or 'outcome' in col.lower()]
            
            if not target_options:
                st.error("No suitable target column found. Please ensure your dataset contains a readmission or outcome column.")
            else:
                target_col = st.selectbox("Select Target Column", target_options)
                
                if st.button("Train XGBoost Model"):
                    with st.spinner("Training model..."):
                        predictor = DiabetesPredictor()
                        
                        # Prepare data
                        X_train, X_test, y_train, y_test = predictor.prepare_data(df, target_col)
                        
                        if X_train is not None:
                            # Train model
                            predictor.train(X_train, y_train)
                            
                            # Evaluate
                            metrics = predictor.evaluate(X_test, y_test)
                            
                            # Make predictions on full dataset
                            X_full = df.drop(columns=[target_col])
                            X_full = X_full[predictor.feature_names]
                            predictions, probabilities = predictor.predict(X_full)
                            
                            # Store results
                            st.session_state.predictions = {
                                'predictions': predictions,
                                'probabilities': probabilities,
                                'predictor': predictor
                            }
                            st.session_state.model_metrics = metrics
                            
                            st.success("Model training completed!")
                            
                            # Display metrics
                            st.subheader("Model Performance Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            report = metrics['classification_report']['1']
                            
                            with col1:
                                st.metric("Recall", f"{metrics['recall']:.3f}")
                            
                            with col2:
                                st.metric("Precision", f"{report['precision']:.3f}")
                            
                            with col3:
                                st.metric("F1-Score", f"{report['f1-score']:.3f}")
                            
                            with col4:
                                st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
                            
                            # Confusion Matrix
                            st.subheader("Confusion Matrix")
                            fig_cm = plot_confusion_matrix(metrics['confusion_matrix'])
                            st.pyplot(fig_cm)
                            
                            # Prediction Results
                            st.subheader("Prediction Results")
                            
                            # Create results dataframe
                            results_df = pd.DataFrame({
                                'Patient_ID': range(1, len(predictions) + 1),
                                'Risk_Probability': probabilities,
                                'Risk_Category': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' for p in probabilities],
                                'Predicted_Class': ['Readmitted' if p == 1 else 'Not Readmitted' for p in predictions]
                            })
                            
                            # Sort by risk
                            results_df = results_df.sort_values('Risk_Probability', ascending=False)
                            
                            # Display top 20 high-risk patients
                            st.dataframe(results_df.head(20))
                            
                            # Risk distribution
                            risk_counts = results_df['Risk_Category'].value_counts()
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("High Risk", risk_counts.get('High', 0))
                            
                            with col2:
                                st.metric("Medium Risk", risk_counts.get('Medium', 0))
                            
                            with col3:
                                st.metric("Low Risk", risk_counts.get('Low', 0))
                            
                            # Download predictions
                            csv_pred = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions",
                                data=csv_pred,
                                file_name="readmission_predictions.csv",
                                mime="text/csv"
                            )
    
    # ========================================================================
    # PAGE 4: TIME-SERIES FORECAST
    # ========================================================================
    
    elif page == "Time-Series Forecast":
        st.header("Prophet Time-Series Forecasting")
        
        if not st.session_state.data_processed:
            st.warning("Please upload and process data first.")
        else:
            df = st.session_state.cleaned_data
            column_types = st.session_state.column_types
            
            if not column_types['temporal'] or not column_types['numeric']:
                st.error("Time-series forecasting requires both temporal and numeric columns.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    date_col = st.selectbox("Select Date Column", column_types['temporal'])
                
                with col2:
                    value_col = st.selectbox("Select Value Column", column_types['numeric'])
                
                forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
                
                if st.button("Generate Forecast"):
                    with st.spinner("Training Prophet model..."):
                        forecaster = DiabetesForecaster()
                        
                        # Prepare data
                        prophet_df = forecaster.prepare_prophet_data(df, date_col, value_col)
                        
                        # Train model
                        forecaster.train(prophet_df)
                        
                        # Generate forecast
                        forecast = forecaster.forecast(periods=forecast_days)
                        
                        # Store results
                        st.session_state.forecast = forecast
                        
                        st.success("Forecast generated successfully!")
                        
                        # Plot forecast
                        st.subheader(f"{forecast_days}-Day Forecast")
                        fig_forecast = plot_forecast(forecast.tail(forecast_days))
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # Forecast summary
                        st.subheader("Forecast Summary")
                        
                        future_forecast = forecast.tail(forecast_days)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Mean Predicted Value", f"{future_forecast['yhat'].mean():.2f}")
                        
                        with col2:
                            st.metric("Trend Direction", "Increasing" if future_forecast['trend'].iloc[-1] > future_forecast['trend'].iloc[0] else "Decreasing")
                        
                        with col3:
                            st.metric("Confidence Interval", "95%")
                        
                        # Component analysis
                        st.subheader("Trend Components")
                        
                        fig_components = forecaster.model.plot_components(forecast)
                        st.pyplot(fig_components)
                        
                        # Download forecast
                        csv_forecast = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
                        st.download_button(
                            label="Download Forecast",
                            data=csv_forecast,
                            file_name="diabetes_forecast.csv",
                            mime="text/csv"
                        )
    
    # ========================================================================
    # PAGE 5: MODEL EXPLAINABILITY
    # ========================================================================
    elif page == "Model Explainability":
        st.header("SHAP Model Explainability")

    if not st.session_state.get('predictions'):
        st.warning("Please train the prediction model first.")
    else:
        df = st.session_state.cleaned_data
        predictor = st.session_state.predictions['predictor']

        st.subheader("Global Feature Importance")

        shap_success = False

        try:
            with st.spinner("Calculating SHAP values..."):
                # Prepare data
                X_data = df[predictor.feature_names].copy()
                X_sample = X_data.sample(min(100, len(X_data)), random_state=42)

                # Compute SHAP safely
                shap_values, explainer = predictor.get_shap_values(X_sample)

                # Validate SHAP output
                if shap_values is not None and hasattr(shap_values, 'shape'):
                    st.success("SHAP values calculated successfully!")

                    # --- SHAP Summary Plot ---
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(
                        shap_values,
                        X_sample,
                        feature_names=predictor.feature_names,
                        show=False,
                        plot_type='bar',
                        max_display=15
                    )
                    st.pyplot(fig)
                    plt.close()

                    # --- Feature Importance Table ---
                    st.subheader("Top Feature Contributions")

                    try:
                        shap_array = np.array(shap_values)
                        if shap_array.ndim > 2:
                            shap_array = shap_array.reshape(shap_array.shape[0], -1)

                        shap_importances = np.abs(shap_array).mean(axis=0)
                        shap_importances = np.ravel(shap_importances)

                        feature_names = np.array(predictor.feature_names).ravel()
                        min_len = min(len(feature_names), len(shap_importances))
                        feature_names = feature_names[:min_len]
                        shap_importances = shap_importances[:min_len]

                        feature_importance = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': shap_importances
                        }).sort_values('Importance', ascending=False)

                    except Exception as e:
                        st.error(f"SHAP feature importance calculation failed: {e}")
                        feature_importance = pd.DataFrame(columns=["Feature", "Importance"])

                    st.dataframe(feature_importance.head(15), use_container_width=True)

                    # --- Bar Chart ---
                    fig_bar = px.bar(
                        feature_importance.head(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Features by SHAP Importance',
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)

                    shap_success = True

                else:
                    raise ValueError("SHAP values are invalid")

        except Exception as e:
            st.warning(f"SHAP calculation encountered an issue: {str(e)}")
            st.info("Falling back to XGBoost native feature importance...")

        # --- Fallback if SHAP fails ---
        if not shap_success:
            try:
                st.subheader("XGBoost Feature Importance (Fallback Method)")

                importance_dict = predictor.model.get_booster().get_score(importance_type='gain')

                feature_importance = pd.DataFrame([
                    {
                        'Feature': predictor.feature_names[int(k.replace('f', ''))]
                        if k.startswith('f') and k.replace('f', '').isdigit()
                        and int(k.replace('f', '')) < len(predictor.feature_names)
                        else k,
                        'Importance': v
                    }
                    for k, v in importance_dict.items()
                ]).sort_values('Importance', ascending=False)

                st.dataframe(feature_importance.head(15), use_container_width=True)

                fig_bar = px.bar(
                    feature_importance.head(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 10 Features by XGBoost Gain',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)

                shap_success = True

            except Exception as e:
                st.error(f"Could not calculate feature importance: {str(e)}")

                try:
                    if hasattr(predictor.model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'Feature': predictor.feature_names,
                            'Importance': predictor.model.feature_importances_
                        }).sort_values('Importance', ascending=False)

                        st.dataframe(feature_importance.head(15), use_container_width=True)
                        shap_success = True
                except:
                    st.error("Unable to extract any feature importance. Please retrain the model.")

        # --- Clinical Insights ---
        if shap_success:
            st.subheader("Clinical Insights")

            try:
                top_features = feature_importance.head(5)['Feature'].tolist()

                insights = {
                    'time_in_hospital': "Longer hospital stays correlate with higher readmission risk due to increased disease complexity.",
                    'num_medications': "Patients on multiple medications show increased readmission likelihood due to comorbidities and polypharmacy risks.",
                    'num_lab_procedures': "High lab procedure counts indicate more intensive monitoring requirements and disease severity.",
                    'age': "Advanced age is associated with higher readmission risk due to decreased physiological reserve.",
                    'num_procedures': "Multiple procedures suggest complex medical needs requiring careful post-discharge management.",
                    'number_diagnoses': "Multiple diagnoses indicate comorbidity burden and increased care complexity.",
                    'num_inpatient': "Prior inpatient visits suggest chronic conditions requiring ongoing management.",
                    'glucose_level': "Elevated glucose levels indicate poor glycemic control and diabetes management challenges.",
                    'A1C_level': "Higher A1C levels reflect long-term glucose control issues.",
                    'num_emergency': "Emergency room visits indicate acute complications or poor outpatient management.",
                    'total_visits': "High healthcare utilization suggests complex medical needs.",
                    'comorbidity_score': "Higher comorbidity burden increases readmission risk."
                }

                for idx, feature in enumerate(top_features, 1):
                    matched = False
                    for key, desc in insights.items():
                        if key in feature.lower():
                            st.info(f"**{idx}. {feature}**: {desc}")
                            matched = True
                            break
                    if not matched:
                        st.info(f"**{idx}. {feature}**: Significant predictor of readmission risk based on model analysis.")

            except Exception:
                st.warning("Could not generate clinical insights.")

        # --- Model Configuration ---
        st.subheader("Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**XGBoost Parameters**")
            st.write("- Algorithm: Gradient Boosting")
            st.write("- Trees: 100")
            st.write("- Max Depth: 6")
            st.write("- Learning Rate: 0.1")
            st.write("- Subsample: 0.8")
            st.write("- Colsample by Tree: 0.8")

        with col2:
            st.markdown("**Explainability Method**")
            st.write("- Primary: SHAP (SHapley Additive exPlanations)")
            st.write("- Fallback: XGBoost Native Importance")
            st.write("- Interpretation: Global Feature Importance")
            st.write("- Transparency: High")
            st.write(f"- Feature Count: {len(predictor.feature_names)}")

        # --- Recommendations ---
        st.subheader("Clinical Recommendations")

        st.markdown("""
        <div style='background-color: #e3f2fd; padding: 15px; border-left: 4px solid #2196f3; margin: 10px 0;'>
            <h4 style='margin-top: 0;'>High-Risk Patient Management</h4>
            <p>Patients with risk probability above 70% should receive enhanced discharge planning,
            including comprehensive medication reconciliation and 48-hour follow-up appointments.</p>
        </div>

        <div style='background-color: #fff3e0; padding: 15px; border-left: 4px solid #ff9800; margin: 10px 0;'>
            <h4 style='margin-top: 0;'>Medication Optimization</h4>
            <p>Review polypharmacy cases where patients are on more than 15 medications.
            Consider deprescribing protocols and pharmacist consultation.</p>
        </div>

        <div style='background-color: #e8f5e9; padding: 15px; border-left: 4px solid #4caf50; margin: 10px 0;'>
            <h4 style='margin-top: 0;'>Resource Allocation</h4>
            <p>Focus intensive case management resources on patients with extended hospital stays
            and multiple comorbidities for maximum impact.</p>
        </div>
        """, unsafe_allow_html=True)

                

    
    # ========================================================================
    # PAGE 6: EXPORT RESULTS
    # ========================================================================
    
    elif page == "Export Results":
        st.header("Export Analysis Results")
        
        st.markdown("### Available Downloads")
        
        # Cleaned data
        if st.session_state.cleaned_data is not None:
            st.subheader("1. Cleaned Dataset")
            csv_clean = st.session_state.cleaned_data.to_csv(index=False)
            st.download_button(
                label="Download Cleaned Data (CSV)",
                data=csv_clean,
                file_name="cleaned_diabetes_data.csv",
                mime="text/csv",
                key="download_cleaned"
            )
        
        # Predictions
        if st.session_state.predictions is not None:
            st.subheader("2. Prediction Results")
            
            predictions = st.session_state.predictions['predictions']
            probabilities = st.session_state.predictions['probabilities']
            
            results_df = pd.DataFrame({
                'Patient_ID': range(1, len(predictions) + 1),
                'Risk_Probability': probabilities,
                'Risk_Category': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' for p in probabilities],
                'Predicted_Class': ['Readmitted' if p == 1 else 'Not Readmitted' for p in predictions]
            })
            
            csv_pred = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions (CSV)",
                data=csv_pred,
                file_name="readmission_predictions.csv",
                mime="text/csv",
                key="download_predictions"
            )
        
        # Forecast
        if st.session_state.forecast is not None:
            st.subheader("3. Time-Series Forecast")
            
            forecast = st.session_state.forecast
            csv_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].to_csv(index=False)
            
            st.download_button(
                label="Download Forecast (CSV)",
                data=csv_forecast,
                file_name="diabetes_forecast.csv",
                mime="text/csv",
                key="download_forecast"
            )
        
        # Model metrics
        if st.session_state.model_metrics is not None:
            st.subheader("4. Model Performance Report")
            
            metrics = st.session_state.model_metrics
            report = metrics['classification_report']
            
            report_text = f"""
            DIABETES READMISSION PREDICTION - MODEL PERFORMANCE REPORT
            =========================================================
            
            Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            PERFORMANCE METRICS:
            -------------------
            Recall (Sensitivity):    {metrics['recall']:.4f}
            Precision:               {report['1']['precision']:.4f}
            F1-Score:                {report['1']['f1-score']:.4f}
            ROC-AUC Score:           {metrics['roc_auc']:.4f}
            
            CLASSIFICATION REPORT:
            ---------------------
            Class 0 (Not Readmitted):
                Precision: {report['0']['precision']:.4f}
                Recall:    {report['0']['recall']:.4f}
                F1-Score:  {report['0']['f1-score']:.4f}
                Support:   {report['0']['support']}
            
            Class 1 (Readmitted):
                Precision: {report['1']['precision']:.4f}
                Recall:    {report['1']['recall']:.4f}
                F1-Score:  {report['1']['f1-score']:.4f}
                Support:   {report['1']['support']}
            
            Overall Accuracy: {report['accuracy']:.4f}
            
            CONFUSION MATRIX:
            ----------------
            {metrics['confusion_matrix']}
            
            MODEL CONFIGURATION:
            -------------------
            Algorithm: XGBoost Classifier
            Trees: 100
            Max Depth: 6
            Learning Rate: 0.1
            Class Balancing: SMOTE
            
            DISCLAIMER:
            ----------
            This system is for research and educational use only.
            Clinical decisions should not be based solely on model predictions.
            Always consult with qualified healthcare professionals.
            """
            
            st.download_button(
                label="Download Performance Report (TXT)",
                data=report_text,
                file_name="model_performance_report.txt",
                mime="text/plain",
                key="download_report"
            )
        
        # Complete analysis summary
        st.subheader("5. Complete Analysis Package")
        
        st.info("Generate a comprehensive ZIP file containing all analysis outputs, visualizations, and reports.")
        
        if st.button("Generate Complete Package"):
            st.warning("Complete package generation requires additional file handling. Individual downloads are available above.")


# ============================================================================
# SAMPLE DATA GENERATOR (FOR TESTING)
# ============================================================================

def generate_sample_diabetes_data(n_samples=1000):
    """Generate synthetic diabetes dataset for testing"""
    np.random.seed(42)
    
    data = {
        'patient_id': range(1, n_samples + 1),
        'age': np.random.choice(['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                                '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], n_samples),
        'time_in_hospital': np.random.randint(1, 14, n_samples),
        'num_lab_procedures': np.random.randint(1, 100, n_samples),
        'num_procedures': np.random.randint(0, 6, n_samples),
        'num_medications': np.random.randint(1, 30, n_samples),
        'num_outpatient': np.random.randint(0, 20, n_samples),
        'num_emergency': np.random.randint(0, 10, n_samples),
        'num_inpatient': np.random.randint(0, 15, n_samples),
        'number_diagnoses': np.random.randint(1, 16, n_samples),
        'glucose_level': np.random.normal(120, 30, n_samples),
        'A1C_level': np.random.normal(7.0, 1.5, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'race': np.random.choice(['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'], n_samples),
        'admission_type': np.random.choice(['Emergency', 'Urgent', 'Elective'], n_samples),
        'discharge_disposition': np.random.choice(['Home', 'Transfer', 'Expired'], n_samples, p=[0.8, 0.15, 0.05]),
        'admission_source': np.random.choice(['Emergency Room', 'Physician Referral', 'Transfer'], n_samples),
        'readmitted': np.random.choice(['NO', '<30', '>30'], n_samples, p=[0.5, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    for col in ['num_procedures', 'A1C_level', 'glucose_level']:
        mask = np.random.random(n_samples) < 0.1
        df.loc[mask, col] = np.nan
    
    return df


# ============================================================================
# ADDITIONAL HELPER FUNCTIONS
# ============================================================================

def create_sample_download():
    """Create sample dataset download button"""
    sample_df = generate_sample_diabetes_data(1000)
    csv = sample_df.to_csv(index=False)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Need Sample Data?")
    st.sidebar.download_button(
        label="Download Sample Dataset",
        data=csv,
        file_name="sample_diabetes_data.csv",
        mime="text/csv",
        help="Download a synthetic diabetes dataset for testing"
    )


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    # Add sample data download to sidebar
    create_sample_download()
    
    # Run main application
    main()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Smart Diabetes Data Analyzer v1.0</strong></p>
        <p>Powered by XGBoost, Prophet, and SHAP</p>
        <p style='font-size: 12px; margin-top: 10px;'>
            © 2025 | For Research and Educational Use Only<br>
            This system should not replace clinical judgment or professional medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)




