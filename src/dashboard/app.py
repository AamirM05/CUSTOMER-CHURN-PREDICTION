"""
Dashboard application for the Telco Customer Churn project.

This module provides a Streamlit dashboard for visualizing churn prediction results.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from datetime import datetime
import xgboost as xgb

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.config_loader import (
    load_config, get_dashboard_config, get_data_paths, 
    get_processing_config, get_model_config
)
from src.data.ingestion import read_csv_data, get_db_engine
from src.models.train import get_model_instance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
# Use absolute path to config file since we might run from different directories
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/config.yaml'))
config = load_config(config_path)
dashboard_config = get_dashboard_config(config)
data_paths = get_data_paths(config)
processing_config = get_processing_config(config)

# Set page configuration
st.set_page_config(
    page_title=dashboard_config.get('title', 'Telco Customer Churn Analysis'),
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply theme
if dashboard_config.get('theme', 'light') == 'dark':
    st.markdown("""
        <style>
        .reportview-container {
            background-color: #1E1E1E;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_raw_data():
    """
    Load raw data from CSV.
    
    Returns:
        DataFrame with raw data
    """
    # Get raw data path from config
    raw_data_path = data_paths.get('raw_data_path', "data/raw/Telco Customer Churn Dataset.csv")
    
    # Convert to absolute path if it's a relative path
    if not os.path.isabs(raw_data_path):
        raw_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', raw_data_path))
    
    # If raw data path doesn't exist, try to find the original CSV
    if not os.path.exists(raw_data_path):
        original_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Telco Customer Churn Dataset.csv'))
        if os.path.exists(original_csv):
            raw_data_path = original_csv
    
    df = read_csv_data(raw_data_path)
    return df

@st.cache_data(ttl=3600)
def load_engineered_data():
    """
    Load engineered data from feature store.
    
    Returns:
        DataFrame with engineered features
    """
    # Get feature store path from config
    feature_store_path = data_paths.get('feature_store_path', 'data/feature_store/')
    
    # Convert to absolute path if it's a relative path
    if not os.path.isabs(feature_store_path):
        feature_store_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', feature_store_path))
    
    features_file = os.path.join(feature_store_path, 'engineered_features.csv')
    
    try:
        df = pd.read_csv(features_file)
        return df
    except FileNotFoundError:
        st.warning(f"Engineered features file not found: {features_file}")
        return None

@st.cache_resource
def load_latest_model():
    """
    Load the latest trained model.
    
    Returns:
        Tuple of (model, metrics, model_path)
    """
    # Get model data path from config
    model_data_path = data_paths.get('model_data_path', 'data/model_data/')
    
    # Convert to absolute path if it's a relative path
    if not os.path.isabs(model_data_path):
        model_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', model_data_path))
    
    # Create directory if it doesn't exist
    os.makedirs(model_data_path, exist_ok=True)
    
    try:
        # Find the latest model file
        model_files = [f for f in os.listdir(model_data_path) if f.startswith('model_') and f.endswith('.pkl')]
        
        if not model_files:
            return None, None, None
        
        # Sort by timestamp (assuming format model_YYYYMMDD_HHMMSS.pkl)
        latest_model_file = sorted(model_files)[-1]
        model_path = os.path.join(model_data_path, latest_model_file)
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metrics if available
        metrics_file = latest_model_file.replace('model_', 'metrics_').replace('.pkl', '.json')
        metrics_path = os.path.join(model_data_path, metrics_file)
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = None
        
        return model, metrics, model_path
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def create_sidebar():
    """
    Create sidebar with navigation and filters.
    
    Returns:
        Selected page and filters
    """
    st.sidebar.title("Navigation")
    
    # Get pages from config
    pages = dashboard_config.get('pages', [
        "Overview", 
        "Customer Analysis", 
        "Churn Prediction", 
        "What-If Analysis"
    ])
    
    selected_page = st.sidebar.radio("Go to", pages)
    
    st.sidebar.title("Filters")
    
    # Load data for filters
    df = load_raw_data()
    
    # Add filters based on data
    filters = {}
    
    if 'Contract' in df.columns:
        contract_types = ['All'] + sorted(df['Contract'].unique().tolist())
        filters['contract'] = st.sidebar.selectbox("Contract Type", contract_types)
    
    if 'tenure' in df.columns:
        max_tenure = int(df['tenure'].max())
        filters['tenure_range'] = st.sidebar.slider("Tenure (months)", 0, max_tenure, (0, max_tenure))
    
    if 'MonthlyCharges' in df.columns:
        min_charge = float(df['MonthlyCharges'].min())
        max_charge = float(df['MonthlyCharges'].max())
        filters['monthly_charges_range'] = st.sidebar.slider(
            "Monthly Charges ($)", 
            min_charge, 
            max_charge, 
            (min_charge, max_charge)
        )
    
    # Add filter for services
    service_columns = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    available_services = [col for col in service_columns if col in df.columns]
    if available_services:
        selected_service = st.sidebar.selectbox("Filter by Service", ['None'] + available_services)
        if selected_service != 'None':
            service_values = ['All'] + sorted(df[selected_service].unique().tolist())
            filters['service'] = {
                'column': selected_service,
                'value': st.sidebar.selectbox(f"{selected_service} Value", service_values)
            }
    
    # Add information about the data
    st.sidebar.title("Dataset Info")
    st.sidebar.info(f"""
    - Total Customers: {len(df)}
    - Churn Rate: {df['Churn'].value_counts(normalize=True).get('Yes', 0)*100:.1f}%
    - Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """)
    
    return selected_page, filters

def apply_filters(df, filters):
    """
    Apply filters to the DataFrame.
    
    Args:
        df: Input DataFrame
        filters: Dictionary of filters
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Apply contract filter
    if 'contract' in filters and filters['contract'] != 'All':
        filtered_df = filtered_df[filtered_df['Contract'] == filters['contract']]
    
    # Apply tenure filter
    if 'tenure_range' in filters:
        min_tenure, max_tenure = filters['tenure_range']
        filtered_df = filtered_df[(filtered_df['tenure'] >= min_tenure) & (filtered_df['tenure'] <= max_tenure)]
    
    # Apply monthly charges filter
    if 'monthly_charges_range' in filters:
        min_charge, max_charge = filters['monthly_charges_range']
        filtered_df = filtered_df[(filtered_df['MonthlyCharges'] >= min_charge) & (filtered_df['MonthlyCharges'] <= max_charge)]
    
    # Apply service filter
    if 'service' in filters and filters['service']['value'] != 'All':
        column = filters['service']['column']
        value = filters['service']['value']
        filtered_df = filtered_df[filtered_df[column] == value]
    
    return filtered_df

def overview_page(filters):
    """
    Display overview page.
    
    Args:
        filters: Dictionary of filters
    """
    st.title("Telco Customer Churn Overview")
    
    # Load data
    df = load_raw_data()
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Display key metrics
    st.header("Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(filtered_df):,}")
    
    with col2:
        churn_rate = filtered_df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        avg_tenure = filtered_df['tenure'].mean()
        st.metric("Avg. Tenure", f"{avg_tenure:.1f} months")
    
    with col4:
        avg_monthly = filtered_df['MonthlyCharges'].mean()
        st.metric("Avg. Monthly Charges", f"${avg_monthly:.2f}")
    
    # Display churn distribution
    st.header("Churn Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by contract type
        if 'Contract' in filtered_df.columns:
            contract_churn = filtered_df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
            
            if 'Yes' in contract_churn.columns:
                contract_churn = contract_churn['Yes'].sort_values(ascending=False) * 100
                
                fig = px.bar(
                    x=contract_churn.index,
                    y=contract_churn.values,
                    labels={'x': 'Contract Type', 'y': 'Churn Rate (%)'},
                    title='Churn Rate by Contract Type',
                    text_auto='.1f'
                )
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn by tenure group
        filtered_df['tenure_group'] = pd.cut(
            filtered_df['tenure'], 
            bins=[0, 12, 24, 36, 48, 60, float('inf')],
            labels=['0-1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5+ years']
        )
        
        tenure_churn = filtered_df.groupby('tenure_group')['Churn'].value_counts(normalize=True).unstack()
        
        if 'Yes' in tenure_churn.columns:
            tenure_churn = tenure_churn['Yes'] * 100
            
            fig = px.bar(
                x=tenure_churn.index,
                y=tenure_churn.values,
                labels={'x': 'Tenure Group', 'y': 'Churn Rate (%)'},
                title='Churn Rate by Tenure',
                text_auto='.1f'
            )
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    # Display service impact on churn
    st.header("Service Impact on Churn")
    
    service_columns = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    available_services = [col for col in service_columns if col in filtered_df.columns]
    
    if available_services:
        # Calculate churn rate for each service
        service_churn_rates = []
        
        for service in available_services:
            service_data = filtered_df.groupby(service)['Churn'].value_counts(normalize=True).unstack()
            
            if 'Yes' in service_data.columns:
                for category in service_data.index:
                    service_churn_rates.append({
                        'Service': service,
                        'Category': category,
                        'Churn Rate': service_data.loc[category, 'Yes'] * 100
                    })
        
        service_churn_df = pd.DataFrame(service_churn_rates)
        
        if not service_churn_df.empty:
            fig = px.bar(
                service_churn_df,
                x='Service',
                y='Churn Rate',
                color='Category',
                barmode='group',
                labels={'Churn Rate': 'Churn Rate (%)'},
                title='Churn Rate by Service Type',
                text_auto='.1f'
            )
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

def customer_analysis_page(filters):
    """
    Display customer analysis page.
    
    Args:
        filters: Dictionary of filters
    """
    st.title("Customer Analysis")
    
    # Load data
    df = load_raw_data()
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Customer demographics
    st.header("Customer Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution
        if 'gender' in filtered_df.columns:
            gender_counts = filtered_df['gender'].value_counts()
            
            fig = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title='Gender Distribution',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Senior citizen distribution
        if 'SeniorCitizen' in filtered_df.columns:
            # Convert 0/1 to No/Yes if needed
            if filtered_df['SeniorCitizen'].isin([0, 1]).all():
                filtered_df['SeniorCitizen'] = filtered_df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
            
            senior_counts = filtered_df['SeniorCitizen'].value_counts()
            
            fig = px.pie(
                values=senior_counts.values,
                names=senior_counts.index,
                title='Senior Citizen Distribution',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Customer tenure analysis
    st.header("Customer Tenure Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tenure distribution
        fig = px.histogram(
            filtered_df,
            x='tenure',
            nbins=20,
            title='Tenure Distribution',
            labels={'tenure': 'Tenure (months)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tenure vs Monthly Charges
        fig = px.scatter(
            filtered_df,
            x='tenure',
            y='MonthlyCharges',
            color='Churn',
            title='Tenure vs Monthly Charges',
            labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Service adoption
    st.header("Service Adoption")
    
    service_columns = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    available_services = [col for col in service_columns if col in filtered_df.columns]
    
    if available_services:
        # Calculate adoption rate for each service
        service_adoption = {}
        
        for service in available_services:
            # Skip if service has more than 5 categories
            if filtered_df[service].nunique() > 5:
                continue
                
            # Calculate adoption rate
            adoption_counts = filtered_df[service].value_counts()
            service_adoption[service] = adoption_counts
        
        # Create subplots
        fig = make_subplots(
            rows=len(service_adoption) // 2 + len(service_adoption) % 2,
            cols=2,
            subplot_titles=list(service_adoption.keys())
        )
        
        # Add bar charts
        for i, (service, counts) in enumerate(service_adoption.items()):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Bar(
                    x=counts.index,
                    y=counts.values,
                    text=counts.values,
                    textposition='auto'
                ),
                row=row,
                col=col
            )
        
        fig.update_layout(
            height=300 * (len(service_adoption) // 2 + len(service_adoption) % 2),
            title_text='Service Adoption',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Payment analysis
    st.header("Payment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment method distribution
        if 'PaymentMethod' in filtered_df.columns:
            payment_counts = filtered_df['PaymentMethod'].value_counts()
            
            fig = px.pie(
                values=payment_counts.values,
                names=payment_counts.index,
                title='Payment Method Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly charges distribution
        fig = px.histogram(
            filtered_df,
            x='MonthlyCharges',
            nbins=20,
            title='Monthly Charges Distribution',
            labels={'MonthlyCharges': 'Monthly Charges ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.header("Feature Correlation")
    
    # Select numerical columns
    numerical_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numerical_cols) > 1:
        # Calculate correlation matrix
        corr_matrix = filtered_df[numerical_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title='Correlation Matrix'
        )
        st.plotly_chart(fig, use_container_width=True)

def churn_prediction_page(filters):
    """
    Display churn prediction page.
    
    Args:
        filters: Dictionary of filters
    """
    st.title("Churn Prediction Analysis")
    
    # Load model and metrics
    model, metrics, model_path = load_latest_model()
    
    if model is None:
        st.warning("No trained model found. Please train a model first.")
        return
    
    # Display model information
    st.header("Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", type(model).__name__)
    
    with col2:
        if metrics and 'test' in metrics and 'accuracy' in metrics['test']:
            accuracy = metrics['test']['accuracy'] * 100
            st.metric("Test Accuracy", f"{accuracy:.1f}%")
    
    with col3:
        if metrics and 'test' in metrics and 'roc_auc' in metrics['test']:
            roc_auc = metrics['test']['roc_auc'] * 100
            st.metric("ROC AUC", f"{roc_auc:.1f}%")
    
    # Display model metrics
    if metrics:
        st.header("Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix
            if 'test' in metrics and 'confusion_matrix' in metrics['test']:
                cm = metrics['test']['confusion_matrix']
                
                # Create confusion matrix plot
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['No Churn', 'Churn'],
                    y=['No Churn', 'Churn'],
                    title='Confusion Matrix'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Metrics table
            if 'test' in metrics:
                test_metrics = metrics['test']
                
                # Create metrics table
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
                    'Value': [
                        f"{test_metrics.get('accuracy', 0) * 100:.1f}%",
                        f"{test_metrics.get('precision', 0) * 100:.1f}%",
                        f"{test_metrics.get('recall', 0) * 100:.1f}%",
                        f"{test_metrics.get('f1', 0) * 100:.1f}%",
                        f"{test_metrics.get('roc_auc', 0) * 100:.1f}%"
                    ]
                })
                
                st.table(metrics_df)
    
    # Load engineered data for feature importance
    engineered_df = load_engineered_data()
    
    if engineered_df is not None:
        st.header("Feature Importance")
        
        # Get target and ID column
        target = processing_config.get('target', 'churn').lower()
        id_column = processing_config.get('id_column', 'customerid').lower()
        
        # Find actual column names in the DataFrame (case-insensitive)
        target_col = next((col for col in engineered_df.columns if col.lower() == target), target)
        id_col = next((col for col in engineered_df.columns if col.lower() == id_column), id_column)
        
        # Extract features
        X = engineered_df.drop([target_col, id_col], axis=1, errors='ignore')
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(X.columns, importances))
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            feature_importance = dict(zip(X.columns, importances))
        else:
            feature_importance = {}
        
        if feature_importance:
            # Sort by importance
            feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
            
            # Display top 20 features
            top_features = list(feature_importance.items())[:20]
            
            fig = px.bar(
                x=[f[1] for f in top_features],
                y=[f[0] for f in top_features],
                orientation='h',
                labels={'x': 'Importance', 'y': 'Feature'},
                title='Top 20 Feature Importances'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Load raw data for churn risk analysis
    df = load_raw_data()
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    st.header("Churn Risk Analysis")
    
    # Prepare data for prediction
    if engineered_df is not None:
        # Join raw data with engineered features
        try:
            # Find ID column in both DataFrames (case-insensitive)
            id_col_filtered = next((col for col in filtered_df.columns if col.lower() == id_column), None)
            id_col_engineered = next((col for col in engineered_df.columns if col.lower() == id_column), None)
            
            if id_col_filtered and id_col_engineered:
                # Rename columns to ensure they match for merging
                filtered_df_copy = filtered_df.copy()
                engineered_df_copy = engineered_df.copy()
                
                filtered_df_copy.rename(columns={id_col_filtered: 'merge_id'}, inplace=True)
                engineered_df_copy.rename(columns={id_col_engineered: 'merge_id'}, inplace=True)
                
                # Merge on the renamed column
                merged_df = pd.merge(
                    filtered_df_copy,
                    engineered_df_copy,
                    on='merge_id',
                    suffixes=('', '_eng')
                )
            else:
                st.error(f"ID column '{id_column}' not found in one or both DataFrames.")
                return
        except Exception as e:
            st.error(f"Error merging DataFrames: {e}")
            return
        
        # Get features for prediction
        X_pred = merged_df.drop([target_col, id_col], axis=1, errors='ignore')
        X_pred = X_pred[X.columns]  # Use only columns that model was trained on
        
        # Make predictions
        try:
            # Convert categorical columns to category type for XGBoost
            if isinstance(model, xgb.XGBClassifier):
                # Convert object columns to category type
                for col in X_pred.select_dtypes(include=['object']).columns:
                    X_pred[col] = X_pred[col].astype('category')
                
                # Try with enable_categorical parameter if available
                try:
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_pred, enable_categorical=True)[:, 1]
                    else:
                        y_pred_proba = model.predict(X_pred, enable_categorical=True)
                except TypeError:
                    # If enable_categorical is not supported, try without it
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_pred)[:, 1]
                    else:
                        y_pred_proba = model.predict(X_pred)
            else:
                # For other model types
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_pred)[:, 1]
                else:
                    y_pred_proba = model.predict(X_pred)
            
            # Add predictions to DataFrame
            filtered_df['churn_probability'] = y_pred_proba
            
            # Create risk categories
            filtered_df['risk_category'] = pd.cut(
                filtered_df['churn_probability'],
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            # Display risk distribution
            risk_counts = filtered_df['risk_category'].value_counts()
            
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title='Customer Churn Risk Distribution',
                color_discrete_sequence=['green', 'orange', 'red']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display high-risk customers
            st.subheader("High-Risk Customers")
            
            high_risk_df = filtered_df[filtered_df['risk_category'] == 'High Risk'].sort_values(
                by='churn_probability', ascending=False
            )
            
            if len(high_risk_df) > 0:
                # Select columns to display
                display_cols = [
                    id_col_filtered, 'churn_probability', 'tenure', 'Contract', 
                    'MonthlyCharges', 'TotalCharges'
                ]
                display_cols = [col for col in display_cols if col in high_risk_df.columns]
                
                # Format probability as percentage
                high_risk_display = high_risk_df[display_cols].copy()
                high_risk_display['churn_probability'] = high_risk_display['churn_probability'].apply(
                    lambda x: f"{x*100:.1f}%"
                )
                
                st.dataframe(high_risk_display.head(10))
            else:
                st.info("No high-risk customers found with current filters.")
        
        except Exception as e:
            st.error(f"Error making predictions: {e}")
    else:
        st.warning("Engineered features not found. Cannot make predictions.")

def what_if_analysis_page(filters):
    """
    Display what-if analysis page.
    
    Args:
        filters: Dictionary of filters
    """
    st.title("What-If Analysis")
    
    # Load model
    model, metrics, model_path = load_latest_model()
    
    if model is None:
        st.warning("No trained model found. Please train a model first.")
        return
    
    # Load raw data for reference
    df = load_raw_data()
    
    # Load engineered data for feature names
    engineered_df = load_engineered_data()
    
    if engineered_df is None:
        st.warning("Engineered features not found. Cannot perform what-if analysis.")
        return
    
    # Get target and ID column
    target = processing_config.get('target', 'churn').lower()
    id_column = processing_config.get('id_column', 'customerid').lower()
    
    # Find actual column names in the DataFrame (case-insensitive)
    target_col = next((col for col in engineered_df.columns if col.lower() == target), target)
    id_col = next((col for col in engineered_df.columns if col.lower() == id_column), id_column)
    
    # Extract features
    X = engineered_df.drop([target_col, id_col], axis=1, errors='ignore')
    
    st.header("Customer Profile Simulator")
    st.write("Adjust the parameters below to simulate different customer profiles and predict churn probability.")
    
    # Create form for customer profile
    with st.form("customer_profile_form"):
        # Create columns for form fields
        col1, col2, col3 = st.columns(3)
        
        # Define customer attributes
        customer_profile = {}
        
        with col1:
            if 'gender' in df.columns:
                customer_profile['gender'] = st.selectbox(
                    "Gender",
                    options=df['gender'].unique()
                )
            
            if 'SeniorCitizen' in df.columns:
                senior_options = df['SeniorCitizen'].unique()
                if set(senior_options) == {0, 1}:
                    senior_options_map = {0: 'No', 1: 'Yes'}
                    customer_profile['SeniorCitizen'] = st.selectbox(
                        "Senior Citizen",
                        options=[0, 1],
                        format_func=lambda x: senior_options_map[x]
                    )
                else:
                    customer_profile['SeniorCitizen'] = st.selectbox(
                        "Senior Citizen",
                        options=senior_options
                    )
            
            if 'Partner' in df.columns:
                customer_profile['Partner'] = st.selectbox(
                    "Partner",
                    options=df['Partner'].unique()
                )
            
            if 'Dependents' in df.columns:
                customer_profile['Dependents'] = st.selectbox(
                    "Dependents",
                    options=df['Dependents'].unique()
                )
            
            if 'tenure' in df.columns:
                customer_profile['tenure'] = st.slider(
                    "Tenure (months)",
                    min_value=int(df['tenure'].min()),
                    max_value=int(df['tenure'].max()),
                    value=int(df['tenure'].median())
                )
        
        with col2:
            if 'PhoneService' in df.columns:
                customer_profile['PhoneService'] = st.selectbox(
                    "Phone Service",
                    options=df['PhoneService'].unique()
                )
            
            if 'MultipleLines' in df.columns:
                customer_profile['MultipleLines'] = st.selectbox(
                    "Multiple Lines",
                    options=df['MultipleLines'].unique()
                )
            
            if 'InternetService' in df.columns:
                customer_profile['InternetService'] = st.selectbox(
                    "Internet Service",
                    options=df['InternetService'].unique()
                )
            
            if 'OnlineSecurity' in df.columns:
                customer_profile['OnlineSecurity'] = st.selectbox(
                    "Online Security",
                    options=df['OnlineSecurity'].unique()
                )
            
            if 'OnlineBackup' in df.columns:
                customer_profile['OnlineBackup'] = st.selectbox(
                    "Online Backup",
                    options=df['OnlineBackup'].unique()
                )
            
            if 'DeviceProtection' in df.columns:
                customer_profile['DeviceProtection'] = st.selectbox(
                    "Device Protection",
                    options=df['DeviceProtection'].unique()
                )
        
        with col3:
            if 'TechSupport' in df.columns:
                customer_profile['TechSupport'] = st.selectbox(
                    "Tech Support",
                    options=df['TechSupport'].unique()
                )
            
            if 'StreamingTV' in df.columns:
                customer_profile['StreamingTV'] = st.selectbox(
                    "Streaming TV",
                    options=df['StreamingTV'].unique()
                )
            
            if 'StreamingMovies' in df.columns:
                customer_profile['StreamingMovies'] = st.selectbox(
                    "Streaming Movies",
                    options=df['StreamingMovies'].unique()
                )
            
            if 'Contract' in df.columns:
                customer_profile['Contract'] = st.selectbox(
                    "Contract",
                    options=df['Contract'].unique()
                )
            
            if 'PaperlessBilling' in df.columns:
                customer_profile['PaperlessBilling'] = st.selectbox(
                    "Paperless Billing",
                    options=df['PaperlessBilling'].unique()
                )
            
            if 'PaymentMethod' in df.columns:
                customer_profile['PaymentMethod'] = st.selectbox(
                    "Payment Method",
                    options=df['PaymentMethod'].unique()
                )
            
            if 'MonthlyCharges' in df.columns:
                customer_profile['MonthlyCharges'] = st.slider(
                    "Monthly Charges ($)",
                    min_value=float(df['MonthlyCharges'].min()),
                    max_value=float(df['MonthlyCharges'].max()),
                    value=float(df['MonthlyCharges'].median())
                )
        
        # Submit button
        submitted = st.form_submit_button("Predict Churn Probability")
    
    # Process form submission
    if submitted:
        try:
            # Create a DataFrame from the customer profile
            profile_df = pd.DataFrame([customer_profile])
            
            # Prepare features for prediction
            # This would typically involve the same preprocessing as in the training pipeline
            # For simplicity, we'll just use the raw features here
            
            # Convert categorical columns to category type for XGBoost
            if isinstance(model, xgb.XGBClassifier):
                # Convert object columns to category type
                for col in profile_df.select_dtypes(include=['object']).columns:
                    profile_df[col] = profile_df[col].astype('category')
                
                # Try with enable_categorical parameter if available
                try:
                    if hasattr(model, 'predict_proba'):
                        churn_probability = model.predict_proba(profile_df, enable_categorical=True)[0, 1]
                    else:
                        churn_probability = model.predict(profile_df, enable_categorical=True)[0]
                except TypeError:
                    # If enable_categorical is not supported, try without it
                    if hasattr(model, 'predict_proba'):
                        churn_probability = model.predict_proba(profile_df)[0, 1]
                    else:
                        churn_probability = model.predict(profile_df)[0]
            else:
                # For other model types
                if hasattr(model, 'predict_proba'):
                    churn_probability = model.predict_proba(profile_df)[0, 1]
                else:
                    churn_probability = model.predict(profile_df)[0]
            
            # Display prediction
            st.header("Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Churn Probability", 
                    f"{churn_probability*100:.1f}%",
                    delta=f"{(churn_probability - 0.5)*100:.1f}%" if churn_probability != 0.5 else None,
                    delta_color="inverse"
                )
            
            with col2:
                # Determine risk category
                if churn_probability < 0.3:
                    risk_category = "Low Risk"
                    color = "green"
                elif churn_probability < 0.6:
                    risk_category = "Medium Risk"
                    color = "orange"
                else:
                    risk_category = "High Risk"
                    color = "red"
                
                st.markdown(f"<h3 style='color: {color};'>Risk Category: {risk_category}</h3>", unsafe_allow_html=True)
            
            # Display risk factors if high risk
            if churn_probability >= 0.6 and hasattr(model, 'feature_importances_'):
                st.subheader("Key Risk Factors")
                st.write("The following factors contribute most to this customer's churn risk:")
                
                # This is a simplified approach - in a real application, you would use
                # SHAP values or other methods to determine feature contributions
                st.info("Note: This is a simplified analysis. For more accurate feature contributions, consider using SHAP values.")
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("The What-If Analysis feature is still under development. Some combinations of features may not work correctly.")

def main():
    """
    Main function to run the dashboard.
    """
    # Create sidebar and get selected page and filters
    selected_page, filters = create_sidebar()
    
    # Display selected page
    if selected_page == "Overview":
        overview_page(filters)
    elif selected_page == "Customer Analysis":
        customer_analysis_page(filters)
    elif selected_page == "Churn Prediction":
        churn_prediction_page(filters)
    elif selected_page == "What-If Analysis":
        what_if_analysis_page(filters)

if __name__ == "__main__":
    main()
