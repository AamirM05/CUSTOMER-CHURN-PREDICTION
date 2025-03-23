"""
Feature engineering module for the Telco Customer Churn project.

This module handles transforming raw data into features suitable for machine learning.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.utils.config_loader import load_config, get_processing_config, get_data_paths

# Configure logging
logger = logging.getLogger(__name__)

def identify_feature_types(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify categorical, numerical, and binary features.
    
    Args:
        df: Input DataFrame
        config: Processing configuration
        
    Returns:
        Tuple of (categorical_features, numerical_features, binary_features)
    """
    processing_config = get_processing_config(config)
    
    # Get features from config if available
    config_categorical = processing_config.get('categorical_features', [])
    config_numerical = processing_config.get('numerical_features', [])
    
    # Convert config feature names to lowercase to match DataFrame column names
    categorical_features = [col.lower() for col in config_categorical]
    numerical_features = [col.lower() for col in config_numerical]
    
    # Check if all specified features exist in the DataFrame
    all_features_exist = True
    for col in categorical_features + numerical_features:
        if col not in df.columns:
            logger.warning(f"Column '{col}' specified in config not found in DataFrame")
            all_features_exist = False
    
    # If not all specified features exist or none specified, infer from data
    if not all_features_exist or (not categorical_features and not numerical_features):
        categorical_features = []
        numerical_features = []
        
        for col in df.columns:
            # Skip target and ID columns
            target = processing_config.get('target', 'churn').lower()
            id_column = processing_config.get('id_column', 'customerid').lower()
            
            if col == target or col == id_column:
                continue
                
            if df[col].dtype == 'object' or df[col].dtype == 'bool' or df[col].nunique() < 10:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
    
    # Identify binary features (subset of categorical features)
    binary_features = [col for col in categorical_features if col in df.columns and df[col].nunique() <= 2]
    
    # Remove binary features from categorical features
    categorical_features = [col for col in categorical_features if col not in binary_features]
    
    logger.info(f"Identified {len(categorical_features)} categorical features, {len(numerical_features)} numerical features, and {len(binary_features)} binary features")
    
    return categorical_features, numerical_features, binary_features

def create_preprocessing_pipeline(
    categorical_features: List[str],
    numerical_features: List[str],
    binary_features: List[str],
    config: Dict[str, Any]
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for feature engineering.
    
    Args:
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        binary_features: List of binary feature names
        config: Processing configuration
        
    Returns:
        ColumnTransformer preprocessing pipeline
    """
    processing_config = get_processing_config(config)
    
    # Numerical features pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])
    
    # Add scaling if specified
    scaling_method = processing_config.get('feature_scaling', 'standard')
    if scaling_method == 'standard':
        numerical_pipeline.steps.append(('scaler', StandardScaler()))
    elif scaling_method == 'minmax':
        numerical_pipeline.steps.append(('scaler', MinMaxScaler()))
    elif scaling_method == 'robust':
        numerical_pipeline.steps.append(('scaler', RobustScaler()))
    
    # Categorical features pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Binary features pipeline
    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    
    # Combine all pipelines
    transformers = []
    
    if numerical_features:
        transformers.append(('numerical', numerical_pipeline, numerical_features))
    
    if categorical_features:
        transformers.append(('categorical', categorical_pipeline, categorical_features))
    
    if binary_features:
        transformers.append(('binary', binary_pipeline, binary_features))
    
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    
    logger.info("Created preprocessing pipeline")
    return preprocessor

def engineer_features(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Engineer features from raw data.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (DataFrame with engineered features, preprocessing pipeline)
    """
    processing_config = get_processing_config(config)
    target_name = processing_config.get('target', 'churn').lower()
    id_column_name = processing_config.get('id_column', 'customerid').lower()
    
    # Find actual column names for target and ID (case-insensitive)
    target_col = next((col for col in df.columns if col.lower() == target_name), None)
    id_col = next((col for col in df.columns if col.lower() == id_column_name), None)
    
    # Identify feature types
    categorical_features, numerical_features, binary_features = identify_feature_types(df, config)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features, binary_features, config)
    
    # Extract features and target
    columns_to_drop = []
    if target_col:
        columns_to_drop.append(target_col)
    if id_col:
        columns_to_drop.append(id_col)
    
    X = df.drop(columns_to_drop, axis=1, errors='ignore')
    
    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(X)
    
    # Get feature names
    feature_names = []
    
    # Get numerical feature names
    if numerical_features:
        feature_names.extend(numerical_features)
    
    # Get one-hot encoded feature names
    if categorical_features and any(name == 'categorical' for name, _, _ in preprocessor.transformers_):
        categorical_idx = [i for i, (name, _, _) in enumerate(preprocessor.transformers_) if name == 'categorical'][0]
        categorical_transformer = preprocessor.transformers_[categorical_idx][1]
        encoder = categorical_transformer.named_steps['encoder']
        
        # Get all categories
        categories = encoder.categories_
        
        # Create feature names for each category
        for i, feature in enumerate(categorical_features):
            for category in categories[i]:
                feature_names.append(f"{feature}_{category}")
    
    # Get binary feature names
    if binary_features:
        feature_names.extend(binary_features)
    
    # Create DataFrame with transformed features
    X_engineered = pd.DataFrame(X_transformed, columns=feature_names, index=df.index)
    
    # Add target and ID columns back
    if target_col:
        X_engineered[target_name] = df[target_col]
    
    if id_col:
        X_engineered[id_column_name] = df[id_col]
    
    logger.info(f"Engineered {X_engineered.shape[1]} features from {df.shape[1]} original columns")
    
    return X_engineered, preprocessor

def create_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features if tenure column exists.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional time-based features
    """
    df_with_time_features = df.copy()
    
    # Find tenure column (case-insensitive)
    tenure_col = next((col for col in df.columns if col.lower() == 'tenure'), None)
    
    if tenure_col:
        # Create tenure-based features
        df_with_time_features['tenure_years'] = df[tenure_col] / 12
        df_with_time_features['tenure_squared'] = df[tenure_col] ** 2
        
        # Create tenure bins
        df_with_time_features['tenure_group'] = pd.cut(
            df[tenure_col], 
            bins=[0, 12, 24, 36, 48, 60, float('inf')],
            labels=['0-1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5+ years']
        )
        
        logger.info("Created time-based features from tenure column")
    
    return df_with_time_features

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between important columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional interaction features
    """
    df_with_interactions = df.copy()
    
    # Check if required columns exist (case-insensitive)
    df_columns_lower = [col.lower() for col in df.columns]
    
    # Find actual column names for monthlycharges and tenure
    monthlycharges_col = next((col for col in df.columns if col.lower() == 'monthlycharges'), None)
    tenure_col = next((col for col in df.columns if col.lower() == 'tenure'), None)
    totalcharges_col = next((col for col in df.columns if col.lower() == 'totalcharges'), None)
    
    # Create price per tenure interaction
    if monthlycharges_col and tenure_col:
        df_with_interactions['price_per_tenure'] = df[monthlycharges_col] / (df[tenure_col] + 1)  # Add 1 to avoid division by zero
    
    # Create ratio of total to monthly charges
    if monthlycharges_col and totalcharges_col:
        df_with_interactions['total_to_monthly_ratio'] = df[totalcharges_col] / (df[monthlycharges_col] + 1)  # Add 1 to avoid division by zero
    
    # Define service column names (both original and lowercase versions)
    service_column_names = [
        'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity',
        'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies'
    ]
    
    # Find actual column names for service columns
    service_columns = []
    for service_name in service_column_names:
        col = next((col for col in df.columns if col.lower() == service_name), None)
        if col:
            service_columns.append(col)
    
    # Create service count feature if enough service columns exist
    if len(service_columns) >= 5:  # Require at least 5 service columns to create service count
        # Convert Yes/No to 1/0 if needed
        for col in service_columns:
            if df[col].dtype == 'object':
                # Handle different formats of Yes/No values
                if set(df[col].dropna().unique()) == {'Yes', 'No'}:
                    df_with_interactions[col] = df[col].map({'Yes': 1, 'No': 0})
                elif set(df[col].dropna().unique()) == {'yes', 'no'}:
                    df_with_interactions[col] = df[col].map({'yes': 1, 'no': 0})
                elif set(df[col].dropna().unique()) == {'YES', 'NO'}:
                    df_with_interactions[col] = df[col].map({'YES': 1, 'NO': 0})
                elif set(df[col].dropna().unique()) == {'Y', 'N'}:
                    df_with_interactions[col] = df[col].map({'Y': 1, 'N': 0})
                elif set(df[col].dropna().unique()) == {'y', 'n'}:
                    df_with_interactions[col] = df[col].map({'y': 1, 'n': 0})
                else:
                    # Try to convert to numeric, replacing errors with 0
                    df_with_interactions[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Count number of services
        df_with_interactions['service_count'] = df_with_interactions[service_columns].sum(axis=1)
        
        # Create service density (services per tenure)
        if tenure_col:
            df_with_interactions['service_density'] = df_with_interactions['service_count'] / (df[tenure_col] + 1)
    
    logger.info("Created interaction features")
    
    return df_with_interactions

def save_engineered_features(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Save engineered features to disk.
    
    Args:
        df: DataFrame with engineered features
        config: Configuration dictionary
    """
    data_paths = get_data_paths(config)
    feature_store_path = data_paths.get('feature_store_path', 'data/feature_store/')
    
    # Create directory if it doesn't exist
    os.makedirs(feature_store_path, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(feature_store_path, 'engineered_features.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved engineered features to {output_path}")

def engineer_all_features(df: pd.DataFrame, config_path: str = "config/config.yaml") -> pd.DataFrame:
    """
    Main function to engineer all features.
    
    Args:
        df: Input DataFrame
        config_path: Path to configuration file
        
    Returns:
        DataFrame with all engineered features
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create time-based features
    df_with_time = create_time_based_features(df)
    
    # Create interaction features
    df_with_interactions = create_interaction_features(df_with_time)
    
    # Engineer features using preprocessing pipeline
    df_engineered, _ = engineer_features(df_with_interactions, config)
    
    # Save engineered features
    save_engineered_features(df_engineered, config)
    
    return df_engineered

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = load_config()
    
    # Import here to avoid circular imports
    from src.data.ingestion import ingest_data
    
    # Ingest data
    df, _ = ingest_data()
    
    # Engineer features
    df_engineered = engineer_all_features(df)
    
    # Print summary
    print(f"Feature engineering completed. Created {df_engineered.shape[1]} features from {df.shape[1]} original columns.")
    print(f"Sample of engineered features:\n{df_engineered.head()}")
