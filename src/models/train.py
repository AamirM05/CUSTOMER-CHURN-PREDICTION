"""
Model training module for the Telco Customer Churn project.

This module handles training machine learning models for churn prediction.
"""

import os
import pandas as pd
import numpy as np
import logging
import pickle
import json
from typing import Dict, Any, List, Tuple, Optional, Union
import time
from datetime import datetime

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import shap

from src.utils.config_loader import (
    load_config, get_model_config, get_mlflow_config, 
    get_data_paths, get_processing_config
)

# Configure logging
logger = logging.getLogger(__name__)

def load_model_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load data for model training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    data_paths = get_data_paths(config)
    processing_config = get_processing_config(config)
    
    # Get paths
    feature_store_path = data_paths.get('feature_store_path', 'data/feature_store/')
    features_file = os.path.join(feature_store_path, 'engineered_features.csv')
    
    # Get target and test size
    target_name = processing_config.get('target', 'churn').lower()
    id_column_name = processing_config.get('id_column', 'customerid').lower()
    test_size = processing_config.get('test_size', 0.2)
    random_state = processing_config.get('random_state', 42)
    
    # Load data
    try:
        df = pd.read_csv(features_file)
        logger.info(f"Loaded {len(df)} rows from {features_file}")
    except FileNotFoundError:
        logger.error(f"Features file not found: {features_file}")
        raise
    
    # Split features and target
    X = df.drop([target_name, id_column_name], axis=1, errors='ignore')
    y = df[target_name]
    
    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Split data into {len(X_train)} training samples and {len(X_test)} test samples")
    
    return X_train, y_train, X_test, y_test

def get_model_instance(config: Dict[str, Any]) -> Any:
    """
    Get a model instance based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_config = get_model_config(config)
    algorithm = model_config.get('algorithm', 'xgboost')
    hyperparameters = model_config.get('hyperparameters', {})
    
    if algorithm == 'logistic':
        params = hyperparameters.get('logistic', {})
        model = LogisticRegression(
            C=params.get('C', 1.0),
            penalty=params.get('penalty', 'l2'),
            solver=params.get('solver', 'liblinear'),
            random_state=model_config.get('random_state', 42),
            max_iter=1000
        )
    elif algorithm == 'random_forest':
        params = hyperparameters.get('random_forest', {})
        model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            random_state=model_config.get('random_state', 42)
        )
    elif algorithm == 'xgboost':
        params = hyperparameters.get('xgboost', {})
        model = xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 5),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            random_state=model_config.get('random_state', 42)
        )
    elif algorithm == 'lightgbm':
        params = hyperparameters.get('lightgbm', {})
        model = lgb.LGBMClassifier(
            n_estimators=params.get('n_estimators', 100),
            num_leaves=params.get('num_leaves', 31),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=model_config.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    logger.info(f"Created {algorithm} model instance")
    return model

def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, config: Dict[str, Any]) -> Any:
    """
    Train a model on the training data.
    
    Args:
        model: Model instance
        X_train: Training features
        y_train: Training target
        config: Configuration dictionary
        
    Returns:
        Trained model
    """
    logger.info(f"Training {type(model).__name__} model on {len(X_train)} samples")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    return model

def evaluate_model(
    model: Any, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        config: Configuration dictionary
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {type(model).__name__} model")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get probability predictions if available
    if hasattr(model, 'predict_proba'):
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_train_prob = y_train_pred
        y_test_prob = y_test_pred
    
    # Calculate metrics
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_train, y_train_prob),
            'confusion_matrix': confusion_matrix(y_train, y_train_pred).tolist()
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_test_prob),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
        }
    }
    
    # Cross-validation if specified
    model_config = get_model_config(config)
    cv_config = model_config.get('cross_validation', {})
    if cv_config:
        cv_method = cv_config.get('method', 'stratified_kfold')
        n_splits = cv_config.get('n_splits', 5)
        
        if cv_method == 'stratified_kfold':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = n_splits
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        metrics['cross_validation'] = {
            'method': cv_method,
            'n_splits': n_splits,
            'scores': cv_scores.tolist(),
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std()
        }
    
    # Log metrics
    for dataset, dataset_metrics in metrics.items():
        if dataset in ['train', 'test']:
            logger.info(f"{dataset.capitalize()} metrics:")
            for metric, value in dataset_metrics.items():
                if metric != 'confusion_matrix':
                    logger.info(f"  {metric}: {value:.4f}")
    
    if 'cross_validation' in metrics:
        logger.info(f"Cross-validation mean score: {metrics['cross_validation']['mean_score']:.4f} ± {metrics['cross_validation']['std_score']:.4f}")
    
    return metrics

def get_feature_importance(model: Any, X: pd.DataFrame) -> Dict[str, float]:
    """
    Get feature importance from the model.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        
    Returns:
        Dictionary of feature importances
    """
    feature_names = X.columns.tolist()
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning(f"Model {type(model).__name__} does not provide feature importances")
        return {}
    
    # Create dictionary of feature importances
    feature_importance = dict(zip(feature_names, importances))
    
    # Sort by importance
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    
    return feature_importance

def calculate_shap_values(model: Any, X: pd.DataFrame) -> Dict[str, List[float]]:
    """
    Calculate SHAP values for feature importance.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        
    Returns:
        Dictionary of SHAP values
    """
    try:
        # Sample data for SHAP analysis (for large datasets)
        if len(X) > 1000:
            X_sample = X.sample(1000, random_state=42)
        else:
            X_sample = X
        
        # Create explainer based on model type
        if isinstance(model, xgb.XGBClassifier):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, lgb.LGBMClassifier):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_sample.iloc[:100])
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, shap_values might be a list with values for each class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use values for positive class
        
        # Calculate mean absolute SHAP value for each feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create dictionary of SHAP values
        shap_dict = dict(zip(X.columns, mean_shap))
        
        # Sort by importance
        shap_dict = {k: float(v) for k, v in sorted(shap_dict.items(), key=lambda item: item[1], reverse=True)}
        
        return shap_dict
    except Exception as e:
        logger.warning(f"Error calculating SHAP values: {e}")
        return {}

def save_model(model: Any, metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    Save the trained model and metrics.
    
    Args:
        model: Trained model
        metrics: Evaluation metrics
        config: Configuration dictionary
        
    Returns:
        Path to the saved model
    """
    data_paths = get_data_paths(config)
    model_data_path = data_paths.get('model_data_path', 'data/model_data/')
    
    # Create directory if it doesn't exist
    os.makedirs(model_data_path, exist_ok=True)
    
    # Create timestamp for model version
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_filename = f"model_{timestamp}.pkl"
    model_path = os.path.join(model_data_path, model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics_filename = f"metrics_{timestamp}.json"
    metrics_path = os.path.join(model_data_path, metrics_filename)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    return model_path

def log_to_mlflow(
    model: Any, 
    metrics: Dict[str, Any], 
    feature_importance: Dict[str, float],
    shap_values: Dict[str, float],
    config: Dict[str, Any]
) -> str:
    """
    Log model, metrics, and artifacts to MLflow.
    
    Args:
        model: Trained model
        metrics: Evaluation metrics
        feature_importance: Feature importance dictionary
        shap_values: SHAP values dictionary
        config: Configuration dictionary
        
    Returns:
        MLflow run ID
    """
    mlflow_config = get_mlflow_config(config)
    model_config = get_model_config(config)
    
    # Set up MLflow tracking
    tracking_uri = mlflow_config.get('tracking_uri', 'sqlite:///data/mlflow.db')
    experiment_name = mlflow_config.get('experiment_name', 'telco_churn_prediction')
    
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Error setting up MLflow experiment: {e}")
    
    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Log parameters
        algorithm = model_config.get('algorithm', 'xgboost')
        hyperparameters = model_config.get('hyperparameters', {}).get(algorithm, {})
        
        mlflow.log_param('algorithm', algorithm)
        for param, value in hyperparameters.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        for dataset, dataset_metrics in metrics.items():
            if dataset in ['train', 'test']:
                for metric, value in dataset_metrics.items():
                    if metric != 'confusion_matrix':
                        mlflow.log_metric(f"{dataset}_{metric}", value)
        
        if 'cross_validation' in metrics:
            mlflow.log_metric('cv_mean_score', metrics['cross_validation']['mean_score'])
            mlflow.log_metric('cv_std_score', metrics['cross_validation']['std_score'])
        
        # Log feature importance
        if feature_importance:
            # Create feature importance plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.barh(list(feature_importance.keys())[:20], list(feature_importance.values())[:20])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            
            # Save plot
            feature_importance_path = 'feature_importance.png'
            plt.savefig(feature_importance_path)
            plt.close()
            
            # Log artifact
            mlflow.log_artifact(feature_importance_path)
            os.remove(feature_importance_path)
            
            # Log feature importance as metrics
            for feature, importance in feature_importance.items():
                # Clean feature name for MLflow (remove special characters)
                clean_feature = feature.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
                mlflow.log_metric(f"importance_{clean_feature}", importance)
        
        # Log SHAP values
        if shap_values:
            # Create SHAP values plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.barh(list(shap_values.keys())[:20], list(shap_values.values())[:20])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Top 20 Features by SHAP Value')
            plt.tight_layout()
            
            # Save plot
            shap_values_path = 'shap_values.png'
            plt.savefig(shap_values_path)
            plt.close()
            
            # Log artifact
            mlflow.log_artifact(shap_values_path)
            os.remove(shap_values_path)
            
            # Log SHAP values as metrics
            for feature, value in shap_values.items():
                # Clean feature name for MLflow (remove special characters)
                clean_feature = feature.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
                mlflow.log_metric(f"shap_{clean_feature}", value)
        
        # Log model
        if isinstance(model, xgb.XGBClassifier):
            mlflow.xgboost.log_model(model, "model")
        elif isinstance(model, lgb.LGBMClassifier):
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Register model if specified
        if mlflow_config.get('register_model', False):
            model_name = mlflow_config.get('model_name', 'telco_churn_model')
            model_uri = f"runs:/{run_id}/model"
            
            try:
                registered_model = mlflow.register_model(model_uri, model_name)
                logger.info(f"Model registered as {model_name} with version {registered_model.version}")
            except Exception as e:
                logger.warning(f"Error registering model: {e}")
            
            logger.info(f"Model registered as {model_name}")
    
    logger.info(f"MLflow run completed with ID: {run_id}")
    return run_id

def train_and_evaluate_model(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Main function to train and evaluate a model.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with model, metrics, and paths
    """
    # Load configuration
    config = load_config(config_path)
    
    # Load data
    X_train, y_train, X_test, y_test = load_model_data(config)
    
    # Get model instance
    model = get_model_instance(config)
    
    # Train model
    model = train_model(model, X_train, y_train, config)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, config)
    
    # Get feature importance
    feature_importance = get_feature_importance(model, X_train)
    
    # Calculate SHAP values if specified
    model_config = get_model_config(config)
    if model_config.get('shap_analysis', False):
        shap_values = calculate_shap_values(model, X_train)
    else:
        shap_values = {}
    
    # Save model
    model_path = save_model(model, metrics, config)
    
    # Log to MLflow
    run_id = log_to_mlflow(model, metrics, feature_importance, shap_values, config)
    
    return {
        'model': model,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'shap_values': shap_values,
        'model_path': model_path,
        'mlflow_run_id': run_id
    }

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train and evaluate model
    result = train_and_evaluate_model()
    
    # Print summary
    print("\nModel Training Summary:")
    print(f"Algorithm: {type(result['model']).__name__}")
    print(f"Test Accuracy: {result['metrics']['test']['accuracy']:.4f}")
    print(f"Test ROC AUC: {result['metrics']['test']['roc_auc']:.4f}")
    print(f"Model saved to: {result['model_path']}")
    print(f"MLflow Run ID: {result['mlflow_run_id']}")
    
    # Print top features
    print("\nTop 10 Features:")
    for i, (feature, importance) in enumerate(list(result['feature_importance'].items())[:10]):
        print(f"{i+1}. {feature}: {importance:.4f}")
