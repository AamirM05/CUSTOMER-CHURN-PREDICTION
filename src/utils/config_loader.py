"""
Configuration loader utility for the Telco Customer Churn project.
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

def get_database_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get database configuration with environment variable substitution.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Database configuration dictionary with passwords from environment variables
    """
    db_config = config.get('database', {})
    db_type = db_config.get('type', 'sqlite')
    
    if db_type == 'postgres' and 'DB_PASSWORD' in os.environ:
        db_config['postgres']['password'] = os.environ['DB_PASSWORD']
    elif db_type == 'mysql' and 'DB_PASSWORD' in os.environ:
        db_config['mysql']['password'] = os.environ['DB_PASSWORD']
    
    return db_config

def get_data_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get data paths from configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary of data paths
    """
    return config.get('data', {})

def get_processing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get data processing configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Data processing configuration dictionary
    """
    return config.get('processing', {})

def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get model training configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Model training configuration dictionary
    """
    return config.get('model', {})

def get_mlflow_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get MLflow tracking configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        MLflow configuration dictionary
    """
    return config.get('mlflow', {})

def get_dashboard_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get dashboard configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dashboard configuration dictionary
    """
    return config.get('dashboard', {})

def get_airflow_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get Airflow configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Airflow configuration dictionary
    """
    return config.get('airflow', {})

def get_logging_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get logging configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Logging configuration dictionary
    """
    return config.get('logging', {})

def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Set up logging based on configuration.
    
    Args:
        config: Configuration dictionary (if None, will load from default path)
    """
    if config is None:
        config = load_config()
    
    log_config = get_logging_config(config)
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', 'logs/telco_churn.log')
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Logging configured")

if __name__ == "__main__":
    # Example usage
    setup_logging()
    config = load_config()
    print("Configuration loaded successfully")
    print(f"Database type: {get_database_config(config)['type']}")
    print(f"Raw data path: {get_data_paths(config)['raw_data_path']}")
    print(f"Model algorithm: {get_model_config(config)['algorithm']}")
