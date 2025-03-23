"""
Data ingestion module for the Telco Customer Churn project.

This module handles reading data from CSV files and loading it into a database.
"""

import os
import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine, inspect, MetaData, Table, Column, String, Float, Integer, Boolean
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Any, Optional, Tuple

from src.utils.config_loader import load_config, get_database_config, get_data_paths

# Configure logging
logger = logging.getLogger(__name__)

def read_csv_data(file_path: str) -> pd.DataFrame:
    """
    Read data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    try:
        logger.info(f"Reading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names to ensure they are database-friendly.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic preprocessing on the data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Clean column names
    df_processed = clean_column_names(df_processed)
    
    # Convert 'Yes'/'No' to boolean
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            if set(df_processed[col].dropna().unique()).issubset({'Yes', 'No'}):
                df_processed[col] = df_processed[col].map({'Yes': True, 'No': False})
    
    # Convert SeniorCitizen from 0/1 to boolean if it exists
    if 'seniorcitizen' in df_processed.columns:
        df_processed['seniorcitizen'] = df_processed['seniorcitizen'].astype(bool)
    
    # Handle TotalCharges - convert to numeric
    if 'totalcharges' in df_processed.columns:
        df_processed['totalcharges'] = pd.to_numeric(df_processed['totalcharges'], errors='coerce')
    
    logger.info("Data preprocessing completed")
    return df_processed

def get_db_engine(db_config: Dict[str, Any]) -> Any:
    """
    Create a SQLAlchemy engine based on database configuration.
    
    Args:
        db_config: Database configuration dictionary
        
    Returns:
        SQLAlchemy engine
    """
    db_type = db_config.get('type', 'sqlite')
    
    if db_type == 'sqlite':
        db_path = db_config.get('sqlite', {}).get('path', 'data/telco_churn.db')
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        connection_string = f"sqlite:///{db_path}"
    elif db_type == 'postgres':
        pg_config = db_config.get('postgres', {})
        host = pg_config.get('host', 'localhost')
        port = pg_config.get('port', 5432)
        database = pg_config.get('database', 'telco_churn')
        user = pg_config.get('user', 'postgres')
        password = pg_config.get('password', '')
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    elif db_type == 'mysql':
        mysql_config = db_config.get('mysql', {})
        host = mysql_config.get('host', 'localhost')
        port = mysql_config.get('port', 3306)
        database = mysql_config.get('database', 'telco_churn')
        user = mysql_config.get('user', 'root')
        password = mysql_config.get('password', '')
        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    logger.info(f"Creating database engine for {db_type}")
    return create_engine(connection_string)

def load_data_to_db(df: pd.DataFrame, table_name: str, engine: Any, if_exists: str = 'replace') -> None:
    """
    Load DataFrame to database.
    
    Args:
        df: DataFrame to load
        table_name: Name of the table
        engine: SQLAlchemy engine
        if_exists: How to behave if the table exists ('fail', 'replace', or 'append')
    """
    try:
        logger.info(f"Loading {len(df)} rows to table '{table_name}'")
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        logger.info(f"Successfully loaded data to table '{table_name}'")
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data to database: {e}")
        raise

def check_table_exists(table_name: str, engine: Any) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        table_name: Name of the table
        engine: SQLAlchemy engine
        
    Returns:
        True if the table exists, False otherwise
    """
    inspector = inspect(engine)
    return inspector.has_table(table_name)

def ingest_data(config_path: str = "config/config.yaml") -> Tuple[pd.DataFrame, Any]:
    """
    Main function to ingest data from CSV to database.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (DataFrame, SQLAlchemy engine)
    """
    # Load configuration
    config = load_config(config_path)
    db_config = get_database_config(config)
    data_paths = get_data_paths(config)
    
    # Get raw data path
    raw_data_path = data_paths.get('raw_data_path', "data/raw/Telco Customer Churn Dataset.csv")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    
    # Check if we need to copy the CSV file to the raw data directory
    if not os.path.exists(raw_data_path):
        original_csv = "Telco Customer Churn Dataset.csv"
        if os.path.exists(original_csv):
            import shutil
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
            shutil.copy(original_csv, raw_data_path)
            logger.info(f"Copied {original_csv} to {raw_data_path}")
        else:
            logger.warning(f"Original CSV file {original_csv} not found")
    
    # Read and preprocess data
    df = read_csv_data(raw_data_path)
    df_processed = preprocess_data(df)
    
    # Create database engine and load data
    engine = get_db_engine(db_config)
    load_data_to_db(df_processed, 'telco_churn', engine)
    
    return df_processed, engine

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run ingestion
    df, engine = ingest_data()
    
    # Print summary
    print(f"Data ingestion completed. Loaded {len(df)} rows to database.")
    print(f"Database type: {engine.name}")
    print(f"Tables: {engine.table_names()}")
