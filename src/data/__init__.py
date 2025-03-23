"""
Data handling modules for the Telco Customer Churn project.
"""

from src.data.ingestion import (
    read_csv_data,
    clean_column_names,
    preprocess_data,
    get_db_engine,
    load_data_to_db,
    check_table_exists,
    ingest_data
)

from src.data.feature_engineering import (
    identify_feature_types,
    create_preprocessing_pipeline,
    engineer_features,
    create_time_based_features,
    create_interaction_features,
    save_engineered_features,
    engineer_all_features
)

__all__ = [
    # Ingestion
    'read_csv_data',
    'clean_column_names',
    'preprocess_data',
    'get_db_engine',
    'load_data_to_db',
    'check_table_exists',
    'ingest_data',
    
    # Feature Engineering
    'identify_feature_types',
    'create_preprocessing_pipeline',
    'engineer_features',
    'create_time_based_features',
    'create_interaction_features',
    'save_engineered_features',
    'engineer_all_features'
]
