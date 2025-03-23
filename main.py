#!/usr/bin/env python
"""
Main entry point for the Telco Customer Churn project.

This script provides a command-line interface to run different components of the project.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'telco_churn_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_argparse():
    """
    Set up command-line argument parser.
    
    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Telco Customer Churn Data Engineering Project',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Check requirements command
    check_parser = subparsers.add_parser('check-requirements', help='Check and install required packages')
    
    # Ingest data command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest data from CSV to database')
    ingest_parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    
    # Feature engineering command
    feature_parser = subparsers.add_parser('engineer-features', help='Engineer features from raw data')
    feature_parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train a machine learning model')
    train_parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    
    # Run dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run the dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8501, help='Port to run the dashboard on')
    
    # Run full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the full data pipeline')
    pipeline_parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    
    return parser

def check_requirements():
    """
    Check and install required packages.
    """
    logger.info("Checking requirements...")
    
    # Import here to avoid circular imports
    import subprocess
    
    # Run checkrequirements.py
    subprocess.check_call([sys.executable, 'checkrequirements.py'])
    
    logger.info("Requirements check completed.")

def ingest_data(config_path):
    """
    Ingest data from CSV to database.
    
    Args:
        config_path: Path to configuration file
    """
    logger.info("Ingesting data...")
    
    # Import here to avoid circular imports
    from src.data.ingestion import ingest_data
    
    # Run ingestion
    df, engine = ingest_data(config_path)
    
    logger.info(f"Data ingestion completed. Loaded {len(df)} rows to database.")
    return df, engine

def engineer_features(config_path):
    """
    Engineer features from raw data.
    
    Args:
        config_path: Path to configuration file
    """
    logger.info("Engineering features...")
    
    # Import here to avoid circular imports
    from src.data.ingestion import ingest_data
    from src.data.feature_engineering import engineer_all_features
    
    # First ingest data
    df, _ = ingest_data(config_path)
    
    # Then engineer features
    df_engineered = engineer_all_features(df, config_path)
    
    logger.info(f"Feature engineering completed. Created {df_engineered.shape[1]} features from {df.shape[1]} original columns.")
    return df_engineered

def train_model(config_path):
    """
    Train a machine learning model.
    
    Args:
        config_path: Path to configuration file
    """
    logger.info("Training model...")
    
    # Import here to avoid circular imports
    from src.models.train import train_and_evaluate_model
    
    # Train and evaluate model
    result = train_and_evaluate_model(config_path)
    
    logger.info(f"Model training completed. Model saved to {result['model_path']}.")
    return result

def run_dashboard(port):
    """
    Run the dashboard.
    
    Args:
        port: Port to run the dashboard on
    """
    logger.info(f"Starting dashboard on port {port}...")
    
    # Import here to avoid circular imports
    import subprocess
    
    # Run streamlit app
    subprocess.check_call([
        'streamlit', 'run', 
        os.path.join('src', 'dashboard', 'app.py'),
        '--server.port', str(port)
    ])

def run_pipeline(config_path):
    """
    Run the full data pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    logger.info("Running full pipeline...")
    
    # Check requirements
    check_requirements()
    
    # Ingest data
    df, _ = ingest_data(config_path)
    
    # Engineer features
    df_engineered = engineer_features(config_path)
    
    # Train model
    result = train_model(config_path)
    
    logger.info("Pipeline completed successfully.")
    
    # Print summary
    print("\nPipeline Summary:")
    print(f"- Processed {len(df)} rows of data")
    print(f"- Created {df_engineered.shape[1]} features")
    print(f"- Trained {type(result['model']).__name__} model")
    print(f"- Model accuracy: {result['metrics']['test']['accuracy']:.4f}")
    print(f"- Model saved to: {result['model_path']}")
    print(f"- MLflow Run ID: {result['mlflow_run_id']}")
    
    # Ask if user wants to run dashboard
    response = input("\nDo you want to run the dashboard now? (y/n): ")
    if response.lower() in ['y', 'yes']:
        run_dashboard(8501)

def main():
    """
    Main function to parse arguments and run commands.
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up argument parser
    parser = setup_argparse()
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run command
    if args.command == 'check-requirements':
        check_requirements()
    elif args.command == 'ingest':
        ingest_data(args.config)
    elif args.command == 'engineer-features':
        engineer_features(args.config)
    elif args.command == 'train':
        train_model(args.config)
    elif args.command == 'dashboard':
        run_dashboard(args.port)
    elif args.command == 'pipeline':
        run_pipeline(args.config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
