# Telco Customer Churn Data Engineering Project

A comprehensive data engineering solution for analyzing and predicting customer churn in the telecommunications industry, built with modern data engineering principles and tools.

## 🔍 Project Overview

This project demonstrates an end-to-end data engineering pipeline for customer churn prediction, showcasing:

- **Data Ingestion Pipeline**: Flexible data loading from multiple sources
- **ETL Processes**: Robust data transformation and feature engineering
- **Data Warehousing**: Structured storage for processed data and features
- **Model Training Pipeline**: Automated model building and evaluation
- **Interactive Dashboard**: Data visualization and prediction interface
- **Experiment Tracking**: MLflow integration for model versioning and metrics

## 🛠️ Data Engineering Components

### Data Ingestion Layer
- Multi-source data extraction (CSV, SQL databases)
- Configurable data connectors
- Data validation and quality checks
- Incremental loading capability

### Data Processing Layer
- Automated feature type detection
- Scikit-learn transformation pipelines
- Advanced feature engineering
- Data partitioning for training/testing

### Data Storage Layer
- Feature store implementation
- Model registry integration
- Metrics storage and versioning
- Configuration management

### Orchestration Layer
- Modular pipeline architecture
- Configuration-driven execution
- Error handling and logging
- Dependency management

### Visualization Layer
- Interactive Streamlit dashboard
- Real-time prediction capabilities
- Data exploration tools
- What-if analysis for scenario testing

## 📊 Dashboard Sections

### Overview Dashboard
The Overview dashboard provides high-level insights into customer churn patterns:

- **Key Metrics Panel**: Displays total customers, churn rate, average tenure, and monthly charges
- **Churn Distribution by Contract**: Visualizes how contract type affects churn rates
- **Churn by Tenure**: Shows relationship between customer tenure and likelihood to churn
- **Service Impact Analysis**: Identifies which services have the highest impact on churn

**Data Engineering Aspect**: Aggregates data from the feature store and implements real-time filtering capabilities, demonstrating effective data transformation for business intelligence.

### Customer Analysis Dashboard
The Customer Analysis dashboard offers deep insights into customer demographics and behavior:

- **Demographic Breakdown**: Visualizes customer distribution by gender and age
- **Tenure Analysis**: Histograms and scatter plots showing customer longevity patterns
- **Service Adoption**: Shows which services are most popular among different customer segments
- **Payment Analysis**: Breaks down payment methods and billing preferences
- **Correlation Matrix**: Identifies relationships between different customer attributes

**Data Engineering Aspect**: Implements complex data transformations and joins between multiple data sources to create a unified view of customer behavior.

### Churn Prediction Dashboard
The Churn Prediction dashboard showcases the machine learning model's performance:

- **Model Performance Metrics**: Displays accuracy, precision, recall, and ROC AUC
- **Confusion Matrix**: Visualizes prediction accuracy
- **Feature Importance**: Shows which factors most strongly influence churn
- **Risk Distribution**: Segments customers by churn risk level
- **High-Risk Customer List**: Identifies specific customers with high churn probability

**Data Engineering Aspect**: Integrates the model pipeline with the data pipeline, demonstrating how to operationalize machine learning models in a data engineering context.

### What-If Analysis Dashboard
The What-If Analysis dashboard allows interactive exploration of churn factors:

- **Customer Profile Simulator**: Adjustable parameters for different customer attributes
- **Real-time Prediction**: Instant churn probability calculation
- **Risk Categorization**: Visual indication of risk level
- **Key Risk Factors**: Highlights the main contributors to churn risk

**Data Engineering Aspect**: Showcases real-time data processing and model inference, demonstrating how data pipelines can support interactive applications.

## 🚀 Installation and Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd telco-churn-project
   ```

2. Check and install requirements:
   ```
   python checkrequirements.py
   ```

## 📋 Project Structure

```
.
├── config/                  # Configuration management
│   └── config.yaml          # Centralized configuration
├── data/                    # Data storage layer
│   ├── raw/                 # Raw data storage
│   ├── processed/           # Processed data
│   ├── feature_store/       # Feature repository
│   └── model_data/          # Model artifacts
├── logs/                    # Logging and monitoring
├── mlruns/                  # MLflow experiment tracking
├── src/                     # Source code
│   ├── data/                # Data pipeline components
│   │   ├── ingestion.py     # Data extraction module
│   │   └── feature_engineering.py  # Feature transformation
│   ├── models/              # Model pipeline components
│   │   └── train.py         # Model training and evaluation
│   ├── dashboard/           # Visualization layer
│   │   └── app.py           # Streamlit dashboard
│   ├── pipelines/           # Pipeline orchestration
│   ├── tests/               # Testing framework
│   └── utils/               # Utility functions
│       └── config_loader.py # Configuration utilities
├── checkrequirements.py     # Dependency management
├── main.py                  # Entry point
├── Makefile                 # Build automation
├── requirements.txt         # Project dependencies
└── setup.py                 # Package configuration
```

## 🔄 Data Flow Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Data       │     │  Feature    │     │  Model      │     │  Dashboard  │
│  Ingestion  │────▶│  Engineering│────▶│  Training   │────▶│  Visualization│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw Data   │     │  Feature    │     │  Model      │     │  User       │
│  Storage    │     │  Store      │     │  Registry   │     │  Interface  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## 🖥️ Usage

### Running the Complete Pipeline

Execute the full data engineering pipeline:

```
python main.py
```

This will:
1. Ingest data from the configured source
2. Process and engineer features
3. Train and evaluate a machine learning model
4. Save all artifacts to their respective locations

### Running the Dashboard

Launch the interactive dashboard:

```
cd src/dashboard
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

## 🔧 Technologies Used

### Data Engineering Tools
- **Pandas & NumPy**: Data manipulation and transformation
- **SQLAlchemy**: Database abstraction and SQL generation
- **Scikit-learn Pipelines**: Reproducible data transformations
- **MLflow**: Experiment tracking and model registry
- **Streamlit**: Interactive data applications

### Machine Learning Components
- **Scikit-learn**: Feature preprocessing and model evaluation
- **XGBoost & LightGBM**: High-performance gradient boosting
- **SHAP**: Model explainability

### Visualization Libraries
- **Plotly**: Interactive data visualizations
- **Matplotlib & Seaborn**: Statistical visualizations

## 🌟 Data Engineering Best Practices Implemented

- **Modular Architecture**: Clear separation of concerns between components
- **Configuration as Code**: Externalized configuration for flexibility
- **Pipeline Automation**: End-to-end workflow automation
- **Data Lineage**: Tracking of data transformations
- **Feature Store**: Centralized repository for feature management
- **Model Versioning**: Tracking of model iterations and performance
- **Reproducibility**: Consistent environment and execution
- **Monitoring**: Logging and performance tracking

## 📈 Future Enhancements

- **Real-time Data Ingestion**: Integration with Apache Kafka or AWS Kinesis
- **Automated Retraining**: Implementation with Apache Airflow
- **Cloud Deployment**: Containerization and deployment to AWS/GCP/Azure
- **Advanced Feature Store**: Integration with Feature Store frameworks
- **A/B Testing Framework**: For evaluating retention strategies
- **Data Quality Monitoring**: Automated checks and alerts

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
