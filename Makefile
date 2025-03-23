# Telco Customer Churn Data Engineering Project Makefile

.PHONY: setup check ingest features train dashboard pipeline clean

# Setup the project
setup:
	python checkrequirements.py

# Check requirements
check:
	python main.py check-requirements

# Ingest data
ingest:
	python main.py ingest

# Engineer features
features:
	python main.py engineer-features

# Train model
train:
	python main.py train

# Run dashboard
dashboard:
	python main.py dashboard

# Run full pipeline
pipeline:
	python main.py pipeline

# Clean generated files
clean:
	rm -rf data/processed/*
	rm -rf data/feature_store/*
	rm -rf data/model_data/*
	rm -rf logs/*
	rm -rf mlruns/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name "*.ipynb_checkpoints" -exec rm -rf {} +

# Install the package
install:
	pip install -e .

# Run tests
test:
	pytest

# Generate documentation
docs:
	mkdir -p docs
	pdoc --html --output-dir docs src

# Help
help:
	@echo "Available commands:"
	@echo "  make setup      - Setup the project"
	@echo "  make check      - Check requirements"
	@echo "  make ingest     - Ingest data"
	@echo "  make features   - Engineer features"
	@echo "  make train      - Train model"
	@echo "  make dashboard  - Run dashboard"
	@echo "  make pipeline   - Run full pipeline"
	@echo "  make clean      - Clean generated files"
	@echo "  make install    - Install the package"
	@echo "  make test       - Run tests"
	@echo "  make docs       - Generate documentation"
	@echo "  make help       - Show this help message"
