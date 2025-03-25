#!/usr/bin/env python3
"""
SimDetector Extract and Train Workflow

This script:
1. Extracts features from all CSV files in the dataset/ directory
   - Processes each column individually
   - Also processes all columns together
   - Uses Kalman filter for noise extraction
2. Trains a model on each feature file
3. Saves the results in organized directories

Usage:
    python extract_train_workflow.py [--output-dir OUTPUT_DIR]
"""

import os
import logging
import argparse
import sys
import glob
from simdetector.features import process_csv_files
from simdetector.core import SIGNAL_TYPES
from simdetector.model import train_with_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simdetector_workflow')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract features and train models from dataset')
parser.add_argument('--output-dir', type=str, default='extraction',
                    help='Directory to store extracted features and models (default: extraction)')
args = parser.parse_args()

# Configuration
INPUT_DIR = "dataset"
OUTPUT_DIR = args.output_dir
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
FILTER_TYPE = "kalman"
WINDOW_SIZE = 20
ALL_COLUMNS_EPSILON = 0.3  # Higher epsilon for all-columns extraction
DEFAULT_EPSILON = 0.1      # Default epsilon for individual columns

def ensure_dirs_exist():
    """Create necessary directories if they don't exist."""
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"Created output directories: {FEATURES_DIR}, {MODELS_DIR}")

def get_csv_files():
    """Get all CSV files in the input directory."""
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {INPUT_DIR}")
        return []
    
    logger.info(f"Found {len(csv_files)} CSV files in {INPUT_DIR}")
    return csv_files

def get_all_columns():
    """Get all signal column types from SIGNAL_TYPES."""
    all_columns = []
    for category, columns in SIGNAL_TYPES.items():
        all_columns.extend(columns)
    return all_columns

def extract_features():
    """Extract features from all CSV files for each column type and all columns together."""
    columns = get_all_columns()
    feature_files = []
    
    # Process each column individually
    for column in columns:
        output_file = os.path.join(FEATURES_DIR, f"combined_{column}_features.csv")
        logger.info(f"Extracting features for column {column} with epsilon={DEFAULT_EPSILON}")
        
        try:
            success = process_csv_files(
                data_directory=INPUT_DIR,
                output_file=output_file,
                target_column=column,
                window_size=WINDOW_SIZE,
                extract_noise=True,
                filter_type=FILTER_TYPE,
                epsilon=DEFAULT_EPSILON
            )
            
            if success:
                logger.info(f"Successfully extracted features for {column} to {output_file}")
                feature_files.append(output_file)
            else:
                logger.error(f"Failed to extract features for {column}")
        except Exception as e:
            logger.error(f"Error extracting features for {column}: {str(e)}")
    
    # Process all columns together
    all_columns_output = os.path.join(FEATURES_DIR, "combined_all_columns_features.csv")
    logger.info(f"Extracting features for all columns with epsilon={ALL_COLUMNS_EPSILON}")
    
    try:
        success = process_csv_files(
            data_directory=INPUT_DIR,
            output_file=all_columns_output,
            target_column=None,
            window_size=WINDOW_SIZE,
            extract_noise=True,
            filter_type=FILTER_TYPE,
            epsilon=ALL_COLUMNS_EPSILON,
            all_columns=True
        )
        
        if success:
            logger.info(f"Successfully extracted features for all columns to {all_columns_output}")
            feature_files.append(all_columns_output)
        else:
            logger.error("Failed to extract features for all columns")
    except Exception as e:
        logger.error(f"Error extracting features for all columns: {str(e)}")
    
    return feature_files

def train_models(feature_files):
    """Train a model on each feature file."""
    trained_models = []
    
    for feature_file in feature_files:
        # Extract column name from feature file
        basename = os.path.basename(feature_file)
        match = basename.replace("combined_", "").replace("_features.csv", "")
        
        if match == "all_columns":
            model_name = "all_columns"
        else:
            model_name = match
        
        model_file = os.path.join(MODELS_DIR, f"model_{model_name}.joblib")
        report_file = os.path.join(MODELS_DIR, f"report_{model_name}.csv")
        
        logger.info(f"Training model for {model_name}")
        
        try:
            success = train_with_features(
                features_file=feature_file,
                model_path=model_file,
                report_file=report_file,
                train_ratio=0.8,
                random_seed=42
            )
            
            if success:
                logger.info(f"Successfully trained model for {model_name} saved to {model_file}")
                trained_models.append(model_file)
            else:
                logger.error(f"Failed to train model for {model_name}")
        except Exception as e:
            logger.error(f"Error training model for {model_name}: {str(e)}")
    
    return trained_models

def main():
    """Run the complete workflow."""
    logger.info("Starting SimDetector Extract and Train Workflow")
    
    # Create necessary directories
    ensure_dirs_exist()
    
    # Check if there are CSV files
    if not get_csv_files():
        return 1
    
    # Extract features
    logger.info("Step 1: Extracting features")
    feature_files = extract_features()
    
    if not feature_files:
        logger.error("No feature files were created, cannot proceed to training")
        return 1
    
    logger.info(f"Successfully created {len(feature_files)} feature files")
    
    # Train models
    logger.info("Step 2: Training models")
    trained_models = train_models(feature_files)
    
    if not trained_models:
        logger.error("No models were trained")
        return 1
    
    logger.info(f"Successfully trained {len(trained_models)} models")
    logger.info("Workflow completed successfully")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())