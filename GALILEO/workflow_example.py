#!/usr/bin/env python3
"""
Process All Columns

This script extracts features for each column and runs analysis on them.
"""

import os
import logging
import sys
from galileo.features import process_csv_files
from galileo.model import train_with_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('process_all_columns')

# Define columns to process
COLUMNS = ["C1", "C2", "C3", "V1", "V2", "V3", "frequency", 
           "power_real", "power_effective", "power_apparent"]

# Define data directory
DATA_DIR = "../Discriminator/data/"

def process_column(column):
    """Process a single column: extract features and train a model."""
    logger.info("=" * 50)
    logger.info(f"Processing column: {column}")
    logger.info("=" * 50)
    
    # Step 1: Extract features for this column
    output_file = f"combined_{column}_features.csv"
    logger.info("Extracting features...")
    
    try:
        success = process_csv_files(
            data_directory=DATA_DIR,
            output_file=output_file,
            target_column=column,
            window_size=10,
            extract_noise=False  # Equivalent to --no-noise
        )
        
        if not success:
            logger.error(f"Feature extraction failed for column {column}")
            return False
            
    except Exception as e:
        logger.error(f"Error during feature extraction for column {column}: {str(e)}")
        return False
    
    # Step 2: Run main analysis on the extracted features
    logger.info("Running analysis...")
    model_file = f"model_{column}.joblib"
    
    try:
        success = train_with_features(
            features_file=output_file,
            model_path=model_file
        )
        
        if not success:
            logger.error(f"Analysis failed for column {column}")
            return False
            
    except Exception as e:
        logger.error(f"Error during analysis for column {column}: {str(e)}")
        return False
    
    logger.info(f"Finished processing {column}")
    logger.info("")
    return True

def main():
    """Process all columns."""
    success_count = 0
    failure_count = 0
    
    for column in COLUMNS:
        if process_column(column):
            success_count += 1
        else:
            failure_count += 1
    
    logger.info("All columns processed.")
    logger.info(f"Results: {success_count} succeeded, {failure_count} failed")
    
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())