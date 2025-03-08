#!/usr/bin/env python3
"""
Three-Step Workflow Example

This script demonstrates a complete workflow:
1. Extract features from data with specific parameters
2. Train a model using the extracted features
3. Analyze multiple files with the model
"""

import os
import logging
import sys
from galileo.features import process_csv_files
from galileo.model import train_with_features, analyze_with_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('workflow_example')

# Define parameters
WINDOW_SIZE = 100
FILTER_TYPE = "moving_average"
TARGET_COLUMN = "C2"
DATA_DIR = "dataset/data"
OUTPUT_DIR = "output"

def step1_extract_features():
    """
    Step 1: Extract features from the dataset.
    Uses window size 20, moving_average filter, and targets the C1 column.
    """
    logger.info("=" * 80)
    logger.info("STEP 1: EXTRACTING FEATURES")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define output file for features
    features_file = os.path.join(OUTPUT_DIR, f"combined_{TARGET_COLUMN}_features.csv")
    
    logger.info(f"Extracting features from {DATA_DIR}")
    logger.info(f"Target column: {TARGET_COLUMN}")
    logger.info(f"Window size: {WINDOW_SIZE}")
    logger.info(f"Filter type: {FILTER_TYPE}")
    
    try:
        success = process_csv_files(
            data_directory=DATA_DIR,
            output_file=features_file,
            target_column=TARGET_COLUMN,
            window_size=WINDOW_SIZE,
            extract_noise=True,
            filter_type=FILTER_TYPE
        )
        
        if not success:
            logger.error("Feature extraction failed")
            return None
        
        logger.info(f"Features successfully extracted to {features_file}")
        return features_file
        
    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}")
        return None

def step2_train_model(features_file):
    """
    Step 2: Train a model using the extracted features.
    
    Args:
        features_file (str): Path to the extracted features file
    
    Returns:
        str: Path to the trained model file, or None if training failed
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: TRAINING MODEL")
    logger.info("=" * 80)
    
    if not features_file or not os.path.exists(features_file):
        logger.error("Feature file not found. Cannot proceed with training.")
        return None
    
    # Define model file path
    model_file = os.path.join(OUTPUT_DIR, f"model_{TARGET_COLUMN}.joblib")
    evaluation_report = os.path.join(OUTPUT_DIR, "evaluation_report.csv")
    
    logger.info(f"Training model using features from {features_file}")
    
    try:
        success = train_with_features(
            features_file=features_file,
            model_path=model_file,
            train_ratio=0.8,
            report_file=evaluation_report
        )
        
        if not success:
            logger.error("Model training failed")
            return None
        
        logger.info(f"Model successfully trained and saved to {model_file}")
        logger.info(f"Evaluation report saved to {evaluation_report}")
        return model_file
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return None

def step3_analyze_files(model_file):
    """
    Step 3: Analyze multiple files using the trained model.
    
    Args:
        model_file (str): Path to the trained model file
    
    Returns:
        bool: True if all analyses completed successfully, False otherwise
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: ANALYZING FILES")
    logger.info("=" * 80)
    
    if not model_file or not os.path.exists(model_file):
        logger.error("Model file not found. Cannot proceed with analysis.")
        return False
    
    # Define files to analyze with their parameters
    files_to_analyze = [
        {
            "file": "dataset/gan/generated_data2.csv",
            "column": TARGET_COLUMN,
            "rename": None,
            "output_dir": os.path.join(OUTPUT_DIR, "analysis_gan")
        },
        {
            "file": "dataset/data/3Panda.csv",
            "column": TARGET_COLUMN,
            "rename": None,
            "output_dir": os.path.join(OUTPUT_DIR, "analysis_panda")
        },
        {
            "file": "dataset/data/2Mosaik.csv",
            "column": TARGET_COLUMN,
            "rename": None,
            "output_dir": os.path.join(OUTPUT_DIR, "analysis_mosaik")
        },
        {
            "file": "dataset/data/processed_EPIC4.csv",
            "column": TARGET_COLUMN,
            "rename": None,
            "output_dir": os.path.join(OUTPUT_DIR, "analysis_EPIC")
        },
        {
            "file": "dataset/morris/data6.csv",
            "column": "R1-PM4:I",
            "rename": TARGET_COLUMN,
            "output_dir": os.path.join(OUTPUT_DIR, "analysis_morris")
        },
        {
            "file": "dataset/morris/data7.csv",
            "column": "R1-PM4:I",
            "rename": TARGET_COLUMN,
            "output_dir": os.path.join(OUTPUT_DIR, "analysis_morris2")
        }
    ]
    
    success_count = 0
    
    for file_info in files_to_analyze:
        file_path = file_info["file"]
        column = file_info["column"]
        rename = file_info["rename"]
        output_dir = file_info["output_dir"]
        
        logger.info(f"\nAnalyzing file: {file_path}")
        logger.info(f"Column: {column}")
        if rename:
            logger.info(f"Renaming to: {rename}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Analyze the file
            is_real, confidence, metrics = analyze_with_model(
                model_path=model_file,
                input_file=file_path,
                target_column=column,
                output_dir=output_dir,
                window_size=WINDOW_SIZE,
                column_rename=rename,
                extract_noise=True,
                filter_type=FILTER_TYPE
            )
            
            logger.info(f"Analysis complete for {file_path}")
            logger.info(f"Classification: {'REAL' if is_real else 'SIMULATED'} with {confidence:.1f}% confidence")
            logger.info(f"Windows: {metrics['real_windows']} real, {metrics['simulated_windows']} simulated out of {metrics['total_windows']} total")
            logger.info(f"Results saved to {output_dir}")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
    
    if success_count == len(files_to_analyze):
        logger.info("\nAll files successfully analyzed.")
        return True
    else:
        logger.warning(f"\n{success_count} out of {len(files_to_analyze)} files successfully analyzed.")
        return False

def main():
    """Run the complete three-step workflow."""
    logger.info("Starting three-step workflow")
    
    # Step 1: Extract features
    features_file = step1_extract_features()
    if not features_file:
        logger.error("Feature extraction failed. Workflow cannot continue.")
        return 1
    
    # Step 2: Train model
    model_file = step2_train_model(features_file)
    if not model_file:
        logger.error("Model training failed. Workflow cannot continue.")
        return 1
    
    # Step 3: Analyze files
    analysis_success = step3_analyze_files(model_file)
    
    logger.info("\nWorkflow complete.")
    
    return 0 if analysis_success else 1

if __name__ == "__main__":
    sys.exit(main())