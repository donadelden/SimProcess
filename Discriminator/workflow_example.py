#!/usr/bin/env python3
"""
One-Class SVM Training and Testing Script

This script:
1. Trains a One-Class SVM model for each feature file in dataset_features/window50/moving_average/
2. Tests each model on specified analysis files
3. Generates a CSV with metrics about the analysis

By default, the script trains on the "real" class (target_class=1). You can change this with the --target-class argument.
"""

import os
import logging
import sys
import csv
import re
import argparse
from galileo.features import process_csv_files
from galileo.model import train_with_features, analyze_with_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ocsvm_training_testing')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train and test One-Class SVM models')
parser.add_argument('--target-class', type=int, choices=[0, 1], default=1,
                    help='Target class to model (1=real, 0=simulated) (default: 1)')
parser.add_argument('--nu', type=float, default=0.1,
                    help='Nu parameter for One-Class SVM (default: 0.1)')
parser.add_argument('--kernel', type=str, choices=['rbf', 'linear', 'poly', 'sigmoid'], default='rbf',
                    help='Kernel type for One-Class SVM (default: rbf)')
parser.add_argument('--gamma', type=str, default='scale',
                    help='Kernel coefficient for rbf, poly and sigmoid kernels (default: scale)')
parser.add_argument('--output-dir', type=str, default='ocsvm_output',
                    help='Directory to store output files (default: ocsvm_output)')
args = parser.parse_args()

# Define parameters
WINDOW_SIZE = 20
FILTER_TYPE = "moving_average"
OUTPUT_DIR = args.output_dir
MODEL_TYPE = "rf"
TARGET_CLASS = args.target_class  # Which class to model (1=real, 0=simulated)

# Files to analyze
FILES_TO_ANALYZE = [
    {
        "file": "dataset/4Mosaik+noise.csv",
        "column": "C1",  # This will be replaced for each model
        "rename": None,
    },
    {
        "file": "dataset/EPIC2.csv",
        "column": "C1",  # This will be replaced for each model
        "rename": None,
    },
    
]

def extract_feature_type(filename):
    """
    Extract the feature type (C1, C2, frequency, etc.) from the filename.
    
    Args:
        filename (str): Name of the feature file
        
    Returns:
        str: Feature type (e.g., C1, C2, frequency)
    """
    match = re.search(r'combined_(.+?)_features\.csv', filename)
    if match:
        return match.group(1)
    return None

def train_model(feature_file, feature_type):
    """
    Train a One-Class SVM model using the specified feature file.
    
    Args:
        feature_file (str): Path to the feature file
        feature_type (str): Type of feature (C1, C2, etc.)
        
    Returns:
        str: Path to the trained model file, or None if training failed
    """
    logger.info(f"Training model for feature type: {feature_type}")
    logger.info(f"Target class: {'real (1)' if TARGET_CLASS == 1 else 'simulated (0)'}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define model file path
    target_class_str = "real" if TARGET_CLASS == 1 else "simulated"
    model_file = os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_{target_class_str}_model_{feature_type}.joblib")
    evaluation_report = os.path.join(OUTPUT_DIR, f"evaluation_report_{feature_type}.csv")
    
    try:
        
        success = train_with_features(
            features_file=feature_file,
            model_path=model_file,
            train_ratio=0.8,
            report_file=evaluation_report,
            model_type=MODEL_TYPE
        )
        
        if not success:
            logger.error(f"Model training failed for {feature_type}")
            return None
        
        logger.info(f"Model for {feature_type} successfully trained and saved to {model_file}")
        return model_file
        
    except Exception as e:
        logger.error(f"Error during model training for {feature_type}: {str(e)}")
        return None

def analyze_files(model_file, feature_type):
    """
    Analyze files using the trained model.
    
    Args:
        model_file (str): Path to the trained model file
        feature_type (str): Type of feature (C1, C2, etc.)
        
    Returns:
        list: List of dictionaries with analysis results
    """
    logger.info(f"Analyzing files with model for {feature_type}")
    
    results = []
    
    if not model_file or not os.path.exists(model_file):
        logger.error("Model file not found. Cannot proceed with analysis.")
        return results
    
    # Morris file column mappings
    morris_column_map = {
        "V1": "R1-PM1:V",
        "V2": "R1-PM2:V",
        "V3": "R1-PM3:V",
        "C1": "R1-PM4:I",
        "C2": "R1-PM5:I",
        "C3": "R1-PM6:I",
        "frequency": "R1:F"
    }
    
    for file_info in FILES_TO_ANALYZE:
        file_path = file_info["file"]
        original_column = file_info["column"]
        
        # Handle Morris files specifically
        is_morris_file = "morris" in file_path
        
        # Update column and rename based on feature type
        if is_morris_file:
            # If it's a Morris file, use the mapping
            if feature_type in morris_column_map:
                column = morris_column_map[feature_type]
                rename = feature_type
            else:
                # Skip this feature if not in mapping for Morris files
                logger.info(f"Skipping feature {feature_type} for Morris file {file_path}")
                continue
        elif feature_type.startswith("C") or feature_type.startswith("V") or feature_type == "frequency":
            # For non-Morris files, handle normally
            column = feature_type
            rename = None if original_column == feature_type else feature_type
        else:
            column = original_column
            rename = original_column  # Keep the original column name
        
        target_class_str = "real" if TARGET_CLASS == 1 else "simulated"
        output_dir = os.path.join(OUTPUT_DIR, f"analysis_{MODEL_TYPE}_{target_class_str}_{feature_type}_{os.path.basename(file_path).split('.')[0]}")
        
        logger.info(f"Analyzing file: {file_path}")
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
            
            # Add result to list
            target_class_str = "real" if TARGET_CLASS == 1 else "simulated"
            result = {
                "model_used": f"{MODEL_TYPE}_{target_class_str}_{feature_type}_window{WINDOW_SIZE}_{FILTER_TYPE}",
                "analyzed_file": file_path,
                "real_windows": metrics['real_windows'],
                "simulated_windows": metrics['simulated_windows'],
                "real_windows_ratio": metrics['real_windows'] / metrics['total_windows'] if metrics['total_windows'] > 0 else 0,
                "simulated_windows_ratio": metrics['simulated_windows'] / metrics['total_windows'] if metrics['total_windows'] > 0 else 0
            }
            
            results.append(result)
            
            logger.info(f"Analysis complete for {file_path}")
            logger.info(f"Classification: {'REAL' if is_real else 'SIMULATED'} with {confidence:.1f}% confidence")
            logger.info(f"Windows: {metrics['real_windows']} real, {metrics['simulated_windows']} simulated out of {metrics['total_windows']} total")
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
    
    return results

def write_results_to_csv(results):
    """
    Write analysis results to a CSV file.
    
    Args:
        results (list): List of dictionaries with analysis results
        
    Returns:
        str: Path to the CSV file
    """
    target_class_str = "real" if TARGET_CLASS == 1 else "simulated"
    csv_file = os.path.join(OUTPUT_DIR, f"analysis_results_{MODEL_TYPE}_{target_class_str}.csv")
    
    with open(csv_file, 'w', newline='') as file:
        fieldnames = [
            "model_used", "analyzed_file", "real_windows", "simulated_windows", 
            "real_windows_ratio", "simulated_windows_ratio"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    logger.info(f"Results written to {csv_file}")
    return csv_file

def main():
    """Run the complete training and testing workflow."""
    logger.info("Starting One-Class SVM training and testing workflow")
    logger.info(f"Target class: {'Real (1)' if TARGET_CLASS == 1 else 'Simulated (0)'}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all feature files in the specified directory
    feature_dir = "dataset_features/window50/moving_average/"
    all_files = [f for f in os.listdir(feature_dir) if f.endswith('_features.csv')]
    
    # Filter out power feature files if requested
    feature_files = [
        os.path.join(feature_dir, f) for f in all_files 
    ]
    
    if not feature_files:
        logger.error(f"No feature files found in {feature_dir}")
        return 1
    
    logger.info(f"Found {len(feature_files)} feature files")
    
    all_results = []
    
    for feature_file in feature_files:
        # Extract feature type from filename
        feature_type = extract_feature_type(os.path.basename(feature_file))
        if not feature_type:
            logger.warning(f"Could not extract feature type from {feature_file}, skipping")
            continue
        
        # Train model
        model_file = train_model(feature_file, feature_type)
        if not model_file:
            logger.warning(f"Model training failed for {feature_file}, skipping analysis")
            continue
        
        # Analyze files
        results = analyze_files(model_file, feature_type)
        all_results.extend(results)
    
    # Write results to CSV
    if all_results:
        target_class_str = "real" if TARGET_CLASS == 1 else "simulated"
        csv_file = write_results_to_csv(all_results)
        logger.info(f"All results saved to {csv_file}")
    else:
        logger.warning("No analysis results to save")
    
    logger.info("Training and testing workflow complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())