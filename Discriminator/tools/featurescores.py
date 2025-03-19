#!/usr/bin/env python3
"""
Feature Importance Extractor for Galileo Framework

This script explores the dataset folder structure and generates feature importance
CSV files for each combined_*_features.csv file in the dataset.
"""

import os
import re
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('galileo.feature_importance')

def extract_csv_type(filename):
    """Extract the type of CSV file from its filename"""
    # Handle patterns like combined_V1_features.csv, combined_C1_features.csv
    v_c_pattern = r'combined_(V\d|C\d)_features\.csv'
    v_c_match = re.search(v_c_pattern, filename)
    if v_c_match:
        return v_c_match.group(1)
    
    # Handle patterns like combined_frequency_features.csv
    freq_pattern = r'combined_(frequency)_features\.csv'
    freq_match = re.search(freq_pattern, filename)
    if freq_match:
        return freq_match.group(1)
    
    # Handle patterns like combined_power_real_features.csv
    power_pattern = r'combined_power_(real|apparent|reactive)_features\.csv'
    power_match = re.search(power_pattern, filename)
    if power_match:
        return f"power_{power_match.group(1)}"
    
    # Fallback general pattern
    general_pattern = r'combined_(.+?)_features\.csv'
    general_match = re.search(general_pattern, filename)
    if general_match:
        return general_match.group(1)
    
    return None

def calculate_feature_importance(features_df, target_col='real'):
    """
    Calculate feature importance using permutation importance.
    
    Args:
        features_df (pandas.DataFrame): DataFrame with features and target column
        target_col (str): Name of the target column
        
    Returns:
        pandas.DataFrame: DataFrame with feature importance scores
    """
    try:
        # Check if the dataset has the target column
        if target_col not in features_df.columns:
            logger.warning(f"No '{target_col}' column found, skipping")
            return None
        
        # Make a copy of the dataframe to avoid modifying the original
        df = features_df.copy()
        
        # Remove non-numeric columns that aren't features (target and source)
        columns_to_drop = [target_col]
        
        # Check for and remove the 'source' column which contains filenames
        if 'source' in df.columns:
            columns_to_drop.append('source')
        
        # Also remove any other string columns that might cause issues
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                if col not in columns_to_drop:
                    columns_to_drop.append(col)
        
        # Prepare data for feature importance calculation
        X = df.drop(columns_to_drop, axis=1)
        y = df[target_col]
        
        # Handle potential NaN values
        X = X.fillna(0)
        
        # Check if we have enough features and samples
        if X.shape[1] == 0:
            logger.warning(f"No features found after dropping non-numeric columns, skipping")
            return None
            
        if X.shape[0] < 10:
            logger.warning(f"Not enough samples (found {X.shape[0]}), skipping")
            return None
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train the model (RandomForestClassifier for binary classification)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        try:
            model.fit(X_scaled, y)
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            # Try using RandomForestRegressor if classification fails
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
        
        # Get feature importance
        # First try using model's feature_importances_ attribute
        try:
            importance_scores = model.feature_importances_
        except:
            # Fall back to permutation importance
            result = permutation_importance(model, X_scaled, y, n_repeats=10, random_state=42)
            importance_scores = result.importances_mean
        
        importance_df = pd.DataFrame({
            'Feature': X.columns.tolist(),
            'Importance': importance_scores
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
        
    except Exception as e:
        logger.error(f"Error calculating feature importance: {str(e)}")
        return None
    
def process_csv_file(csv_path):
    """Process a single CSV file and save feature importance results in the same location"""
    try:
        # Extract the type from filename
        csv_type = extract_csv_type(os.path.basename(csv_path))
        if not csv_type:
            logger.warning(f"Could not extract type from {csv_path}, skipping")
            return False
        
        # Read the CSV file
        logger.info(f"Processing {csv_path}")
        features_df = pd.read_csv(csv_path)
        
        # Calculate feature importance
        importance_df = calculate_feature_importance(features_df)
        if importance_df is None:
            logger.warning(f"Failed to calculate feature importance for {csv_path}")
            return False
        
        # Create output filename and path - save to the same directory as the input file
        output_dir = os.path.dirname(csv_path)
        output_filename = f"feature_importance_{csv_type}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the results
        importance_df.to_csv(output_path, index=False)
        logger.info(f"Saved feature importance to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {csv_path}: {str(e)}")
        return False

def traverse_dataset_directory(root_dir):
    """Traverse the dataset directory structure and process each CSV file"""
    # Count processed files
    processed_count = 0
    error_count = 0
    
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv') and filename.startswith('combined_') and '_features' in filename:
                csv_path = os.path.join(dirpath, filename)
                success = process_csv_file(csv_path)
                if success:
                    processed_count += 1
                else:
                    error_count += 1
    
    logger.info(f"Feature importance calculation completed:")
    logger.info(f"  Successfully processed {processed_count} files")
    if error_count > 0:
        logger.warning(f"  Encountered errors in {error_count} files")
    
    return processed_count, error_count

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Feature Importance Extractor for Galileo Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset', '-d',
                       default='dataset',
                       help='Path to the dataset folder (default: dataset)')
    
    parser.add_argument('--quiet', '-q',
                       action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Adjust logging level if quiet mode is enabled
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Convert paths to absolute
    dataset_path = os.path.abspath(args.dataset)
    
    # Check if dataset directory exists
    if not os.path.isdir(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        return 1
    
    logger.info(f"Exploring dataset at: {dataset_path}")
    logger.info(f"Feature importance files will be saved alongside each input file")
    
    # Process the dataset
    processed_count, error_count = traverse_dataset_directory(dataset_path)
    
    if processed_count == 0:
        logger.warning("No files were processed successfully")
        return 1
    
    if error_count > 0:
        return 2
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
