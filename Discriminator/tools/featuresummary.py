#!/usr/bin/env python3
"""
Dataset Summary Generator for SimDetector Framework

This script explores all window directories and their subfolders, analyzing feature CSV files
to generate a comprehensive summary of the dataset statistics, including counts of real and
simulated samples and their ratios.
"""

import os
import re
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simdetector.dataset_summary')

def extract_feature_type(filename):
    """Extract the type of feature from a CSV filename"""
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

def analyze_csv_file(file_path, window_size, filter_type):
    """
    Analyze a CSV file to extract statistics about real and simulated samples.
    
    Args:
        file_path (str): Path to the CSV file
        window_size (str): Window size directory name (e.g., 'window60')
        filter_type (str): Filter type directory name (e.g., 'savgol')
        
    Returns:
        dict: Dictionary with statistics or None if analysis fails
    """
    try:
        # Extract feature type from filename
        filename = os.path.basename(file_path)
        feature_type = extract_feature_type(filename)
        if not feature_type:
            logger.warning(f"Could not extract feature type from {filename}, skipping")
            return None
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if 'real' column exists
        if 'real' not in df.columns:
            logger.warning(f"No 'real' column found in {file_path}, skipping")
            return None
        
        # Count total rows, real=1 rows, and real=0 rows
        total_rows = len(df)
        real_rows = df['real'].sum()
        simulated_rows = total_rows - real_rows
        
        # Calculate ratios
        real_ratio = real_rows / total_rows if total_rows > 0 else 0
        simulated_ratio = simulated_rows / total_rows if total_rows > 0 else 0
        
        # Create feature statistics dictionary
        stats = {
            'window_size': window_size,
            'filter_type': filter_type,
            'feature_type': feature_type,
            'total_samples': total_rows,
            'real_samples': int(real_rows),
            'simulated_samples': int(simulated_rows),
            'real_ratio': real_ratio,
            'simulated_ratio': simulated_ratio,
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {str(e)}")
        return None

def explore_dataset(root_dir):
    """
    Explore dataset directory structure to analyze all feature CSV files.
    
    Args:
        root_dir (str): Root directory of the dataset
        
    Returns:
        list: List of dictionaries containing statistics for each CSV file
    """
    dataset_stats = []
    
    # Pattern to identify window directories
    window_pattern = re.compile(r'window\d+')
    
    # Known filter type directories
    filter_types = ['no_noise', 'savgol', 'moving_average', 'butterworth']
    
    # Walk through dataset directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Get current directory name
        current_dir = os.path.basename(dirpath)
        
        # Check if this is a window directory
        if not window_pattern.match(current_dir):
            # Check if this is a filter type directory inside a window directory
            parent_dir = os.path.basename(os.path.dirname(dirpath))
            if window_pattern.match(parent_dir) and current_dir in filter_types:
                window_size = parent_dir
                filter_type = current_dir
                
                # Process CSV files in this directory
                for filename in filenames:
                    if filename.endswith('.csv') and filename.startswith('combined_') and '_features' in filename:
                        file_path = os.path.join(dirpath, filename)
                        logger.info(f"Analyzing {window_size}/{filter_type}/{filename}")
                        
                        # Analyze the CSV file
                        stats = analyze_csv_file(file_path, window_size, filter_type)
                        if stats:
                            dataset_stats.append(stats)
    
    return dataset_stats

def save_summary(dataset_stats, output_file):
    """
    Save dataset statistics to a CSV file.
    
    Args:
        dataset_stats (list): List of dictionaries containing statistics
        output_file (str): Path to save the summary CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not dataset_stats:
            logger.warning("No statistics to save")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset_stats)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Summary saved to {output_file}")
        
        # Print a summary of findings
        window_sizes = df['window_size'].unique()
        filter_types = df['filter_type'].unique()
        feature_types = df['feature_type'].unique()
        
        logger.info(f"Dataset Summary:")
        logger.info(f"  Window sizes: {', '.join(window_sizes)}")
        logger.info(f"  Filter types: {', '.join(filter_types)}")
        logger.info(f"  Feature types: {', '.join(feature_types)}")
        logger.info(f"  Total files analyzed: {len(df)}")
        logger.info(f"  Total samples across all files: {df['total_samples'].sum()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving summary: {str(e)}")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Dataset Summary Generator for SimDetector Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset', '-d',
                       default='dataset',
                       help='Path to the dataset folder (default: dataset)')
    
    parser.add_argument('--output', '-o',
                       default='dataset_summary.csv',
                       help='Path to save the summary CSV file (default: dataset_summary.csv)')
    
    parser.add_argument('--quiet', '-q',
                       action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Adjust logging level if quiet mode is enabled
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Convert paths to absolute
    dataset_path = os.path.abspath(args.dataset)
    output_path = os.path.abspath(args.output)
    
    # Check if dataset directory exists
    if not os.path.isdir(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        return 1
    
    logger.info(f"Exploring dataset at: {dataset_path}")
    logger.info(f"Summary will be saved to: {output_path}")
    
    # Explore dataset and gather statistics
    dataset_stats = explore_dataset(dataset_path)
    
    if not dataset_stats:
        logger.warning("No files were analyzed successfully")
        return 1
    
    # Save summary to CSV
    success = save_summary(dataset_stats, output_path)
    if not success:
        logger.error("Failed to save summary")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)