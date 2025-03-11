#!/usr/bin/env python3
"""
A self-contained script to extract noise from signal data using Kalman filtering.
The noise is calculated as the original value minus the filtered value.
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys

def kalman_filter(data, process_variance=1e-5, measurement_variance=1e-1):
    """
    Apply a simple Kalman filter to the data.
    
    Args:
        data (numpy.ndarray): Data to filter
        process_variance (float): Process variance parameter (Q)
        measurement_variance (float): Measurement variance parameter (R)
        
    Returns:
        numpy.ndarray: Filtered data
    """
    # Handle empty or NaN data
    if len(data) == 0 or np.isnan(data).all():
        return np.zeros_like(data)
    
    # Replace NaN with interpolated values or zeros
    clean_data = pd.Series(data).interpolate().fillna(0).values
    
    # Initialize state and covariance
    x_hat = clean_data[0]  # Initial state estimate
    P = 1.0  # Initial covariance estimate
    
    # Allocate space for filtered data
    filtered_data = np.zeros_like(clean_data)
    
    # Kalman filter loop
    for i, measurement in enumerate(clean_data):
        # Prediction step (time update)
        x_hat_minus = x_hat
        P_minus = P + process_variance
        
        # Correction step (measurement update)
        K = P_minus / (P_minus + measurement_variance)  # Kalman gain
        x_hat = x_hat_minus + K * (measurement - x_hat_minus)
        P = (1 - K) * P_minus
        
        # Store the estimate
        filtered_data[i] = x_hat
    
    return filtered_data

def extract_noise(file_path, output_file=None, process_variance=1e-5, measurement_variance=1e-1):
    """
    Extract noise from all relevant columns in a CSV file using Kalman filtering.
    
    Args:
        file_path (str): Path to the input CSV file
        output_file (str, optional): Path to save the output CSV. If None, auto-generated.
        process_variance (float): Process variance parameter for Kalman filter
        measurement_variance (float): Measurement variance parameter for Kalman filter
        
    Returns:
        str: Path to the output CSV file
    """
    print(f"Loading data from {file_path}...")
    
    # Load the data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        sys.exit(1)
    
    # Check if timestamp column exists
    if 'timestamp' not in df.columns:
        print("Warning: 'timestamp' column not found. Using row index instead.")
        df['timestamp'] = pd.Series(range(len(df)))
    
    # Define columns to process
    signal_types = {
        'current': ['C1', 'C2', 'C3'],
        'voltage': ['V1', 'V2', 'V3'],
        'power': ['power_real', 'power_effective', 'power_apparent'],
        'frequency': ['frequency']
    }
    
    # Flatten the list of columns
    columns_to_process = []
    for category, columns in signal_types.items():
        for col in columns:
            if col in df.columns:
                columns_to_process.append(col)
    
    if not columns_to_process:
        print("No valid columns found to process. Available columns:")
        print(", ".join(df.columns))
        sys.exit(1)
    
    print(f"Processing columns: {', '.join(columns_to_process)}")
    
    # Create result dataframe with timestamp
    result_df = pd.DataFrame()
    result_df['timestamp'] = df['timestamp']
    
    # Process each column
    for column in columns_to_process:
        print(f"Extracting noise from {column}...")
        
        # Get the original data
        original_data = df[column].values
        
        # Apply Kalman filter
        filtered_data = kalman_filter(
            original_data,
            process_variance=process_variance,
            measurement_variance=measurement_variance
        )
        
        # Calculate noise (original - filtered)
        noise = original_data - filtered_data
        
        # Add to result dataframe
        result_df[f"{column}_noise"] = noise
    
    # Generate output file name if not provided
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = f"{base_name}_noise.csv"
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Noise extraction complete. Results saved to {output_file}")
    
    return output_file

def main():
    """
    Main function to parse arguments and run the noise extraction.
    """
    parser = argparse.ArgumentParser(description='Extract noise from signal data using Kalman filtering.')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV file path (default: input_filename_noise.csv)')
    parser.add_argument('--process-variance', '-p', type=float, default=1e-5,
                        help='Process variance parameter for Kalman filter (default: 1e-5)')
    parser.add_argument('--measurement-variance', '-m', type=float, default=1e-1,
                        help='Measurement variance parameter for Kalman filter (default: 0.1)')
    
    args = parser.parse_args()
    
    extract_noise(
        args.input,
        args.output,
        process_variance=args.process_variance,
        measurement_variance=args.measurement_variance
    )

if __name__ == "__main__":
    main()