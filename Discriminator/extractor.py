#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import argparse
from galileo import load_data, extract_window_features, extract_features
from preprocessor import extract_noise_signal

def process_csv_files(data_directory, output_file=None, target_column='V1', window_size=10, 
                  extract_noise=True, filter_type='savgol', cutoff=0.1, fs=1.0, poly_order=2, 
                  output_column_prefix=None):
    """
    Process all CSV files in the given directory, extract features from specified column,
    and combine them into a single CSV file.
    
    Args:
        data_directory (str): Directory containing CSV files
        output_file (str, optional): Path to save the combined features. If None, auto-generated based on column name
        target_column (str): Column to extract features from (default: 'V1')
        window_size (int): Size of the window for feature extraction
        extract_noise (bool): Whether to extract noise features
        filter_type (str): Type of filter to use for noise extraction ('moving_average', 'butterworth', 'savgol')
        cutoff (float): Cutoff frequency for Butterworth filter
        fs (float): Sampling frequency for Butterworth filter
        poly_order (int): Polynomial order for Savitzky-Golay filter
        output_column_prefix (str, optional): Prefix to use for output column names. If None, uses target_column
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if the directory exists
    if not os.path.isdir(data_directory):
        print(f"Error: Directory '{data_directory}' does not exist")
        return False
        
    # Generate default output filename if not provided
    if output_file is None:
        output_file = f"combined_{target_column}_features.csv"
    
    # Use target_column as the output column prefix if not specified
    if output_column_prefix is None:
        output_column_prefix = target_column
    else:
        print(f"Using '{output_column_prefix}' as prefix for feature names instead of '{target_column}'")
    
    # Find all CSV files in the directory
    csv_files = [f for f in os.listdir(data_directory) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in '{data_directory}'")
        return False
    
    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Extracting features from column: {target_column}")
    
    # List to store all features DataFrames
    all_features = []
    
    # Process each CSV file
    for i, file_name in enumerate(csv_files, 1):
        file_path = os.path.join(data_directory, file_name)
        print(f"Processing file {i}/{len(csv_files)}: {file_name}")
        
        # Load data
        df = load_data(file_path)
        if df is None:
            print(f"  Skipping {file_name} due to loading error")
            continue
        
        
        # Check if target column exists
        if target_column not in df.columns:
            print(f"  Warning: Column '{target_column}' not found in {file_name}")
            print(f"  Available columns: {', '.join(df.columns)}")
            continue
            
        # Extract features using the same logic as in galileo
        features_df, feature_names = extract_window_features(df, window_size=window_size, target_column=target_column)
        
        if features_df.empty:
            print(f"  No features extracted from {file_name}")
            continue
            
        # Rename columns if a different output column prefix is specified
        if output_column_prefix != target_column:
            # Create a mapping for column renaming
            rename_map = {}
            for col in features_df.columns:
                if col.startswith(target_column + '_'):
                    new_col = col.replace(target_column + '_', output_column_prefix + '_', 1)
                    rename_map[col] = new_col
            
            # Rename columns
            if rename_map:
                features_df = features_df.rename(columns=rename_map)
        
        # Extract noise features if requested
        if extract_noise:
            noise_features = []
            
            # Process each window to extract noise features
            for i in range(0, len(df), window_size//2):  # 50% overlap
                window = df.iloc[i:i+window_size].copy()
                
                # Skip if window is too small
                if len(window) < window_size:
                    continue
                    
                # Calculate appropriate window size and poly order for noise extraction
                noise_window_size = max(window_size//5, 3)  # Ensure minimum window size of 3
                # Ensure poly order is always less than window size
                noise_poly_order = min(poly_order, noise_window_size - 1)
                
                # Extract noise from the window
                try:
                    noise_df = extract_noise_signal(
                        window, 
                        filter=filter_type,
                        window_size=noise_window_size,
                        cutoff=cutoff,
                        fs=fs,
                        poly_order=noise_poly_order,
                        keep_noise_only=True,
                        target_column=target_column
                    )
                except Exception as e:
                    print(f"  Warning: Error extracting noise - {str(e)}")
                    continue
                
                # Skip if noise extraction failed
                if noise_df.empty or target_column not in noise_df.columns:
                    continue
                    
                # Extract features from the noise signal
                noise_signal = noise_df[target_column].values
                if len(noise_signal) > 1:
                    # Explicitly indicate this is a noise signal
                    noise_feature_dict = extract_features(noise_signal, is_noise=True)
                    if noise_feature_dict:
                        # Add 'noise_' prefix to feature names
                        # If output_column_prefix is different from target_column, use it for noise features too
                        noise_feature_dict = {f"noise_{output_column_prefix}_{k}": v for k, v in noise_feature_dict.items()}
                        noise_features.append(noise_feature_dict)
            
            # If we have noise features, create a DataFrame and align with original features
            if noise_features:
                noise_df = pd.DataFrame(noise_features)
                
                # Make sure we have the same number of rows
                min_rows = min(len(features_df), len(noise_df))
                features_df = features_df.iloc[:min_rows].reset_index(drop=True)
                noise_df = noise_df.iloc[:min_rows].reset_index(drop=True)
                
                # Merge noise features with original features
                for col in noise_df.columns:
                    features_df[col] = noise_df[col]
                
                print(f"  Added {len(noise_df.columns)} noise features")
        
        # Add binary label based on filename
        # 1 if "EPIC" is in the filename, 0 otherwise
        is_real = 1 if "EPIC" in file_name else 0
        features_df['real'] = is_real
        print(f"  Label: {'Real (1)' if is_real else 'Not Real (0)'}")
        
        # Add to the collection
        all_features.append(features_df)
        
        print(f"  Extracted {len(features_df)} windows with {len(features_df.columns) - 1} features")
    
    # Check if we have any features
    if not all_features:
        print("No features were extracted from any of the files")
        return False
    
    # Combine all features into a single DataFrame
    combined_df = pd.concat(all_features, ignore_index=True)
    
    # Clean the features dataframe
    combined_df = clean_features_dataframe(combined_df)
    
    # Count the number of real and not real samples
    real_count = combined_df['real'].sum()
    not_real_count = len(combined_df) - real_count
    print(f"\nData distribution:")
    print(f"  Real (1): {real_count} windows ({real_count/len(combined_df)*100:.2f}%)")
    print(f"  Not Real (0): {not_real_count} windows ({not_real_count/len(combined_df)*100:.2f}%)")
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved combined features to '{output_file}'")
    print(f"Total windows: {len(combined_df)}")
    print(f"Total features: {len(combined_df.columns) - 1}")  # -1 for the 'real' label column
    
    return True


def clean_features_dataframe(combined_df):
    """
    Clean up the features dataframe by handling null values, empty strings, and invalid data.
    
    Args:
        combined_df (pandas.DataFrame): The dataframe with extracted features
        
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    # Make a copy to avoid modifying the original
    cleaned_df = combined_df.copy()
    
    # Replace empty strings with NaN
    cleaned_df = cleaned_df.replace('', np.nan)
    
    # Find columns containing critical statistical measures
    critical_cols = [col for col in cleaned_df.columns if any(term in col for term in 
                    ['std', 'variance', 'entropy', 'autocorr', 'kurtosis'])]
    
    # If we found any critical columns, filter rows where all these values are near zero
    if critical_cols:
        # Identify rows where all critical features are very close to zero
        all_critical_near_zero = (cleaned_df[critical_cols].abs() < 1e-4).all(axis=1)
        cleaned_df = cleaned_df[~all_critical_near_zero]
        print(f"Removed {all_critical_near_zero.sum()} rows where all critical features are near zero")
    
    # Replace NaN with 0 for numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
    
    # Check for rows with too many zeros or near-zero values (potential low-quality data)
    # Consider a row low quality if more than 50% of its features are zero or very close to zero
    near_zero_values = (cleaned_df[numeric_cols].abs() < 1e-6)
    zero_percentage = near_zero_values.sum(axis=1) / len(numeric_cols)
    low_quality_rows = zero_percentage > 0.5
    
    # Also check if the row has at least some minimum variance or meaningful features
    # Flag rows where the first few numeric features are all zeros
    key_features = numeric_cols[:5]  # First 5 numeric features
    key_features_zero = (cleaned_df[key_features].abs() < 1e-6).all(axis=1)
    low_quality_rows = low_quality_rows | key_features_zero
    
    # Print information about low quality rows
    low_quality_count = low_quality_rows.sum()
    if low_quality_count > 0:
        print(f"Found {low_quality_count} low quality rows ({low_quality_count/len(cleaned_df)*100:.2f}%)")
        print("These rows have more than 80% of features as zeros or null values")
        
        # Option 1: Remove these rows
        cleaned_df = cleaned_df[~low_quality_rows]
        print(f"Removed {low_quality_count} low quality rows")
        
        # Option 2: Just report them but keep them
        # pass
    
    return cleaned_df

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Extract features from a specified column in CSV files')
    parser.add_argument('--data_dir', '-d', default='data/', help='Directory containing CSV files')
    parser.add_argument('--output', '-o', help='Output CSV file path (default: combined_COLUMN_features.csv)')
    parser.add_argument('--column', '-c', help='Column to extract features from')
    parser.add_argument('--window', '-w', type=int, default=10, help='Window size for feature extraction')
    parser.add_argument('--rename', '-r', help='Rename the column prefix in output features (default: use the --column value)')
    
    # Noise extraction arguments
    parser.add_argument('--no-noise', action='store_true', help='Disable noise feature extraction')
    parser.add_argument('--filter', choices=['moving_average', 'butterworth', 'savgol'], 
                        default='savgol', help='Filter type for noise extraction')
    parser.add_argument('--cutoff', type=float, default=0.1, help='Cutoff frequency for Butterworth filter')
    parser.add_argument('--fs', type=float, default=1.0, help='Sampling frequency for Butterworth filter')
    parser.add_argument('--poly-order', type=int, default=2, help='Polynomial order for Savitzky-Golay filter (must be < window size)')
    
    args = parser.parse_args()
    
    process_csv_files(
        args.data_dir, 
        args.output, 
        args.column, 
        args.window,
        not args.no_noise,  # extract_noise is True unless --no-noise is specified
        args.filter,
        args.cutoff,
        args.fs,
        args.poly_order,
        args.rename  # Pass the rename parameter to the function
    )

if __name__ == "__main__":
    main()