"""
Feature extraction and processing functions for the Galileo framework.
"""

import os
import pandas as pd
import numpy as np
import logging
from tsfresh.feature_extraction import feature_calculators
from galileo.core import (
    SIGNAL_TYPES, 
    CRITICAL_FEATURES, 
    FeatureExtractionError,
    validate_directory
)
from galileo.data import load_data, filter_data, extract_noise_signal

logger = logging.getLogger('galileo.features')

def extract_features(signal, is_noise=False):
    """
    Extract features from a signal with improved error handling.
    
    Args:
        signal: The signal data to extract features from
        is_noise: Boolean indicating if the signal is noise data
        
    Returns:
        dict: Dictionary of extracted features
    """
    if len(signal) < 2:
        return None
        
    features = {}
    
    if is_noise:
        # Extract all tsfresh minimal features for noise data
        from tsfresh.feature_extraction.feature_calculators import (
            # Basic statistics
            mean, median, standard_deviation, variance, 
            minimum, maximum, absolute_maximum, absolute_sum_of_changes,
            mean_change, mean_abs_change, root_mean_square,
            
            # Entropy features
            approximate_entropy, sample_entropy, permutation_entropy,
            
            # Complexity measures
            binned_entropy, lempel_ziv_complexity, 
            
            # Peaks and crossings
            number_crossing_m, number_peaks, longest_strike_above_mean,
            longest_strike_below_mean,
            
            # Auto-correlation features
            autocorrelation, partial_autocorrelation,
            
            # Distribution features
            skewness, kurtosis, quantile,
            
            # Frequency domain
            fft_coefficient, fft_aggregated,
            
            # Statistical tests
            augmented_dickey_fuller,
            
            # Others
            ratio_beyond_r_sigma, count_above_mean, count_below_mean,
            energy_ratio_by_chunks, percentage_of_reoccurring_values_to_all_values,
            percentage_of_reoccurring_datapoints_to_all_datapoints,
            benford_correlation, time_reversal_asymmetry_statistic
        )
        
        # Basic statistics
        try:
            features['mean'] = mean(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate mean: {str(e)}")
            features['mean'] = 0.0
            
        try:
            features['median'] = median(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate median: {str(e)}")
            features['median'] = 0.0
        
        try:
            features['standard_deviation'] = standard_deviation(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate standard_deviation: {str(e)}")
            features['standard_deviation'] = 0.0
        
        try:
            features['variance'] = variance(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate variance: {str(e)}")
            features['variance'] = 0.0
            
        try:
            features['minimum'] = minimum(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate minimum: {str(e)}")
            features['minimum'] = 0.0
            
        try:
            features['maximum'] = maximum(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate maximum: {str(e)}")
            features['maximum'] = 0.0
            
        try:
            features['absolute_maximum'] = absolute_maximum(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate absolute_maximum: {str(e)}")
            features['absolute_maximum'] = 0.0
            
        try:
            features['absolute_sum_of_changes'] = absolute_sum_of_changes(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate absolute_sum_of_changes: {str(e)}")
            features['absolute_sum_of_changes'] = 0.0
            
        try:
            features['mean_change'] = mean_change(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate mean_change: {str(e)}")
            features['mean_change'] = 0.0
            
        try:
            features['mean_abs_change'] = mean_abs_change(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate mean_abs_change: {str(e)}")
            features['mean_abs_change'] = 0.0
            
        try:
            features['root_mean_square'] = root_mean_square(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate root_mean_square: {str(e)}")
            features['root_mean_square'] = 0.0
        
        # Entropy features
        try:
            features['approx_entropy'] = approximate_entropy(signal, m=2, r=0.3)
        except Exception as e:
            logger.debug(f"Failed to calculate approx_entropy: {str(e)}")
            features['approx_entropy'] = 0.0
              
        try:
            features['permutation_entropy'] = permutation_entropy(signal, tau=1, dimension=3)
        except Exception as e:
            logger.debug(f"Failed to calculate permutation_entropy: {str(e)}")
            features['permutation_entropy'] = 0.0
            
        # Complexity measures
        try:
            features['binned_entropy'] = binned_entropy(signal, max_bins=10)
        except Exception as e:
            logger.debug(f"Failed to calculate binned_entropy: {str(e)}")
            features['binned_entropy'] = 0.0
            
        try:
            features['lempel_ziv_complexity'] = lempel_ziv_complexity(signal, bins=10)
        except Exception as e:
            logger.debug(f"Failed to calculate lempel_ziv_complexity: {str(e)}")
            features['lempel_ziv_complexity'] = 0.0
            
        # Peaks and crossings
        try:
            features['number_crossing_m'] = number_crossing_m(signal, m=0)
        except Exception as e:
            logger.debug(f"Failed to calculate number_crossing_m: {str(e)}")
            features['number_crossing_m'] = 0
            
        try:
            features['number_peaks'] = number_peaks(signal, n=1)
        except Exception as e:
            logger.debug(f"Failed to calculate number_peaks: {str(e)}")
            features['number_peaks'] = 0
            
        try:
            features['longest_strike_above_mean'] = longest_strike_above_mean(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate longest_strike_above_mean: {str(e)}")
            features['longest_strike_above_mean'] = 0
            
        try:
            features['longest_strike_below_mean'] = longest_strike_below_mean(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate longest_strike_below_mean: {str(e)}")
            features['longest_strike_below_mean'] = 0
            
        # Auto-correlation features
        try:
            features['autocorrelation_lag1'] = autocorrelation(signal, lag=1)
            if pd.isna(features['autocorrelation_lag1']):
                features['autocorrelation_lag1'] = 0.0
        except Exception as e:
            logger.debug(f"Failed to calculate autocorrelation: {str(e)}")
            features['autocorrelation_lag1'] = 0.0
                        
        # Distribution features
        try:
            features['skewness'] = skewness(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate skewness: {str(e)}")
            features['skewness'] = 0.0
            
        try:
            features['kurtosis'] = kurtosis(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate kurtosis: {str(e)}")
            features['kurtosis'] = 0.0
            
        try:
            features['quantile_25'] = quantile(signal, 0.25)
            features['quantile_75'] = quantile(signal, 0.75)
        except Exception as e:
            logger.debug(f"Failed to calculate quantile: {str(e)}")
            features['quantile_25'] = 0.0
            features['quantile_75'] = 0.0
            
        # Frequency domain features
        try:
            # Extract the first few coefficients
            for i in range(3):
                for attr in ['real', 'imag', 'abs', 'angle']:
                    coef = fft_coefficient(signal, [{'coeff': i, 'attr': attr}])
                    features[f'fft_coeff_{i}_{attr}'] = coef[0][1]
        except Exception as e:
            logger.debug(f"Failed to calculate fft_coefficient: {str(e)}")
            for i in range(3):
                for attr in ['real', 'imag', 'abs', 'angle']:
                    features[f'fft_coeff_{i}_{attr}'] = 0.0
            
        try:
            fft_agg = fft_aggregated(signal, ['centroid', 'variance', 'skew', 'kurtosis'])
            for i, agg in enumerate(['centroid', 'variance', 'skew', 'kurtosis']):
                features[f'fft_aggregated_{agg}'] = fft_agg[i]
        except Exception as e:
            logger.debug(f"Failed to calculate fft_aggregated: {str(e)}")
            for agg in ['centroid', 'variance', 'skew', 'kurtosis']:
                features[f'fft_aggregated_{agg}'] = 0.0
            
        # Statistical tests
        try:
            adf = augmented_dickey_fuller(signal, [{'attr': 'pvalue'}])
            features['adf_pvalue'] = adf[0][1]
        except Exception as e:
            logger.debug(f"Failed to calculate augmented_dickey_fuller: {str(e)}")
            features['adf_pvalue'] = 0.5  # Default p-value
            
        # Others
        try:
            features['ratio_beyond_r_sigma_1'] = ratio_beyond_r_sigma(signal, r=1)
            features['ratio_beyond_r_sigma_2'] = ratio_beyond_r_sigma(signal, r=2)
            features['ratio_beyond_r_sigma_3'] = ratio_beyond_r_sigma(signal, r=3)
        except Exception as e:
            logger.debug(f"Failed to calculate ratio_beyond_r_sigma: {str(e)}")
            features['ratio_beyond_r_sigma_1'] = 0.0
            features['ratio_beyond_r_sigma_2'] = 0.0
            features['ratio_beyond_r_sigma_3'] = 0.0
            
        try:
            features['count_above_mean'] = count_above_mean(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate count_above_mean: {str(e)}")
            features['count_above_mean'] = 0
            
        try:
            features['count_below_mean'] = count_below_mean(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate count_below_mean: {str(e)}")
            features['count_below_mean'] = 0
            
        try:
            energy_ratio = energy_ratio_by_chunks(signal, [{'num_segments': 10, 'segment_focus': 0}])
            features['energy_ratio_by_chunks'] = energy_ratio[0][1]
        except Exception as e:
            logger.debug(f"Failed to calculate energy_ratio_by_chunks: {str(e)}")
            features['energy_ratio_by_chunks'] = 0.0
            
        try:
            features['percentage_reoccurring_values'] = percentage_of_reoccurring_values_to_all_values(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate percentage_of_reoccurring_values_to_all_values: {str(e)}")
            features['percentage_reoccurring_values'] = 0.0
            
        try:
            features['percentage_reoccurring_datapoints'] = percentage_of_reoccurring_datapoints_to_all_datapoints(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate percentage_of_reoccurring_datapoints_to_all_datapoints: {str(e)}")
            features['percentage_reoccurring_datapoints'] = 0.0
            
        try:
            features['benford_correlation'] = benford_correlation(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate benford_correlation: {str(e)}")
            features['benford_correlation'] = 0.0
            
        try:
            time_reversal = time_reversal_asymmetry_statistic(signal, lag=1)
            features['time_reversal_asymmetry'] = time_reversal
        except Exception as e:
            logger.debug(f"Failed to calculate time_reversal_asymmetry_statistic: {str(e)}")
            features['time_reversal_asymmetry'] = 0.0
            
    else:
        # Original features for non-noise data (keep the existing code)
        try:
            mean_value = feature_calculators.mean(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate mean: {str(e)}")
            mean_value = 0.0
            
        try:
            variance = feature_calculators.variance(signal)
            features['variance_ratio'] = (variance / mean_value) if mean_value != 0 else 0
        except Exception as e:
            logger.debug(f"Failed to calculate variance_ratio: {str(e)}")
            features['variance_ratio'] = 0.0
            
        # Calculate std_perc with error handling
        try:
            std = feature_calculators.standard_deviation(signal)
            features['std_ratio'] = (std / mean_value) if mean_value != 0 else 0
        except Exception as e:
            logger.debug(f"Failed to calculate std_ratio: {str(e)}")
            features['std_ratio'] = 0.0
            
        try:
            features['skewness'] = feature_calculators.skewness(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate skewness: {str(e)}")
            features['skewness'] = 0.0
        
        try:
            features['approx_entropy'] = feature_calculators.approximate_entropy(signal, m=2, r=0.3)
        except Exception as e:
            logger.debug(f"Failed to calculate approx_entropy: {str(e)}")
            features['approx_entropy'] = 0.0
            
        try:
            features['autocorr'] = feature_calculators.autocorrelation(signal, lag=1)
            if pd.isna(features['autocorr']):
                features['autocorr'] = 0.0
        except Exception as e:
            logger.debug(f"Failed to calculate autocorr: {str(e)}")
            features['autocorr'] = 0.0
            
        try:
            features['kurtosis'] = feature_calculators.kurtosis(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate kurtosis: {str(e)}")
            features['kurtosis'] = 0.0
        
        try:
            features['lempev'] = feature_calculators.lempel_ziv_complexity(signal, bins=20)
        except Exception as e:
            logger.debug(f"Failed to calculate lempev: {str(e)}")
            features['lempev'] = 0.0
            
        try:
            features['longest_above_mean'] = feature_calculators.longest_strike_above_mean(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate longest_above_mean: {str(e)}")
            features['longest_above_mean'] = 0
            
        try:
            features['longest_below_mean'] = feature_calculators.longest_strike_below_mean(signal)
        except Exception as e:
            logger.debug(f"Failed to calculate longest_below_mean: {str(e)}")
            features['longest_below_mean'] = 0
            
        try:
            features['n_peaks'] = feature_calculators.number_peaks(signal, n=1)
        except Exception as e:
            logger.debug(f"Failed to calculate n_peaks: {str(e)}")
            features['n_peaks'] = 0
        
        try:
            features['permutation_entropy'] = feature_calculators.permutation_entropy(signal, tau=1, dimension=4)
        except Exception as e:
            logger.debug(f"Failed to calculate permutation_entropy: {str(e)}")
            features['permutation_entropy'] = 0.0
    
    return features

def extract_window_features(df, window_size=10, target_column=None):
    """
    Extract features from a sliding window of the data with preprocessing.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the signal data
        window_size (int): Size of the window for feature extraction
        target_column (str, optional): Specific column to extract features from
        
    Returns:
        tuple: (pandas.DataFrame of features, list of feature names)
    """
    # Validate target_column exists in data
    if target_column and target_column not in df.columns:
        logger.error(f"Specified column '{target_column}' not found in data")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        return pd.DataFrame(), []
    
    all_features = []
    feature_names = [] 
    
    epsilon = 0.08 
    filtered_df = filter_data(df, window_size=window_size, epsilon=epsilon, target_column=target_column)
    
    for i in range(0, len(filtered_df), window_size//2):  # 50% overlap
        window = filtered_df.iloc[i:i+window_size].copy()
        
        # Skip if window is too small
        if len(window) < window_size:
            continue
        
        # Skip windows with insufficient variance in the target column
        if target_column:
            if target_column in window.columns:
                # Check if data has enough variation to be meaningful
                signal = window[target_column].values
                # Skip if standard deviation is extremely low (nearly constant signal)
                if np.std(signal) < 1e-5:
                    continue
                # Skip if range is too small (signal barely changes)
                if np.ptp(signal) < 1e-4:  # ptp = peak to peak
                    continue
                # Skip if most values are repeated (low information content)
                unique_ratio = len(np.unique(signal)) / len(signal)
                if unique_ratio < 0.1:  # Less than 10% of values are unique
                    continue
                # Skip if signal is mostly zeros
                if np.mean(signal == 0) > 0.7:  # More than 70% zeros
                    continue

        initial_size = window_size * len(window.columns)
                
        window = window.dropna()
        
        non_null_count_after = window.count().sum()
        
        # Skip window if it has less than 60% of its original data after removing nulls
        if non_null_count_after < 0.6 * initial_size:
            continue
            
        window_features = {}
        
        # If target_column is specified, only analyze that column
        if target_column:
            if target_column in filtered_df.columns:
                signal = window[target_column].values
                if len(signal) > 0:
                    # Determine if the signal is noise
                    is_noise = "noise" in target_column.lower()
                    
                    features = extract_features(signal, is_noise=is_noise)
                    if features:
                        for fname, fval in features.items():
                            feature_name = f"{target_column}_{fname}"
                            window_features[feature_name] = fval
                            if feature_name not in feature_names:
                                feature_names.append(feature_name)
        else:
            # Otherwise, analyze all columns by category
            for category, columns in SIGNAL_TYPES.items():
                for col in columns:
                    if col in filtered_df.columns:
                        signal = window[col].values  
                        if len(signal) > 0:
                            # Determine if the signal is noise
                            is_noise = "noise" in col.lower()
                            
                            features = extract_features(signal, is_noise=is_noise)
                            if features:
                                for fname, fval in features.items():
                                    feature_name = f"{col}_{fname}"
                                    window_features[feature_name] = fval
                                    if feature_name not in feature_names:
                                        feature_names.append(feature_name)
        
        if window_features:
            all_features.append(window_features)
    
    if not all_features:
        logger.warning("No features extracted")
        return pd.DataFrame(), feature_names
        
    return pd.DataFrame(all_features), feature_names


def clean_features_dataframe(combined_df):
    """
    Clean up the features dataframe by:
    1. Dropping columns that contain any NaN values
    2. Dropping columns where more than 20% of values are zero
    
    Args:
        combined_df (pandas.DataFrame): The dataframe with extracted features
        
    Returns:
        pandas.DataFrame: Cleaned dataframe with problematic columns removed
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger('galileo.features')
    
    # Make a copy to avoid modifying the original
    cleaned_df = combined_df.copy()
    
    # Get initial column count for reporting
    initial_column_count = len(cleaned_df.columns)
    
    # Step 1: Drop columns that contain any NaN values
    columns_with_nan = cleaned_df.columns[cleaned_df.isna().any()].tolist()
    if columns_with_nan:
        cleaned_df = cleaned_df.drop(columns=columns_with_nan)
        logger.info(f"Dropped {len(columns_with_nan)} columns containing NaN values")
        logger.debug(f"NaN columns dropped: {columns_with_nan}")
    
    # Step 2: Drop columns where more than 20% of values are zero
    # Get numeric columns only
    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Calculate the percentage of zeros in each column
    zero_percentages = (cleaned_df[numeric_cols] == 0).mean()
    
    # Find columns where more than 20% of values are zero
    high_zero_cols = zero_percentages[zero_percentages > 0.2].index.tolist()
    
    if high_zero_cols:
        cleaned_df = cleaned_df.drop(columns=high_zero_cols)
        logger.info(f"Dropped {len(high_zero_cols)} columns where >20% of values are zero")
        logger.debug(f"High-zero columns dropped: {high_zero_cols}")
    
    # Preserve the 'real' column if it exists (for classification)
    if 'real' in combined_df.columns and 'real' not in cleaned_df.columns:
        cleaned_df['real'] = combined_df['real']
    
    # Report final results
    final_column_count = len(cleaned_df.columns)
    removed_count = initial_column_count - final_column_count
    
    logger.info(f"Feature cleaning complete: Removed {removed_count}/{initial_column_count} columns ({removed_count/initial_column_count:.1%})")
    logger.info(f"Remaining features: {final_column_count}")
    
    return cleaned_df

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
    try:
        # Check if the directory exists
        validate_directory(data_directory)
        
        # Generate default output filename if not provided
        if output_file is None:
            output_file = f"combined_{target_column}_features.csv"
        
        # Use target_column as the output column prefix if not specified
        if output_column_prefix is None:
            output_column_prefix = target_column
        else:
            logger.info(f"Using '{output_column_prefix}' as prefix for feature names instead of '{target_column}'")
        
        # Find all CSV files in the directory
        csv_files = [f for f in os.listdir(data_directory) if f.lower().endswith('.csv')]
        
        if not csv_files:
            logger.error(f"No CSV files found in '{data_directory}'")
            return False
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        logger.info(f"Extracting features from column: {target_column}")
        
        # List to store all features DataFrames
        all_features = []
        
        # Process each CSV file
        for i, file_name in enumerate(csv_files, 1):
            file_path = os.path.join(data_directory, file_name)
            logger.info(f"Processing file {i}/{len(csv_files)}: {file_name}")
            
            # Load data
            df = load_data(file_path)
            if df is None:
                logger.warning(f"Skipping {file_name} due to loading error")
                continue
            
            # Check if target column exists
            if target_column not in df.columns:
                logger.warning(f"Column '{target_column}' not found in {file_name}")
                logger.info(f"Available columns: {', '.join(df.columns)}")
                continue
                
            # Extract features using the same logic as in galileo
            features_df, feature_names = extract_window_features(df, window_size=window_size, target_column=target_column)
            
            if features_df.empty:
                logger.warning(f"No features extracted from {file_name}")
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
                            filter_type=filter_type,
                            window_size=noise_window_size,
                            cutoff=cutoff,
                            fs=fs,
                            poly_order=noise_poly_order,
                            keep_noise_only=True,
                            target_column=target_column
                        )
                    except Exception as e:
                        logger.warning(f"Error extracting noise: {str(e)}")
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
                    
                    logger.info(f"Added {len(noise_df.columns)} noise features")
            
            # Add binary label based on filename
            # 1 if "EPIC" is in the filename, 0 otherwise
            is_real = 1 if "EPIC" in file_name else 0
            features_df['real'] = is_real
            logger.info(f"Label: {'Real (1)' if is_real else 'Not Real (0)'}")
            
            # Add to the collection
            all_features.append(features_df)
            
            logger.info(f"Extracted {len(features_df)} windows with {len(features_df.columns) - 1} features")
        
        # Check if we have any features
        if not all_features:
            logger.error("No features were extracted from any of the files")
            return False
        
        # Combine all features into a single DataFrame
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Clean the features dataframe
        combined_df = clean_features_dataframe(combined_df)
        
        # Count the number of real and not real samples
        real_count = combined_df['real'].sum()
        not_real_count = len(combined_df) - real_count
        logger.info(f"\nData distribution:")
        logger.info(f"  Real (1): {real_count} windows ({real_count/len(combined_df)*100:.2f}%)")
        logger.info(f"  Not Real (0): {not_real_count} windows ({not_real_count/len(combined_df)*100:.2f}%)")
        
        # Save to CSV
        combined_df.to_csv(output_file, index=False)
        logger.info(f"\nSuccessfully saved combined features to '{output_file}'")
        logger.info(f"Total windows: {len(combined_df)}")
        logger.info(f"Total features: {len(combined_df.columns) - 1}")  # -1 for the 'real' label column
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing CSV files: {str(e)}")
        raise FeatureExtractionError(f"Failed to process CSV files: {str(e)}")