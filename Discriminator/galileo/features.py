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

def extract_window_features(df, window_size=10, target_column=None, noise_column=None, epsilon=0.1):
    """
    Extract features from a sliding window of the data with preprocessing.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the signal data
        window_size (int): Size of the window for feature extraction
        target_column (str, optional): Specific column to extract features from
        noise_column (str, optional): Corresponding noise column for target_column
        epsilon (float): Epsilon value for filtering outliers
        
    Returns:
        tuple: (pandas.DataFrame of features, list of feature names)
    """
    # Validate target_column exists in data
    if target_column and target_column not in df.columns:
        logger.error(f"Specified column '{target_column}' not found in data")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        return pd.DataFrame(), []
    
    # Validate noise_column exists if specified
    if noise_column and noise_column not in df.columns:
        logger.warning(f"Specified noise column '{noise_column}' not found in data")
        noise_column = None
    
    all_features = []
    feature_names = [] 
    
    # Filter data using the specified epsilon value
    logger.info(f"Filtering data with epsilon={epsilon}")
    filtered_df = filter_data(df, window_size=window_size, epsilon=epsilon, target_column=target_column)
    
    for i in range(0, len(filtered_df), window_size//2):  # 50% overlap
        window = filtered_df.iloc[i:i+window_size].copy()
        
        # Skip if window is too small
        if len(window) < window_size:
            continue
        
        # Skip windows with insufficient variance in the target column
        if target_column and target_column in window.columns:
            # Skip if there's not enough data
            if window[target_column].isna().sum() > window_size * 0.4:  # More than 40% NaN
                continue
                
            # Check if data has enough variation to be meaningful
            signal = window[target_column].dropna().values
            
            # Skip if no valid data
            if len(signal) == 0:
                continue
                
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
                
        if target_column:
            window = window.dropna(subset=[target_column])
            if noise_column:
                window = window.dropna(subset=[noise_column])
        else:
            # Import SIGNAL_TYPES from core module
            from galileo.core import SIGNAL_TYPES
            
            columns_to_check = []
            for category, cols in SIGNAL_TYPES.items():
                columns_to_check.extend([col for col in cols if col in window.columns])
            window = window.dropna(subset=columns_to_check, how='all')

        
        non_null_count_after = window.count().sum()
        
        # Skip window if it has less than 60% of its original data after removing nulls
        if non_null_count_after < 0.6 * initial_size:
            continue
            
        window_features = {}
        
        # If target_column is specified, extract features from it
        if target_column:
            if target_column in filtered_df.columns:
                signal = window[target_column].values
                if len(signal) > 0:
                    features = extract_features(signal, is_noise=False)
                    if features:
                        for fname, fval in features.items():
                            feature_name = f"{target_column}_{fname}"
                            window_features[feature_name] = fval
                            if feature_name not in feature_names:
                                feature_names.append(feature_name)
            
            # Extract features from pre-extracted noise if available
            if noise_column and noise_column in filtered_df.columns:
                noise_signal = window[noise_column].values
                if len(noise_signal) > 0:
                    noise_features = extract_features(noise_signal, is_noise=True)
                    if noise_features:
                        for fname, fval in noise_features.items():
                            feature_name = f"noise_{target_column}_{fname}"
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
                            features = extract_features(signal, is_noise=False)
                            if features:
                                for fname, fval in features.items():
                                    feature_name = f"{col}_{fname}"
                                    window_features[feature_name] = fval
                                    if feature_name not in feature_names:
                                        feature_names.append(feature_name)
                        
                        # Check for corresponding noise column
                        noise_col = f"{col}_noise_raw"
                        if noise_col in filtered_df.columns:
                            noise_signal = window[noise_col].values
                            if len(noise_signal) > 0:
                                noise_features = extract_features(noise_signal, is_noise=True)
                                if noise_features:
                                    for fname, fval in noise_features.items():
                                        feature_name = f"noise_{col}_{fname}"
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
    critical_cols = [col for col in cleaned_df.columns if any(term in col for term in CRITICAL_FEATURES)]
    
    # If we found any critical columns, filter rows where all these values are near zero
    if critical_cols:
        # Identify rows where all critical features are very close to zero
        all_critical_near_zero = (cleaned_df[critical_cols].abs() < 1e-4).all(axis=1)
        cleaned_df = cleaned_df[~all_critical_near_zero]
        logger.info(f"Removed {all_critical_near_zero.sum()} rows where all critical features are near zero")
    
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
        logger.info(f"Found {low_quality_count} low quality rows ({low_quality_count/len(cleaned_df)*100:.2f}%)")
        logger.info("These rows have more than 80% of features as zeros or null values")
        
        # Remove these rows
        cleaned_df = cleaned_df[~low_quality_rows]
        logger.info(f"Removed {low_quality_count} low quality rows")
    
    return cleaned_df


def process_csv_files(data_directory, output_file=None, target_column=None, window_size=10, 
                  extract_noise=True, filter_type='savgol', cutoff=0.1, fs=1.0, poly_order=2, 
                  output_column_prefix=None, process_variance=1e-5, measurement_variance=1e-1,
                  epsilon=0.1, all_columns=False):
    """
    Process all CSV files in the given directory, extract features from specified column or all columns in parallel,
    and combine them into a single CSV file.
    
    Args:
        data_directory (str): Directory containing CSV files
        output_file (str, optional): Path to save the combined features. If None, auto-generated based on column name
        target_column (str, optional): Column to extract features from. If None and all_columns is True, features are extracted from all suitable columns
        window_size (int): Size of the window for feature extraction
        extract_noise (bool): Whether to extract noise features
        filter_type (str): Type of filter to use for noise extraction ('moving_average', 'butterworth', 'savgol', 'kalman')
        cutoff (float): Cutoff frequency for Butterworth filter
        fs (float): Sampling frequency for Butterworth filter
        poly_order (int): Polynomial order for Savitzky-Golay filter
        output_column_prefix (str, optional): Prefix to use for output column names. If None, uses target_column
        epsilon (float): Epsilon value for filtering outliers
        all_columns (bool): Whether to extract features from all suitable columns in parallel
        process_variance (float): Process variance parameter for Kalman filter
        measurement_variance (float): Measurement variance parameter for Kalman filter
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if the directory exists
        validate_directory(data_directory)
        
        # Import required functions from core
        from galileo.core import is_numeric_column, SIGNAL_TYPES
        
        # Generate default output filename if not provided
        if output_file is None:
            if target_column:
                output_file = f"combined_{target_column}_features.csv"
            elif all_columns:
                output_file = "combined_all_columns_features.csv"
            else:
                output_file = "combined_features.csv"
        
        # Find all CSV files in the directory
        csv_files = [f for f in os.listdir(data_directory) if f.lower().endswith('.csv')]
        
        if not csv_files:
            logger.error(f"No CSV files found in '{data_directory}'")
            return False
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
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
            
            # Identify columns to process
            if all_columns:
                # Get all available numeric columns that are relevant for signal analysis
                target_columns = []
                for category, cols in SIGNAL_TYPES.items():
                    for col in cols:
                        if col in df.columns and is_numeric_column(df, col):
                            target_columns.append(col)
                
                # If no columns were found from SIGNAL_TYPES, select all numeric columns except timestamp and date
                if not target_columns:
                    for col in df.columns:
                        if col not in ['timestamp', 'date'] and is_numeric_column(df, col):
                            target_columns.append(col)
                
                logger.info(f"Found {len(target_columns)} suitable columns for feature extraction: {', '.join(target_columns)}")
            else:
                # Check if target column exists
                if target_column not in df.columns:
                    logger.warning(f"Column '{target_column}' not found in {file_name}")
                    logger.info(f"Available columns: {', '.join(df.columns)}")
                    continue
                target_columns = [target_column]
            
            # Extract noise from the entire dataset for all relevant columns
            noise_columns = {}
            if extract_noise:
                noise_window_size = max(window_size//5, 3)  # Smaller window for noise extraction
                noise_poly_order = min(poly_order, noise_window_size - 1)
                
                for col in target_columns:
                    logger.info(f"Extracting noise from column '{col}'")
                    noise_df = extract_noise_signal(
                        df, 
                        filter_type=filter_type,
                        window_size=noise_window_size,
                        cutoff=cutoff,
                        fs=fs,
                        poly_order=noise_poly_order,
                        keep_noise_only=True,
                        target_column=col
                    )
                    
                    # Add noise column to original dataframe
                    if not noise_df.empty and col in noise_df.columns:
                        noise_column = f"{col}_noise_raw"
                        df[noise_column] = noise_df[col]
                        noise_columns[col] = noise_column
                        logger.info(f"Added noise column '{noise_column}' to dataframe")
                    else:
                        logger.warning(f"Noise extraction failed for column '{col}', proceeding without noise features")
            
            # If processing all columns in parallel, we need to extract features from windows
            # that contain data from all target columns
            if all_columns:
                logger.info(f"Extracting features from all columns in parallel with epsilon={epsilon}")
                
                # Filter data using the specified epsilon value
                filtered_df = filter_data(df, window_size=window_size, epsilon=epsilon)
                
                # Process windows with 50% overlap
                window_features_list = []
                for i in range(0, len(filtered_df), window_size//2):
                    window = filtered_df.iloc[i:i+window_size].copy()
                    
                    # Skip if window is too small
                    if len(window) < window_size:
                        continue
                    
                    window_features = {}
                    valid_window = True
                    
                    # Check if we have sufficient valid data for all target columns
                    for col in target_columns:
                        if col not in window.columns or window[col].isna().all():
                            valid_window = False
                            break
                        
                        # Check if data has enough variation to be meaningful
                        signal = window[col].dropna().values
                        if len(signal) < window_size * 0.6:  # Require at least 60% valid data
                            valid_window = False
                            break
                        
                        # Skip if standard deviation is extremely low (nearly constant signal)
                        if np.std(signal) < 1e-5:
                            valid_window = False
                            break
                            
                        # Skip if range is too small (signal barely changes)
                        if np.ptp(signal) < 1e-4:  # ptp = peak to peak
                            valid_window = False
                            break
                    
                    if not valid_window:
                        continue
                    
                    # Extract features for each target column
                    for col in target_columns:
                        signal = window[col].values
                        
                        # Extract regular features
                        features = extract_features(signal, is_noise=False)
                        if features:
                            for fname, fval in features.items():
                                feature_name = f"{col}_{fname}"
                                window_features[feature_name] = fval
                        
                        # Extract noise features if available
                        if col in noise_columns:
                            noise_col = noise_columns[col]
                            if noise_col in window.columns:
                                noise_signal = window[noise_col].values
                                noise_features = extract_features(noise_signal, is_noise=True)
                                if noise_features:
                                    for fname, fval in noise_features.items():
                                        feature_name = f"noise_{col}_{fname}"
                                        window_features[feature_name] = fval
                    
                    if window_features:
                        window_features_list.append(window_features)
                
                if window_features_list:
                    # Create a DataFrame with all features
                    features_df = pd.DataFrame(window_features_list)
                    
                    # Add metadata
                    is_real = 1 if ("EPIC" in file_name or "MORRIS" in file_name) else 0
                    features_df['real'] = is_real
                    features_df['source'] = file_name
                    
                    all_features.append(features_df)
                    logger.info(f"Extracted {len(features_df)} windows with {len(features_df.columns) - 2} features")
            else:
                # Process a single target column
                logger.info(f"Extracting features from column '{target_column}' with epsilon={epsilon}")
                col = target_column
                noise_col = noise_columns.get(col) if col in noise_columns else None
                
                features_df, _ = extract_window_features(
                    df, 
                    window_size=window_size, 
                    target_column=col,
                    noise_column=noise_col,
                    epsilon=epsilon
                )
                
                if not features_df.empty:
                    # Add metadata
                    is_real = 1 if ("EPIC" in file_name or "MORRIS" in file_name) else 0
                    features_df['real'] = is_real
                    features_df['source'] = file_name
                    
                    all_features.append(features_df)
                    logger.info(f"Extracted {len(features_df)} windows with {len(features_df.columns) - 2} features from column '{col}'")
                else:
                    logger.warning(f"No features extracted from column '{col}' in {file_name}")
        
        # Check if we have any features
        if not all_features:
            logger.error("No features were extracted from any of the files")
            return False
        
        # Combine all features into a single DataFrame
        combined_df = pd.concat(all_features, ignore_index=True)
        
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
        logger.info(f"Total features: {len(combined_df.columns) - 2}")  # -2 for the 'real' and 'source' columns
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing CSV files: {str(e)}")
        raise FeatureExtractionError(f"Failed to process CSV files: {str(e)}")