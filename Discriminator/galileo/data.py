"""
Data loading and preprocessing functions for the Galileo framework.
"""

import pandas as pd
import numpy as np
import scipy.signal as signal
import logging
from galileo.core import DataLoadError, is_numeric_column

logger = logging.getLogger('galileo.data')

def load_data(file_path):
    """
    Load a CSV file and convert timestamp to datetime if present.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
        
    Raises:
        DataLoadError: If file loading fails
    """
    try:
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise DataLoadError(f"Failed to load data from {file_path}: {str(e)}")


def filter_window(window, epsilon=0.1, target_column=None):
    """
    Filter a window of data by identifying values that are within epsilon of the mean.
    
    Args:
        window (pandas.DataFrame): Window of data to filter
        epsilon (float): Tolerance as a percentage
        target_column (str, optional): Specific column to filter. If provided, only this column will be filtered.
        
    Returns:
        pandas.DataFrame: Mask of values to keep
    """
    filter_columns = []
    
    # If target column is specified, only filter that column
    if target_column and target_column in window.columns:
        filter_columns = [target_column]
    else:
        # Otherwise, filter all relevant columns EXCEPT noise columns
        if 'frequency' in window.columns:
            filter_columns.append('frequency')
        
        for prefix in ['V', 'C']:
            for i in range(1, 4):
                col = f"{prefix}{i}"
                col_lower = col.lower()
                matching_cols = [c for c in window.columns if c.lower() == col_lower]
                if matching_cols:
                    filter_columns.extend(matching_cols)
        
        power_cols = [col for col in window.columns if 'power' in col.lower()]
        filter_columns.extend(power_cols)
    
    # Exclude any noise columns from filtering
    filter_columns = [col for col in filter_columns if '_noise_raw' not in col]
    
    # Make sure columns exist and are numeric
    filter_columns = [col for col in filter_columns if col in window.columns]
    filter_columns = [col for col in filter_columns if is_numeric_column(window, col)]
    
    if not filter_columns:
        logger.warning("No valid columns to filter on.")
        return pd.DataFrame(True, index=window.index, columns=window.columns)
    
    keep_mask = pd.DataFrame(True, index=window.index, columns=window.columns)
    
    for column in filter_columns:
        if window[column].isna().all() or (window[column] == 0).all():
            continue
            
        avg = window[column].mean()
        
        lower_bound = avg - abs(avg * epsilon)
        upper_bound = avg + abs(avg * epsilon)
        
        within_range = (window[column] >= lower_bound) & (window[column] <= upper_bound)
        
        keep_mask[column] = within_range
    
    return keep_mask

def filter_data(df, window_size=10, epsilon=0.1, target_column=None):
    """
    Filter data by applying window-based filtering.
    
    Args:
        df (pandas.DataFrame): Data to filter
        window_size (int): Size of the window for filtering
        epsilon (float): Tolerance as a percentage
        target_column (str, optional): Specific column to filter. If provided, only this column will be filtered
                                      and rows will only be dropped if this column contains null values.
        
    Returns:
        pandas.DataFrame: Filtered data
    """
    filtered_df = df.copy()
    
    total_windows = 0
    total_cells = 0
    nullified_cells = 0
    
    # Only filter the target column, not its noise column
    filter_columns = [target_column] if target_column else []
    
    for i in range(0, len(df), window_size):
        window = df.iloc[i:i+window_size]
        
        if len(window) < window_size:
            continue
        
        total_windows += 1
        
        keep_mask = filter_window(window, epsilon, target_column)
        
        # For single column analysis, only operate on that column (not noise)
        if target_column:
            if target_column in keep_mask.columns:
                total_cells += len(window[target_column])
                nullified_values = (~keep_mask[target_column]).sum()
                nullified_cells += nullified_values
                
                filtered_df.loc[window.index[~keep_mask[target_column]], target_column] = np.nan
                
                # If a corresponding noise column exists, nullify noise values at the same positions
                # This keeps noise and signal aligned
                noise_column = f"{target_column}_noise_raw"
                if noise_column in filtered_df.columns:
                    filtered_df.loc[window.index[~keep_mask[target_column]], noise_column] = np.nan
        else:
            # For multiple column analysis, operate on all non-noise columns
            for col in window.columns:
                if col in keep_mask.columns and '_noise_raw' not in col:
                    total_cells += len(window[col])
                    nullified_values = (~keep_mask[col]).sum()
                    nullified_cells += nullified_values
                    
                    filtered_df.loc[window.index[~keep_mask[col]], col] = np.nan
                    
                    # If a corresponding noise column exists, nullify noise values at the same positions
                    noise_column = f"{col}_noise_raw"
                    if noise_column in filtered_df.columns:
                        filtered_df.loc[window.index[~keep_mask[col]], noise_column] = np.nan
    
    logger.info(f"Filtered {nullified_cells}/{total_cells} cells ({nullified_cells/total_cells*100:.2f}%) "
                f"across {total_windows} windows")
    
    return filtered_df

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
    # Initialize state and covariance
    x_hat = data[0]  # Initial state estimate
    P = 1.0  # Initial covariance estimate
    
    # Allocate space for filtered data
    filtered_data = np.zeros_like(data)
    
    # Kalman filter loop
    for i, measurement in enumerate(data):
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

def moving_average_filter(data, window_size):
    """
    Apply a moving average filter to the data where each point is replaced by
    the average of values in the range from x-a to x+a, where a is the square root
    of the window_size.
    
    Args:
        data (numpy.ndarray): Data to filter
        window_size (int): Parameter to determine the window range
        
    Returns:
        numpy.ndarray: Filtered data
    """
    # Ensure window_size is positive
    window_size = max(1, window_size)
    
    # Calculate 'a' as the integer part of the square root of window_size
    a = int(np.sqrt(window_size))
    
    # Create an output array with the same shape as the input
    filtered_data = np.zeros_like(data)
    
    # For each position in the data
    for i in range(len(data)):
        # Calculate window start and end indices, handling boundary cases
        start_idx = max(0, i - a)
        end_idx = min(len(data), i + a + 1)  # Add 1 because slicing is exclusive for the end index
        
        # Calculate the average of values in this window
        window_values = data[start_idx:end_idx]
        filtered_data[i] = np.mean(window_values)
    
    return filtered_data
        
def butterworth_filter(data, cutoff, fs, order=4):
    """
    Apply a Butterworth low-pass filter to the data.
    
    Args:
        data (numpy.ndarray): Data to filter
        cutoff (float): Cutoff frequency
        fs (float): Sampling frequency
        order (int): Filter order
        
    Returns:
        numpy.ndarray: Filtered data
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


def savgol_filter(data, window_size, poly_order):
    """
    Apply a Savitzky-Golay filter to the data.
    
    Args:
        data (numpy.ndarray): Data to filter
        window_size (int): Size of the window
        poly_order (int): Order of the polynomial
        
    Returns:
        numpy.ndarray: Filtered data
    """
    return signal.savgol_filter(data, window_size, poly_order)

def extract_noise_signal(df, filter_type='savgol', window_size=5, cutoff=0.1, fs=1.0, 
                        poly_order=2, keep_noise_only=True, target_column=None,
                        process_variance=1e-5, measurement_variance=1e-1):
    """
    Extract noise from signals using various filtering methods.
    
    Args:
        df (pandas.DataFrame): Data containing signals
        filter_type (str): Type of filter to use ('moving_average', 'butterworth', 'savgol', 'kalman')
        window_size (int): Size of the window for filtering
        cutoff (float): Cutoff frequency for Butterworth filter
        fs (float): Sampling frequency for Butterworth filter
        poly_order (int): Polynomial order for Savitzky-Golay filter
        keep_noise_only (bool): If True, return only the noise component
        target_column (str, optional): Specific column to extract noise from
        process_variance (float): Process variance parameter for Kalman filter
        measurement_variance (float): Measurement variance parameter for Kalman filter
        
    Returns:
        pandas.DataFrame: Data with extracted noise signals
    """
    noise_signals = pd.DataFrame(index=df.index)

    # Clean the data just in case
    df_clean = df.dropna().copy()
    
    if df_clean.empty:
        logger.warning("No valid data for noise extraction after dropping NA values")
        return noise_signals
    
    # Select columns to process
    columns_to_process = [target_column] if target_column and target_column in df_clean.columns else df_clean.columns
    
    # Noise extraction on each column
    for column in columns_to_process:
        if column == 'timestamp' or column == 'date':
            noise_signals[column] = df_clean[column]
            continue
            
        if not is_numeric_column(df_clean, column):
            continue
        
        data = df_clean[column].values
        
        if filter_type == 'moving_average':
            filtered_signal = moving_average_filter(data, window_size=window_size)
        elif filter_type == 'butterworth':
            filtered_signal = butterworth_filter(data, cutoff=cutoff, fs=fs)
        elif filter_type == 'savgol':
            # Ensure window_size is odd for savgol
            if window_size % 2 == 0:
                window_size += 1
            filtered_signal = savgol_filter(data, window_size=window_size, poly_order=poly_order)
        elif filter_type == 'kalman':
            filtered_signal = kalman_filter(data, 
                                          process_variance=process_variance, 
                                          measurement_variance=measurement_variance)
        else:
            logger.error(f"Unknown filter type: {filter_type}")
            continue

        if keep_noise_only:
            noise_signals[column] = data - filtered_signal
        else:
            noise_signals[column] = filtered_signal
            
    return noise_signals