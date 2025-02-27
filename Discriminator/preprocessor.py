import pandas as pd
import numpy as np
import argparse
import os
import scipy.signal as signal


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def filter_window(window, epsilon=0.1):
    filter_columns = []
    
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
    
    filter_columns = [col for col in filter_columns if col in window.columns]
    filter_columns = [col for col in filter_columns if np.issubdtype(window[col].dtype, np.number)]
    
    if not filter_columns:
        print("Warning: No valid columns to filter on.")
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


def filter_data(df, window_size=10, epsilon=0.1):
    filtered_df = df.copy()
    
    total_windows = 0
    total_cells = 0
    nullified_cells = 0
    
    for i in range(0, len(df), window_size):
        window = df.iloc[i:i+window_size]
        
        if len(window) < window_size:
            continue
        
        total_windows += 1
        
        keep_mask = filter_window(window, epsilon)
        
        for col in window.columns:
            if col in keep_mask.columns:
                total_cells += len(window[col])
                nullified_values = (~keep_mask[col]).sum()
                nullified_cells += nullified_values
                
                filtered_df.loc[window.index[~keep_mask[col]], col] = np.nan
    
    return filtered_df

def moving_average_filter(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def butterworth_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)

def savgol_filter(data, window_size, poly_order):
    return signal.savgol_filter(data, window_size, poly_order)


def extract_noise_signal(df, filter, window_size=5, cutoff=0.1, fs=1.0, poly_order=2, keep_noise_only=True):
    noise_signals = pd.DataFrame(index=df.index)

    # clean the data just in case
    df.dropna(inplace=True)
    
    # noise extraction on each column
    for column in df.columns:
        if column == 'timestamp' or column == 'date':
            noise_signals[column] = df[column]
            continue
        
        data = df[column].values
        
        if filter == 'moving_average':
            noise_signals[column] = moving_average_filter(data, window_size=window_size)
        elif filter == 'butterworth':
            noise_signals[column] = butterworth_filter(data, cutoff=cutoff, fs=fs)
        elif filter == 'savgol':
            noise_signals[column] = savgol_filter(data, window_size=window_size, poly_order=poly_order)
        else:
            raise ValueError(f"Unknown filter type: {filter}")

        if keep_noise_only:
            noise_signals[column] = data - noise_signals[column]
            
    return noise_signals



def main():
    parser = argparse.ArgumentParser(description='Filter CSV data using a window-based approach.')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV file path (default: filtered_{input})')
    parser.add_argument('--window-size', '-w', type=int, default=20, help='Window size (default: 20)')
    parser.add_argument('--epsilon', '-e', type=float, default=0.05, help='Epsilon value for filtering (default: 0.05)')
    
    args = parser.parse_args()
    
    if not args.output:
        input_basename = os.path.basename(args.input)
        input_dirname = os.path.dirname(args.input)
        args.output = os.path.join(input_dirname, f"filtered_{input_basename}")
    
    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    initial_rows = len(df)
    print(f"Loaded {initial_rows} rows of data.")
    
    print(f"Filtering data with window size {args.window_size} and epsilon {args.epsilon}...")
    filtered_df = filter_data(df, args.window_size, args.epsilon)
    
    print(f"Saving filtered data to {args.output}...")
    filtered_df.to_csv(args.output, index=False)
    
    original_values = df.count().sum()
    filtered_values = filtered_df.count().sum()
    nullified_values = original_values - filtered_values
    percent_kept = (filtered_values / original_values) * 100 if original_values > 0 else 0
    
    print(f"Filtering complete:")
    print(f"  - Original total values: {original_values}")
    print(f"  - Values kept: {filtered_values}")
    print(f"  - Values nullified: {nullified_values}")
    print(f"  - Kept {percent_kept:.2f}% of the values")
    print(f"Filtered data saved to {args.output}")


if __name__ == "__main__":
    main()