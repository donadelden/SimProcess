import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.dates import DateFormatter

def plot_power_system_data(csv_file, output_folder=None, show_plots=True, dpi=300):
    """
    Plot power system simulation data from the processed CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the processed CSV file
    output_folder : str, optional
        Folder to save plot images. If None, plots won't be saved.
    show_plots : bool
        Whether to display the plots interactively
    dpi : int
        Resolution for saved plots
    """
    # Create output folder if specified
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read the data
    try:
        df = pd.read_csv(csv_file)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            print("Warning: No timestamp column found. Using row index for x-axis.")
            df['timestamp'] = range(len(df))
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Define time formatter for x-axis
    date_format = DateFormatter('%H:%M:%S')
    
    # Set the style
    plt.style.use('ggplot')
    
    # Set up the plots - using subplots for better organization
    # Figure 1: Voltages
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    if 'V1' in df.columns:
        ax1.plot(df['timestamp'], df['V1'], label='V1', color='red')
    if 'V2' in df.columns:
        ax1.plot(df['timestamp'], df['V2'], label='V2', color='green')
    if 'V3' in df.columns:
        ax1.plot(df['timestamp'], df['V3'], label='V3', color='blue')
    if 'V1_V2' in df.columns:
        ax1.plot(df['timestamp'], df['V1_V2'], label='V1-V2', color='orange', linestyle=':')
    if 'V2_V3' in df.columns:
        ax1.plot(df['timestamp'], df['V2_V3'], label='V2-V3', color='purple', linestyle=':')
    if 'V1_V3' in df.columns:
        ax1.plot(df['timestamp'], df['V1_V3'], label='V1-V3', color='brown', linestyle=':')
    
    ax1.set_title('Phase and Line-to-Line Voltages Over Time', fontsize=16)
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Voltage (V)', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    fig1.tight_layout()
    
    # Figure 2: Currents
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    if 'C1' in df.columns:
        ax2.plot(df['timestamp'], df['C1'], label='Current A', color='red')
    if 'C2' in df.columns:
        ax2.plot(df['timestamp'], df['C2'], label='Current B', color='green')
    if 'C3' in df.columns:
        ax2.plot(df['timestamp'], df['C3'], label='Current C', color='blue')
    
    ax2.set_title('Phase Currents Over Time', fontsize=16)
    ax2.set_xlabel('Time', fontsize=14)
    ax2.set_ylabel('Current (A)', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    fig2.tight_layout()
    
    # Figure 3: Frequency
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    if 'frequency' in df.columns:
        ax3.plot(df['timestamp'], df['frequency'], label='Frequency', color='purple')
        
        # Add reference line at nominal frequency (assuming 50 or 60 Hz)
        nominal_freq = 50 if df['frequency'].mean() < 55 else 60
        ax3.axhline(y=nominal_freq, color='gray', linestyle='--', alpha=0.7)
        
        # Highlight the frequency variation range
        freq_min = df['frequency'].min()
        freq_max = df['frequency'].max()
        freq_range = freq_max - freq_min
        
        # Set y-axis limits to highlight variations but not too tight
        y_margin = max(0.1, freq_range * 2)
        ax3.set_ylim([nominal_freq - y_margin, nominal_freq + y_margin])
    
    ax3.set_title('System Frequency Over Time', fontsize=16)
    ax3.set_xlabel('Time', fontsize=14)
    ax3.set_ylabel('Frequency (Hz)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    fig3.tight_layout()
    
    # Figure 4: Powers
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    if 'power_real' in df.columns:
        ax4.plot(df['timestamp'], df['power_real'], label='Real Power (P)', color='blue')
    if 'power_effective' in df.columns:
        ax4.plot(df['timestamp'], df['power_effective'], label='Reactive Power (Q)', color='red')
    if 'power_apparent' in df.columns:
        ax4.plot(df['timestamp'], df['power_apparent'], label='Apparent Power (S)', color='green')
    
    ax4.set_title('Power Measurements Over Time', fontsize=16)
    ax4.set_xlabel('Time', fontsize=14)
    ax4.set_ylabel('Power (W/VAR/VA)', fontsize=14)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    fig4.tight_layout()
    
    # Save the figures if output folder is specified
    if output_folder:
        fig1.savefig(os.path.join(output_folder, 'voltages.png'), dpi=dpi, bbox_inches='tight')
        fig2.savefig(os.path.join(output_folder, 'currents.png'), dpi=dpi, bbox_inches='tight')
        fig3.savefig(os.path.join(output_folder, 'frequency.png'), dpi=dpi, bbox_inches='tight')
        fig4.savefig(os.path.join(output_folder, 'powers.png'), dpi=dpi, bbox_inches='tight')
        print(f"Plots saved to {output_folder}")
    
    # Show the plots if requested
    if show_plots:
        plt.show()
    else:
        plt.close('all')

def analyze_noise_characteristics(csv_file):
    """
    Analyze and print statistics about the noise characteristics in the data.
    
    Parameters:
    -----------
    csv_file : str
        Path to the processed CSV file
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    print("\n=== NOISE CHARACTERISTICS ANALYSIS ===")
    
    # Analyze voltage noise
    if all(col in df.columns for col in ['V1', 'V2', 'V3']):
        v_cols = ['V1', 'V2', 'V3']
        v_mean = df[v_cols].mean().mean()
        
        # Calculate coefficient of variation for each voltage
        v_cv = df[v_cols].std() / df[v_cols].mean() * 100
        
        print("\nVOLTAGE ANALYSIS:")
        print(f"  Mean voltage: {v_mean:.2f} V")
        print(f"  Coefficient of variation (%):")
        for col, cv in v_cv.items():
            print(f"    {col}: {cv:.4f}%")
            
        # Check for impulses (spikes)
        v_zscore = (df[v_cols] - df[v_cols].mean()) / df[v_cols].std()
        impulse_threshold = 3.0  # Z-score threshold for impulses
        impulse_counts = (v_zscore.abs() > impulse_threshold).sum()
        
        print(f"  Impulse count (|z-score| > {impulse_threshold}):")
        for col, count in impulse_counts.items():
            print(f"    {col}: {count} ({count/len(df)*100:.4f}%)")
    
    # Analyze frequency noise
    if 'frequency' in df.columns:
        freq_mean = df['frequency'].mean()
        freq_std = df['frequency'].std()
        freq_min = df['frequency'].min()
        freq_max = df['frequency'].max()
        
        print("\nFREQUENCY ANALYSIS:")
        print(f"  Mean: {freq_mean:.4f} Hz")
        print(f"  Standard deviation: {freq_std:.6f} Hz")
        print(f"  Range: {freq_min:.4f} - {freq_max:.4f} Hz")
        print(f"  Max deviation from mean: {max(abs(freq_max-freq_mean), abs(freq_min-freq_mean)):.6f} Hz")
        
        # Check if there's a trend in frequency
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(df)), df['frequency'])
        print(f"  Linear trend: {slope*len(df):.6f} Hz over the entire period (p-value: {p_value:.4f})")
    
    # Analyze current noise
    if all(col in df.columns for col in ['C1', 'C2', 'C3']):
        c_cols = ['C1', 'C2', 'C3']
        c_mean = df[c_cols].mean().mean()
        
        # Calculate coefficient of variation for each current
        c_cv = df[c_cols].std() / df[c_cols].mean() * 100
        
        print("\nCURRENT ANALYSIS:")
        print(f"  Mean current: {c_mean:.2f} A")
        print(f"  Coefficient of variation (%):")
        for col, cv in c_cv.items():
            print(f"    {col}: {cv:.4f}%")
    
    print("\n===================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize power system simulation data')
    parser.add_argument('csv_file', type=str, help='Path to the processed CSV file')
    parser.add_argument('--output', type=str, default=None, help='Folder to save plot images')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots (only save)')
    parser.add_argument('--dpi', type=int, default=300, help='Resolution for saved plots')
    parser.add_argument('--analyze', action='store_true', help='Analyze noise characteristics')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_noise_characteristics(args.csv_file)
    
    plot_power_system_data(
        args.csv_file, 
        output_folder=args.output, 
        show_plots=not args.no_show,
        dpi=args.dpi
    )