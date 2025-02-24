import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Analyze time series data from a CSV file with voltage, current, and power measurements.'
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to the CSV file to analyze'
    )
    parser.add_argument(
        '--window',
        type=float,
        default=5.0,
        help='Analysis window size in seconds (default: 5.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to store results (default: results)'
    )
    return parser.parse_args()

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['timestamp'])
    return df

def calculate_snr(signal):
    if len(signal) == 0:
        return float('inf')
    mean = np.mean(signal)
    variance = np.var(signal)
    if variance == 0:
        return float('inf')
    return 10 * np.log10((mean**2) / variance)

def analyze_window(data, measurements):
    results = {}
    for measure_type, columns in measurements.items():
        for col in columns:
            if col in data.columns:
                signal = data[col].dropna().values
                if len(signal) > 0:
                    mean_val = np.mean(signal)
                    std_val = np.std(signal)
                    var_val = np.var(signal)
                    results[f"{col}_mean"] = mean_val
                    results[f"{col}_std"] = std_val
                    results[f"{col}_std_pct"] = (std_val / mean_val) * 100 if mean_val != 0 else 0
                    results[f"{col}_var"] = var_val
                    results[f"{col}_snr"] = calculate_snr(signal)
    return results

def plot_analysis_results(results_df, measurements, output_dir):
    plt.style.use('bmh')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    colors = plt.cm.tab10(np.linspace(0, 1, 3))
    fig_size = (15, 15)
    
    for measure_type, columns in measurements.items():
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=fig_size)
        fig.suptitle(f'{measure_type.title()} Analysis over Time')
        
        for idx, col in enumerate(columns):
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"
            std_pct_col = f"{col}_std_pct"
            var_col = f"{col}_var"
            snr_col = f"{col}_snr"
            
            if mean_col in results_df.columns:
                q1 = results_df[std_pct_col].quantile(0.25)
                q3 = results_df[std_pct_col].quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                lower_bound = max(0, q1 - 1.5 * iqr)
                
                ax1.plot(results_df['window_start'], results_df[mean_col], 
                        label=col, color=colors[idx])
                ax1.fill_between(results_df['window_start'],
                               results_df[mean_col] - results_df[std_col],
                               results_df[mean_col] + results_df[std_col],
                               alpha=0.2, color=colors[idx])
                ax1.set_ylabel('Mean ± Std')
                ax1.set_ylim(bottom=0) 
                ax1.legend()
                
                ax2.plot(results_df['window_start'], results_df[std_col],
                        label=col, color=colors[idx])
                ax2.set_ylabel('Standard Deviation')
                ax2.set_ylim(bottom=0)  
                ax2.legend()
                
                ax3.plot(results_df['window_start'], results_df[std_pct_col].clip(lower_bound, upper_bound),
                        label=col, color=colors[idx])
                ax3.set_ylabel('Standard Deviation (%) [clipped]')
                ax3.set_ylim(bottom=0)  
                ax3.legend()
                
                ax4.plot(results_df['window_start'], results_df[var_col],
                        label=col, color=colors[idx])
                ax4.set_ylabel('Variance')
                ax4.set_ylim(bottom=0)  
                ax4.legend()
                
                ax5.plot(results_df['window_start'], results_df[snr_col],
                        label=col, color=colors[idx])
                ax5.set_ylabel('SNR (dB)')
                ax5.set_ylim(bottom=0) 
                ax5.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/analysis_{measure_type}_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
def main():
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        df = load_data(args.csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find file '{args.csv_file}'")
        return
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return
    
    measurements = {
        'voltage': ['V1', 'V2', 'V3'],
        'current': ['C1', 'C2', 'C3'],
        'voltage_diff': ['V1_V2', 'V2_V3', 'V1_V3'],
        'power': ['power_real', 'power_effective', 'power_apparent'],
        'frequency': ['frequency']
    }
    
    window_size = pd.Timedelta(seconds=args.window)
    analysis_results = []
    
    start_time = df['date'].min()
    end_time = df['date'].max()
    current_time = start_time
    
    print(f"Starting analysis from {start_time} to {end_time}")
    print(f"Using window size of {args.window} seconds")
    
    while current_time < end_time:
        window_end = current_time + window_size
        window_data = df[(df['date'] >= current_time) & (df['date'] < window_end)]
        
        if not window_data.empty:
            result = analyze_window(window_data, measurements)
            result['window_start'] = current_time
            result['window_end'] = window_end
            analysis_results.append(result)
        
        current_time += window_size
    
    results_df = pd.DataFrame(analysis_results)
    results_df.to_csv(f'{args.output_dir}/analysis_results.csv', index=False)
    print(f"Analysis complete. Results saved to {args.output_dir}/analysis_results.csv")
    
    plot_analysis_results(results_df, measurements, args.output_dir)
    print(f"Plots generated in {args.output_dir}/ folder")
    
    stats_output = []
    for measure_type, columns in measurements.items():
        stats_output.append(f"\n{measure_type.upper()} MEASUREMENTS:")
        for col in columns:
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"
            std_pct_col = f"{col}_std_pct"
            var_col = f"{col}_var"
            snr_col = f"{col}_snr"
            if mean_col in results_df.columns:
                stats_output.extend([
                    f"\n{col}:",
                    f"Mean of means: {results_df[mean_col].mean():.2f}",
                    f"Mean of std: {results_df[std_col].mean():.2f}",
                    f"Mean of std (%): {results_df[std_pct_col].mean():.2f}",
                    f"Mean of variance: {results_df[var_col].mean():.2f}",
                    f"Mean SNR (dB): {results_df[snr_col].mean():.2f}"
                ])
    
    with open(f'{args.output_dir}/analysis_statistics.txt', 'w') as f:
        f.write('\n'.join(stats_output))

if __name__ == "__main__":
    main()