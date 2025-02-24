import pandas as pd
import numpy as np
import sys
import os

def preprocess_data(input_file):
    base_name = os.path.basename(input_file)
    output_file = 'processed_' + base_name
    
    df = pd.read_csv(input_file)
    
    df['timestamp'] = pd.to_datetime(df['Timestamp'])
    processed = pd.DataFrame()
    processed['timestamp'] = pd.to_datetime(df['Timestamp'])
    
    voltage_map = {
        'Generation.GIED1.Measurement.V1': 'V1',
        'Generation.GIED1.Measurement.V2': 'V2',
        'Generation.GIED1.Measurement.V3': 'V3'
    }
    
    current_map = {
        'Generation.GIED1.Measurement.L1_Current': 'C1',
        'Generation.GIED1.Measurement.L2_Current': 'C2',
        'Generation.GIED1.Measurement.L3_Current': 'C3'
    }
    
    power_map = {
        'Generation.GIED1.Measurement.Real': 'power_real',
        'Generation.GIED1.Measurement.Reactive': 'power_effective',
        'Generation.GIED1.Measurement.Apparent': 'power_apparent'
    }
    
    processed['frequency'] = df['Generation.GIED1.Measurement.Frequency']
    
    for old_name, new_name in voltage_map.items():
        processed[new_name] = df[old_name]
        
    for old_name, new_name in current_map.items():
        processed[new_name] = df[old_name]
        
    processed['V1_V2'] = processed['V1'] - processed['V2']
    processed['V2_V3'] = processed['V2'] - processed['V3']
    processed['V1_V3'] = processed['V1'] - processed['V3']
    
    for old_name, new_name in power_map.items():
        processed[new_name] = df[old_name]
    
    processed.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 preprocessor.py <input_csv_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
        
    if not input_file.endswith('.csv'):
        print("Error: Input file must be a CSV file")
        sys.exit(1)
        
    preprocess_data(input_file)