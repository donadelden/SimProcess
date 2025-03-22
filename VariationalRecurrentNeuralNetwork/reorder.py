#!/usr/bin/env python3
import pandas as pd
import argparse
import os
import re

def reorder_csv_columns(input_file, output_file=None):
    """
    Reorder columns in a CSV file to a specific order and drop any other columns.
    Handles columns with suffixes like "_noise" by matching the base column name.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to save the output CSV file. 
                                    If None, will use input_file_reordered.csv
    """
    # Define the required column order
    required_columns_order = [
        "timestamp",
        "C1", "C2", "C3", 
        "V1", "V2", "V3", 
        "power_real", "power_reactive", "power_apparent",
        "frequency"
    ]
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully read {input_file}")
        print(f"Original columns: {', '.join(df.columns)}")
        
        # Specifically drop columns containing V1_V2, V2_V3, V1_V3
        columns_to_drop = [col for col in df.columns if any(pattern in col for pattern in ["V1_V2", "V2_V3", "V1_V3"])]
        if columns_to_drop:
            print(f"Dropping specific columns: {', '.join(columns_to_drop)}")
            df = df.drop(columns=columns_to_drop)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return
    
    # Create a mapping of base column names to actual column names
    column_mapping = {}
    for col in df.columns:
        # Extract the base name (before potential "_noise" or other suffixes)
        # This regex finds the match for any of our required column patterns
        for base_col in required_columns_order:
            if re.match(f"^{base_col}(_.*)?$", col):
                column_mapping[base_col] = col
                break
    
    # Filter to keep only the columns that exist in the mapping
    existing_ordered_columns = [column_mapping[base_col] for base_col in required_columns_order 
                               if base_col in column_mapping]
    
    # Check if any required columns are missing
    found_base_columns = list(column_mapping.keys())
    missing_columns = [col for col in required_columns_order if col not in found_base_columns]
    if missing_columns:
        print(f"Warning: The following required base columns are missing: {', '.join(missing_columns)}")
    
    # Check if there are any extra columns that will be dropped
    mapped_actual_columns = list(column_mapping.values())
    extra_columns = [col for col in df.columns if col not in mapped_actual_columns]
    if extra_columns:
        print(f"The following columns will be dropped: {', '.join(extra_columns)}")
    
    # Create a new dataframe with the columns in the correct order
    df_reordered = df[existing_ordered_columns]
    
    # Set the output file name if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_reordered.csv"
    
    # Save the reordered dataframe
    df_reordered.to_csv(output_file, index=False)
    print(f"Reordered CSV saved to {output_file}")
    print(f"Final columns: {', '.join(df_reordered.columns)}")
    print(f"Final order (base names): {', '.join([base for base in required_columns_order if base in found_base_columns])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorder CSV columns to a specific order and drop other columns")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("-o", "--output_file", help="Path to save the output CSV file (optional)")
    
    args = parser.parse_args()
    
    reorder_csv_columns(args.input_file, args.output_file)