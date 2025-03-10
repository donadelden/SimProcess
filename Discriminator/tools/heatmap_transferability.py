#!/usr/bin/env python3
"""
Classifier Results Heatmap Visualization Script

This script:
1. Reads the analysis_results.csv file from the rf_output directory
2. Groups results by model configuration and file type (Morris vs GAN)
3. Calculates the average percentage of windows classified as real for each group
4. Creates a heatmap visualization of the results and saves it as PDF
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

def extract_feature_type(model_name):
    """Extract the feature type (C1, C2, etc.) from the model name."""
    match = re.search(r'rf_(.+?)_window', model_name)
    if match:
        return match.group(1)
    return model_name

def analyze_results(csv_file):
    """
    Analyze the results from the analysis_results.csv file.
    
    Args:
        csv_file (str): Path to the CSV file with analysis results
        
    Returns:
        pandas.DataFrame: Processed results for visualization
    """
    # Read the CSV file
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Results file not found: {csv_file}")
        
    df = pd.read_csv(csv_file)
    
    # Print some debugging information
    print(f"Total rows in CSV: {len(df)}")
    
    # Add file type column (morris or gan or other)
    df['file_type'] = df['analyzed_file'].apply(
        lambda x: 'gan' if 'gan' in x.lower() 
               else ('morris' if 'morris' in x.lower() else 'other')
    )
    
    # Print file type distribution
    file_type_counts = df['file_type'].value_counts()
    print(f"File type distribution:\n{file_type_counts}")
    
    # Extract feature type from model_used for better labels
    df['feature_type'] = df['model_used'].apply(extract_feature_type)
    
    # Group by model and file type, then calculate average real_windows_ratio
    grouped = df.groupby(['feature_type', 'file_type']).agg({
        'real_windows_ratio': 'mean',
        'analyzed_file': 'count'
    }).reset_index()
    
    # Rename columns for clarity
    grouped = grouped.rename(columns={
        'real_windows_ratio': 'avg_real_ratio',
        'analyzed_file': 'file_count'
    })
    
    # Convert ratio to percentage for better visualization
    grouped['avg_real_percentage'] = grouped['avg_real_ratio'] * 100
    
    return grouped

def create_heatmap(data, output_file=None):
    """
    Create a heatmap visualization.
    
    Args:
        data (pandas.DataFrame): Processed results data
        output_file (str, optional): Path to save the heatmap image
    """
    # Filter out 'other' file type and keep only 'gan' and 'morris'
    filtered_data = data[data['file_type'].isin(['gan', 'morris'])]
    
    # Print information about filtered data
    print(f"\nData after filtering (gan/morris only):")
    print(f"  Rows: {len(filtered_data)}")
    print(f"  Feature types: {sorted(filtered_data['feature_type'].unique())}")
    
    # Pivot data for the heatmap
    heatmap_data = filtered_data.pivot(index='feature_type', columns='file_type', values='avg_real_percentage')
    
    # Print the pivoted data
    print("\nData for heatmap:")
    print(heatmap_data)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Generate heatmap - customize to your liking
    sns.heatmap(
        heatmap_data, 
        annot=True,           # Show values in cells
        fmt='.1f',            # Format for the annotations
        cmap='YlGnBu',        # Blue color map
        vmin=0,               # Minimum value for color scale
        vmax=100,             # Maximum value for color scale
        cbar_kws={'label': '% Windows Classified as Real'},
        linewidths=0.5,       # Add lines between cells
        annot_kws={"size": 14}  # Increased font size for annotations
    )
    
    # Customize the heatmap
    plt.title('Percentage of Windows Classified as Real by Model and File Type', fontsize=16)
    plt.xlabel('File Type', fontsize=14)
    plt.ylabel('Model Feature Type', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the heatmap if output file is specified
    if output_file:
        plt.savefig(output_file, format='pdf', bbox_inches='tight')
        print(f"Heatmap saved to {output_file}")
    
    # Display the heatmap
    plt.show()

def main():
    """Main function to run the analysis and visualization."""
    # Define input and output paths
    results_file = "rf_output/analysis_results.csv"  # Look for file in current directory
    heatmap_output = "classification_heatmap.pdf"  # Changed to PDF
    
    try:
        # Analyze the results
        print(f"Analyzing results from {results_file}...")
        results_data = analyze_results(results_file)
        
        # Create heatmap visualization
        print("\nCreating heatmap visualization...")
        create_heatmap(results_data, heatmap_output)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())