import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_f1_scores(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize lists to store data for each block
    blocks = ['RF_nonoise', 'RF_withMA', 'RF_onlyMA', 'RF_withBW', 'RF_onlyBW', 'RF_withSG', 'RF_onlySG']
    block_data = {block: [] for block in blocks}
    current_block = 0
    
    # Process data row by row
    for i, row in df.iterrows():
        # Check for separator row (often has NaN values in important columns)
        if pd.isna(row['f1score']) and pd.isna(row['precision']):
            current_block += 1
            continue
        
        # Add f1score to the appropriate block if not a separator
        if current_block < len(blocks):
            block_data[blocks[current_block]].append({
                'column': row['column'],
                'f1score': row['f1score']
            })
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Variables for bar positioning
    bar_width = 0.15  # Made narrower to accommodate more bars
    index = np.arange(len(block_data[blocks[0]]))
    opacity = 0.8
    
    # Plot bars for each block
    for i, block in enumerate(blocks):
        columns = [item['column'] for item in block_data[block]]
        f1scores = [item['f1score'] for item in block_data[block]]
        position = index + (i * bar_width)
        
        plt.bar(position, f1scores, bar_width,
                alpha=opacity,
                color=plt.cm.tab10(i),
                label=block)
    
    # Add labels and title
    plt.xlabel('Columns')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores by Column and Model Type')
    plt.xticks(index + bar_width * 2, columns, rotation=45, ha='right')  # Adjusted for 5 bars
    plt.legend()
    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save and show the plot
    plt.savefig('f1_scores_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = "evaluation_report.csv"
    plot_f1_scores(csv_file)