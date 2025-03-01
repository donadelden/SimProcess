#!/bin/bash
# process_all_columns.sh
# This script extracts features for each column and runs analysis on them

# Define an array of columns to process
COLUMNS=("C1" "C2" "C3" "V1" "V2" "V3" "frequency" "power_real" "power_effective" "power_apparent")

# Process each column
for COLUMN in "${COLUMNS[@]}"; do
    echo "=========================================="
    echo "Processing column: $COLUMN"
    echo "=========================================="
    
    # Step 1: Extract features for this column
    echo "Extracting features..."
    python3 extractor.py -c "$COLUMN" --no-noise
    
    # Check if extraction was successful
    if [ $? -ne 0 ]; then
        echo "Error: Feature extraction failed for column $COLUMN"
        continue
    fi
    
    # Step 2: Run main analysis on the extracted features
    echo "Running analysis..."
    python3 main.py -i "combined_${COLUMN}_features.csv"
    
    # Check if analysis was successful
    if [ $? -ne 0 ]; then
        echo "Error: Analysis failed for column $COLUMN"
    fi
    
    echo "Finished processing $COLUMN"
    echo ""
done

echo "All columns processed."