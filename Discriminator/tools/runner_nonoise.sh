#!/bin/bash
# process_all_columns.sh
# This script extracts features for each column and runs analysis on them

# Define an array of columns to process
COLUMNS=("C1")
WIDTHS=("5" "10,15" "20" "30" "40" "50" "60" "70" "80" "90" "100")
FILTERS=(no-noise)

# Process each column
for COLUMN in "${COLUMNS[@]}"; do
for WIDTH in "${WIDTHS[@]}"; do
for FILTER in "${FILTERS[@]}"; do

    echo "=========================================="
    echo "Processing column: $COLUMN, width: $WIDTH, filter: $FILTER"
    echo "=========================================="
    
    # Step 1: Extract features for this column
    echo "Extracting features..."
    python3 main.py extract -d dataset/ -c "$COLUMN" -w "$WIDTH" --output dataset_features/window${WIDTH}/no_noise/combined_${COLUMN}_features.csv --"$FILTER"

    # Check if extraction was successful
    if [ $? -ne 0 ]; then
        echo "Error: Feature extraction failed for column $COLUMN"
        continue
    fi
    
    # Step 2: Run main analysis on the extracted features
    echo "Running analysis..."
    #python3 main.py train -i "combined_${COLUMN}_features.csv" --noise-only
    
    # Check if analysis was successful
    if [ $? -ne 0 ]; then
        echo "Error: Analysis failed for column $COLUMN"
    fi
    
    echo "Finished processing $COLUMN"
    echo ""
done
done
done

echo "All columns processed."