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
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 5 --no-noise --output dataset_features/window5/no_noise/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 10 --no-noise --output dataset_features/window10/no_noise/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 15 --no-noise --output dataset_features/window15/no_noise/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 20 --no-noise --output dataset_features/window20/no_noise/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 30 --no-noise --output dataset_features/window30/no_noise/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 40 --no-noise --output dataset_features/window40/no_noise/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 50 --no-noise --output dataset_features/window50/no_noise/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 60 --no-noise --output dataset_features/window60/no_noise/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 70 --no-noise --output dataset_features/window70/no_noise/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 80 --no-noise --output dataset_features/window80/no_noise/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 90 --no-noise --output dataset_features/window90/no_noise/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 100 --no-noise --output dataset_features/window100/no_noise/combined_${COLUMN}_features.csv 

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

echo "All columns processed."