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
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 60 --filter butterworth --output dataset_features/window60/butterworth/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 70 --filter butterworth --output dataset_features/window70/butterworth/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 80 --filter butterworth --output dataset_features/window80/butterworth/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 90 --filter butterworth --output dataset_features/window90/butterworth/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 100 --filter butterworth --output dataset_features/window100/butterworth/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 60 --filter savgol --output dataset_features/window60/savgol/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 70 --filter savgol --output dataset_features/window70/savgol/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 80 --filter savgol --output dataset_features/window80/savgol/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 90 --filter savgol --output dataset_features/window90/savgol/combined_${COLUMN}_features.csv 
    python3 main.py extract -d dataset/data/ -c "$COLUMN" -w 100 --filter savgol --output dataset_features/window100/savgol/combined_${COLUMN}_features.csv 


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