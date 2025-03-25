#!/bin/bash
# process_all_columns_parallel.sh
# This script extracts features for each column and runs analysis on them in parallel with different filters

# Make sure GNU Parallel is installed
if ! command -v parallel &>/dev/null; then
    echo "GNU Parallel is not installed. Please install it first:"
    echo "  sudo apt-get install parallel   # On Debian/Ubuntu"
    echo "  sudo yum install parallel       # On CentOS/RHEL"
    exit 1
fi

# Define parameters
COLUMNS=("C1" "C2" "C3" "V1" "V2" "V3" "frequency" "power_real" "power_reactive" "power_apparent")
WINDOWS=(5 10 15 20 30 40 50 60 70 80 90 100)
FILTERS=("kalman")
OUTPUT_DIR="dataset_features"

# Create main output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Ensuring main output directory exists: $OUTPUT_DIR"

# Create a wrapper script for running the extraction with filters
cat > run_extraction_filter.sh << 'EOF'
#!/bin/bash
COLUMN="$1"
WINDOW="$2"
FILTER="$3"
OUTPUT_DIR="$4/window$WINDOW/$FILTER"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the extraction command
echo "Running extraction for column $COLUMN, window $WINDOW, filter $FILTER"
python3 main.py extract -d dataset/ -c "$COLUMN" -w "$WINDOW" --output "$OUTPUT_DIR/combined_${COLUMN}_features.csv" --filter "$FILTER"
EOF

# Create a wrapper script for running the extraction with no noise
cat > run_extraction_no_noise.sh << 'EOF'
#!/bin/bash
COLUMN="$1"
WINDOW="$2"
OUTPUT_DIR="$3/window$WINDOW/no_noise"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the extraction command
echo "Running extraction for column $COLUMN, window $WINDOW, no noise"
python3 main.py extract -d dataset/ -c "$COLUMN" -w "$WINDOW" --no-noise --output "$OUTPUT_DIR/combined_${COLUMN}_features.csv"
EOF

# Make the wrapper scripts executable
chmod +x run_extraction_filter.sh
chmod +x run_extraction_no_noise.sh

echo "Starting parallel feature extraction with multiple filters..."
echo "=========================================="

# Run the extractions with filters in parallel
for FILTER in "${FILTERS[@]}"; do
    echo "Processing filter: $FILTER"
    parallel --progress --bar "./run_extraction_filter.sh {1} {2} $FILTER $OUTPUT_DIR" ::: "${COLUMNS[@]}" ::: "${WINDOWS[@]}"
    
    if [ $? -ne 0 ]; then
        echo "Some $FILTER filter extraction jobs failed. Check the output above for details."
    else
        echo "All $FILTER filter extraction jobs completed successfully."
    fi
    echo "----------------------------------------"
done

echo "Starting parallel feature extraction with no noise..."
echo "=========================================="

# Run the extractions with no noise in parallel
#parallel --progress --bar "./run_extraction_no_noise.sh {1} {2} $OUTPUT_DIR" ::: "${COLUMNS[@]}" ::: "${WINDOWS[@]}"

if [ $? -ne 0 ]; then
    echo "Some no-noise extraction jobs failed. Check the output above for details."
else
    echo "All no-noise extraction jobs completed successfully."
fi

# Clean up the wrapper scripts
rm run_extraction_filter.sh
rm run_extraction_no_noise.sh

echo "All extractions completed."
echo "=========================================="

echo "Feature extraction complete. Analysis step commented out."
echo "All processing complete."