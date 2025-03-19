#!/bin/bash
# process_all_filters_windows_parallel.sh
# This script runs feature extraction for all filters and all window sizes in parallel

# Make sure GNU Parallel is installed
if ! command -v parallel &>/dev/null; then
    echo "GNU Parallel is not installed. Please install it first:"
    echo "  sudo apt-get install parallel   # On Debian/Ubuntu"
    echo "  sudo yum install parallel       # On CentOS/RHEL"
    exit 1
fi

# Define parameters
WINDOWS=(5 10 15 20 30 40 50 60 70 80 90 100)
EPSILON=0.3
FILTERS=("kalman")
OUTPUT_DIR="dataset_features"

# Create a wrapper script for running the extraction with filters
cat > run_extraction_filter.sh << 'EOF'
#!/bin/bash
FILTER="$1"
WINDOW_SIZE="$2"
EPSILON="$3"
OUTPUT_DIR="$4"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}/window${WINDOW_SIZE}/${FILTER}"

# Run the extraction command
echo "Running extraction for filter $FILTER, window $WINDOW_SIZE"
python3 main.py extract -d dataset --all-columns --epsilon $EPSILON \
  --output ${OUTPUT_DIR}/window${WINDOW_SIZE}/${FILTER}/combined_allcolumns_features.csv \
  --filter $FILTER -w $WINDOW_SIZE
EOF

# Create a wrapper script for running the extraction with no noise
cat > run_extraction_no_noise.sh << 'EOF'
#!/bin/bash
WINDOW_SIZE="$1"
EPSILON="$2"
OUTPUT_DIR="$3"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}/window${WINDOW_SIZE}/no_noise"

# Run the extraction command
echo "Running extraction with window $WINDOW_SIZE, no noise"
python3 main.py extract -d dataset --all-columns --epsilon $EPSILON \
  --output ${OUTPUT_DIR}/window${WINDOW_SIZE}/no_noise/combined_allcolumns_features.csv \
  --no-noise -w $WINDOW_SIZE
EOF

# Make the wrapper scripts executable
chmod +x run_extraction_filter.sh
chmod +x run_extraction_no_noise.sh

echo "Starting parallel feature extraction with multiple filters and windows..."
echo "=========================================="

# Run the extractions with filters in parallel
for FILTER in "${FILTERS[@]}"; do
    echo "Processing filter: $FILTER"
    parallel --progress --bar "./run_extraction_filter.sh $FILTER {1} $EPSILON $OUTPUT_DIR" ::: "${WINDOWS[@]}"
    
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
#parallel --progress --bar "./run_extraction_no_noise.sh {1} $EPSILON $OUTPUT_DIR" ::: "${WINDOWS[@]}"

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
echo "All processing complete."