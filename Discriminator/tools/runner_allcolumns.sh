#!/bin/bash
# process_all_filters_parallel.sh
# This script runs feature extraction for all filters with epsilon 0.3 in parallel

# Make sure GNU Parallel is installed
if ! command -v parallel &>/dev/null; then
    echo "GNU Parallel is not installed. Please install it first:"
    echo "  sudo apt-get install parallel   # On Debian/Ubuntu"
    echo "  sudo yum install parallel       # On CentOS/RHEL"
    exit 1
fi

# Define parameters
WINDOW_SIZE=20
EPSILON=0.3
FILTERS=("moving_average" "kalman" "butterworth" "savgol")
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

# Return the status
exit $?
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

# Return the status
exit $?
EOF

# Make the wrapper scripts executable
chmod +x run_extraction_filter.sh
chmod +x run_extraction_no_noise.sh

echo "Starting parallel feature extraction with multiple filters..."
echo "=========================================="

# Run the extractions with filters in parallel
parallel --progress --bar "./run_extraction_filter.sh {1} $WINDOW_SIZE $EPSILON $OUTPUT_DIR" ::: "${FILTERS[@]}"

if [ $? -ne 0 ]; then
    echo "Some filter extraction jobs failed. Check the output above for details."
else
    echo "All filter extraction jobs completed successfully."
fi

echo "=========================================="
echo "Running final extraction with no-noise and window size 50"
echo "=========================================="

# Run the final no-noise extraction
./run_extraction_no_noise.sh $WINDOW_SIZE $EPSILON $OUTPUT_DIR

if [ $? -ne 0 ]; then
    echo "Final no-noise extraction job failed. Check the output above for details."
else
    echo "Final no-noise extraction job completed successfully."
fi

# Clean up the wrapper scripts
rm run_extraction_filter.sh
rm run_extraction_no_noise.sh

echo "All processing complete."