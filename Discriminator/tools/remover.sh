#!/bin/bash

# Script to recursively remove all CSV files from dataset_features/ folder

# Define the target directory
TARGET_DIR="dataset_features"

# Check if the directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

# Count CSV files before removal
CSV_COUNT=$(find "$TARGET_DIR" -type f -name "*.csv" | wc -l)
echo "Found $CSV_COUNT CSV files in $TARGET_DIR"

# Ask for confirmation
read -p "Do you want to remove all these CSV files? (y/n): " CONFIRM

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Find and remove all CSV files
find "$TARGET_DIR" -type f -name "*.csv" -print -delete

echo "All CSV files have been removed from $TARGET_DIR"