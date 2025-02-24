#!/bin/bash

echo "Starting batch analysis at $(date)"
echo "----------------------------------------"

for file in data/*.csv; do
    echo "Analyzing $file..."
    echo "----------------------------------------"
    python3 galileo.py analyze -i "$file"
    echo "----------------------------------------"
    echo
done