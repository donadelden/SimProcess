#!/bin/bash

# Function to generate Mosaik datasets (noisy and denoised versions)
generate_mosaik() {
  local name=$1
  local voltage=$2
  local current=$3
  local frequency=$4
  local samples=$5
  
  echo "Generating ${name}Mosaik dataset (with noise)..."
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise-type uniform --noise-scale 0.05 --processed-output "${name}Mosaik.csv"
  
  echo "Generating ${name}Mosaik_denoised dataset (without noise)..."
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise-type none --processed-output "${name}Mosaik_denoised.csv"
  
  echo "✅ Generated ${name}Mosaik.csv and ${name}Mosaik_denoised.csv"
  echo ""
}

# Make sure we're starting fresh
echo "🚀 Starting Mosaik datasets generation..."
echo ""

# Dataset 1: Voltage=245V, Current=20A, Frequency=50Hz, Samples=1200
generate_mosaik "1" 245 20 50 1200

# Dataset 2: Voltage=245V, Current=20A, Frequency=50Hz, Samples=3000
generate_mosaik "2" 245 20 50 3000

# Dataset 3: Voltage=400V, Current=10A, Frequency=50Hz, Samples=500
generate_mosaik "3" 400 10 50 500

# Dataset 4: Voltage=350V, Current=10A, Frequency=55Hz, Samples=300
generate_mosaik "4" 350 10 55 300

echo "🎉 All Mosaik datasets generated successfully!"
echo "Generated files:"
ls -la *Mosaik*.csv