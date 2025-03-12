#!/bin/bash

# Function to generate Mosaik datasets (noisy and denoised versions)
generate_mosaik() {
  local name=$1
  local voltage=$2
  local current=$3
  local frequency=$4
  local samples=$5
  
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise-type uniform --noise-scale 0.05 --processed-output "${name}Mosaik+uniform.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise-type poisson --noise-scale 0.05 --processed-output "${name}Mosaik+poisson.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise-type laplace --noise-scale 0.02 --processed-output "${name}Mosaik+laplace.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise-type brownian --noise-scale 0.03 --processed-output "${name}Mosaik+brownian.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise-type pink --noise-scale 0.03 --processed-output "${name}Mosaik+pink.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise-type none --processed-output "${name}Mosaik.csv"
  
}

# Make sure we're starting fresh
echo "🚀 Starting Mosaik datasets generation..."
echo ""

generate_mosaik "5" 230 18 50 800

generate_mosaik "6" 480 50 60 800



echo "🎉 All Mosaik datasets generated successfully!"
echo "Generated files:"
ls -la *Mosaik*.csv