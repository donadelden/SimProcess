#!/bin/bash

# Function to generate Mosaik datasets (with various noise configurations)
generate_mosaik() {
  local name=$1
  local voltage=$2
  local current=$3
  local frequency=$4
  local samples=$5
  
  # Single noise types (for backward compatibility and comparison)
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "uniform:scale=0.01" --processed-output "${name}Mosaik+uniform.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "gaussian:scale=0.01" --processed-output "${name}Mosaik+gaussian1.csv"
  
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "gaussian:scale=0.05" --processed-output "${name}Mosaik+gaussian2.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "poisson:scale=0.01,poisson_lambda=1.5" --processed-output "${name}Mosaik+poisson.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "laplace:scale=0.01" --processed-output "${name}Mosaik+laplace.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "pink:scale=0.01" --processed-output "${name}Mosaik+pink.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "none" --processed-output "${name}Mosaik.csv"
  
  # Gaussian Mixture Model noise
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "gmm:scale=0.02,n_components=3,reference_file=../../Discriminator/dataset/EPIC6.csv" \
    --processed-output "${name}Mosaik+gmm.csv"
  
  # Multiple noise combinations
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "gaussian:scale=0.01" --noise "uniform:scale=0.01" \
    --processed-output "${name}Mosaik+gaussian+uniform.csv"
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "laplace:scale=0.01" --noise "uniform:scale=0.01" \
    --processed-output "${name}Mosaik+laplace+uniform.csv"
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "laplace:scale=0.01" --noise "poisson:scale=0.01,poisson_lambda:1.5" \
    --processed-output "${name}Mosaik+laplace+poisson.csv"

}

# Make sure we're starting fresh
echo "🚀 Starting Mosaik datasets generation with single and multiple noise types..."
echo ""

generate_mosaik "dynamic_" 245 20 50 3000

echo "🎉 All Mosaik datasets generated successfully!"
echo "Generated files:"
ls -la *Mosaik*.csv