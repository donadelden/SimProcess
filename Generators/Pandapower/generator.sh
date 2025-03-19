#!/bin/bash

# Function to generate datasets with the enhanced pandapower implementation
generate_datasets() {
  local name=$1
  local voltage=$2
  local current=$3
  local frequency=$4
  local samples=$5
  
  # Single noise types (for backward compatibility and comparison)
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "uniform:scale=0.01" --output-noisy "${name}Panda+uniform.csv" --output-noiseless "${name}Panda_denoised.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "gaussian:scale=0.01" --output-noisy "${name}Panda+gaussian1.csv" --output-noiseless "${name}Panda_denoised.csv"
  
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "gaussian:scale=0.05" --output-noisy "${name}Panda+gaussian2.csv" --output-noiseless "${name}Panda_denoised.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "poisson:scale=0.01,poisson_lambda=1.5" --output-noisy "${name}Panda+poisson.csv" --output-noiseless "${name}Panda_denoised.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "laplace:scale=0.01" --output-noisy "${name}Panda+laplace.csv" --output-noiseless "${name}Panda_denoised.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "pink:scale=0.01" --output-noisy "${name}Panda+pink.csv" --output-noiseless "${name}Panda_denoised.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "none" --output-noisy "${name}Panda_clean.csv" --output-noiseless "${name}Panda_denoised.csv"
  
  # Gaussian Mixture Model noise
  # Note: You may need to change the reference_file path to match your environment
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "gmm:scale=0.02,n_components=3" --reference-file "../../Discriminator/dataset/EPIC6.csv" \
    --output-noisy "${name}Panda+gmm.csv" --output-noiseless "${name}Panda_denoised.csv"
  
  # Multiple noise combinations
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "gaussian:scale=0.01" --noise "uniform:scale=0.01" \
    --output-noisy "${name}Panda+gaussian+uniform.csv" --output-noiseless "${name}Panda_denoised.csv"
    
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "laplace:scale=0.01" --noise "uniform:scale=0.01" \
    --output-noisy "${name}Panda+laplace+uniform.csv" --output-noiseless "${name}Panda_denoised.csv"
    
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "laplace:scale=0.01" --noise "poisson:scale=0.01,poisson_lambda=1.5" \
    --output-noisy "${name}Panda+laplace+poisson.csv" --output-noiseless "${name}Panda_denoised.csv"

}

# Make sure we're starting fresh
echo "🚀 Starting Pandapower dataset generation with single and multiple noise types..."
echo ""

# Generate datasets with default parameters (voltage=245, current=20, frequency=50, samples=3000)
generate_datasets "" 245 20 50 3000

echo "🎉 All Pandapower datasets generated successfully!"
echo "Generated files:"
ls -la *Panda*.csv