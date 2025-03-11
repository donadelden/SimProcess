#!/bin/bash

# Function to generate and rename Panda files
generate_panda() {
  local name=$1
  local voltage=$2
  local current=$3
  local frequency=$4
  local samples=$5
  
  echo "Generating ${name}Panda dataset..."
  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency --noise-type uniform --noise-scale 0.05
  
  # Rename the output files
  mv Panda.csv "${name}Panda.csv"
  mv Panda_denoised.csv "${name}Panda_denoised.csv"
  
  echo "✅ Generated ${name}Panda.csv and ${name}Panda_denoised.csv"
  echo ""
}

# Make sure we're starting fresh
echo "🚀 Starting Panda datasets generation..."
echo ""

# Dataset 1: Voltage=245V, Current=20A, Frequency=50Hz, Samples=1200
generate_panda "1" 245 20 50 1200

# Dataset 2: Voltage=245V, Current=20A, Frequency=50Hz, Samples=3000
generate_panda "2" 245 20 50 3000

# Dataset 3: Voltage=445V, Current=15A, Frequency=55Hz, Samples=500
generate_panda "3" 445 15 55 500

# Dataset 4: Voltage=300V, Current=10A, Frequency=50Hz, Samples=400
generate_panda "4" 300 10 50 400

echo "🎉 All datasets generated successfully!"
echo "Generated files:"
ls -la *Panda*.csv