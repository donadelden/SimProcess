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
  mv Panda.csv "${name}Panda+uniform.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency --noise-type poisson --noise-scale 0.05
  mv Panda.csv "${name}Panda+poisson.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency --noise-type laplace --noise-scale 0.05
  mv Panda.csv "${name}Panda+laplace.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency --noise-type laplace --noise-scale 0.02
  mv Panda.csv "${name}Panda+laplace.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency --noise-type brownian --noise-scale 0.03
  mv Panda.csv "${name}Panda+brownian.csv"

  python3 main.py $samples --voltage $voltage --current $current --frequency $frequency --noise-type pink --noise-scale 0.03
  mv Panda.csv "${name}Panda+pink.csv"
  mv Panda_denoised.csv "${name}Panda.csv"
  
  echo ""
}

cho "🚀 Starting Panda datasets generation..."
echo ""

generate_panda "5" 230 18 50 800

generate_panda "6" 480 50 60 800

echo "🎉 All datasets generated successfully!"
echo "Generated files:"
ls -la *Panda*.csv