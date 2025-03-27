# PandaPower Electric Grid Simulator

This package provides tools for generating synthetic electric power system data using PandaPower and visualizing the results.

## Components

- **main.py**: Power system data generator based on the PandaPower framework
- **plotter.py**: Visualization tool for analyzing the generated data

## Quick Start

### Generating Synthetic Power Data

```bash
# Generate 1 hour of data with default parameters
python main.py 3600

# Generate data with specific noise profile
python main.py 3600 --noise "gaussian:scale=0.02"

# Generate data with multiple layered noise types
python main.py 3600 --noise "gaussian:scale=0.01" --noise "impulse:impulse_prob=0.005"

# Generate data with custom voltage, current, and frequency
python main.py 3600 --voltage 240 --current 15 --frequency 60

# Specify output filenames for noisy and noiseless data
python main.py 3600 --output-noisy "data_noisy.csv" --output-noiseless "data_clean.csv"
```

### Visualizing Generated Data

```bash
# Create basic plots of the generated data
python plotter.py Panda.csv

# Save plots to a specific directory
python plotter.py Panda.csv --output plots

# Analyze noise characteristics
python plotter.py Panda.csv --analyze

# Generate high-resolution plots and save without displaying
python plotter.py Panda.csv --output plots --dpi 600 --no-show
```

## Batch Processing Using Shell Script

You can use a shell script to generate multiple datasets with different noise configurations:

```bash
#!/bin/bash

# Function to generate datasets with various noise profiles
generate_datasets() {
  local name=$1
  local voltage=$2
  local current=$3
  local frequency=$4
  local samples=$5

  # Generate data with gaussian noise
  python main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "gaussian:scale=0.01" --output-noisy "${name}Panda+gaussian1.csv"

  # Generate data with uniform noise
  python main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "uniform:scale=0.01" --output-noisy "${name}Panda+uniform.csv"

  # Generate data with combined noise types
  python main.py $samples --voltage $voltage --current $current --frequency $frequency \
    --noise "gaussian:scale=0.01" --noise "uniform:scale=0.01" \
    --output-noisy "${name}Panda+gaussian+uniform.csv"
}

# Generate datasets
generate_datasets "dynamic_" 245 20 50 3600
```

## Noise Types

The generator supports various noise models that can be applied to the power system signals:

- `gaussian`: Normal distribution noise (default)
- `uniform`: Uniformly distributed noise
- `laplace`: Laplace distribution noise
- `poisson`: Poisson distribution noise
- `impulse`: Random impulse/spike noise
- `brownian`: Brownian motion noise
- `pink`: 1/f (pink) noise spectrum
- `gmm`: Gaussian Mixture Model noise
- `none`: No noise applied

## Noise Parameters

Each noise type accepts the following parameters:

- `scale`: Scale factor for noise amplitude (default: 0.01)
- `impulse_prob`: Probability of impulse occurring (for impulse noise)
- `poisson_lambda`: Lambda parameter for Poisson noise
- For GMM noise: `n_components`, `means`, `variances`, `weights`

## GMM Reference Data

For Gaussian Mixture Model noise, you can use a reference file:

```bash
python main.py 3600 --noise "gmm:scale=0.02" --reference-file "reference_data.csv"
```

Or manually specify GMM parameters:

```bash
python main.py 3600 --noise "gmm:scale=0.02,n_components=3" \
  --gmm-means "0.0,0.05,-0.05" --gmm-variances "0.01,0.05,0.02" --gmm-weights "0.6,0.3,0.1"
```

## Current Pattern

The generator creates a predetermined load pattern with five segments:

- First 20% of duration: Base current
- Next 25% of duration: Base current +20%
- Next 20% of duration: Base current
- Next 20% of duration: Base current -15%
- Final 15% of duration: Base current

## Output Files

The simulator produces two CSV files:

- **Noisy data file**: Contains measurements with the configured noise applied
- **Noiseless data file**: Contains clean measurements for reference

## Requirements

- Python 3.7 or higher
- pandapower
- numpy
- pandas
- scikit-learn
- matplotlib
- scipy

## Example

```bash
# Complete simulation workflow
python main.py 7200 --voltage 230 --current 25 --noise "gaussian:scale=0.015" \
  --noise "impulse:impulse_prob=0.001" --output-noisy power_data.csv
python plotter.py power_data.csv --output visualization
```
