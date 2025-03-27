# Mosaik Electric Power System Simulator

This package provides tools for generating synthetic electric power system data and visualizing the results.

## Components

- **main.py**: Power system data generator based on the Mosaik framework
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
```

### Visualizing Generated Data

```bash
# Create basic plots of the generated data
python plotter.py Mosaik.csv

# Save plots to a specific directory
python plotter.py Mosaik.csv --output plots

# Analyze noise characteristics
python plotter.py Mosaik.csv --analyze

# Generate high-resolution plots and save without displaying
python plotter.py Mosaik.csv --output plots --dpi 600 --no-show
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

## Current Pattern

The generator creates a predetermined load pattern with five segments:

- First 20% of duration: Base current
- Next 25% of duration: Base current +20%
- Next 20% of duration: Base current
- Next 20% of duration: Base current -15%
- Final 15% of duration: Base current

## Requirements

- Python 3.7 or higher
- mosaik
- mosaik-csv-writer
- numpy
- pandas
- scikit-learn
- matplotlib
- scipy

## Example

```bash
# Complete simulation workflow
python main.py 7200 --voltage 230 --current 25 --noise "gaussian:scale=0.015" --noise "impulse:impulse_prob=0.001" --processed-output power_data.csv
python plotter.py power_data.csv --output visualization
```
