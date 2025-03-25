# SimProcess: High Fidelity Simulation of Noisy ICS Physical Processes

This project provides tools for generating synthetic power grid data, processing real power grid measurements, and discriminating between real and simulated scenarios using machine learning.

## Components

### Generators

The project includes two power grid data generators based on industry-standard simulation frameworks:

#### Mosaik Generator

```
usage: main.py [-h] [--no-noise] [--processed-output PROCESSED_OUTPUT] [--voltage VOLTAGE] [--current CURRENT] [--frequency FREQUENCY] duration
```

#### Pandapower Generator

```
usage: main.py [-h] [--voltage VOLTAGE] [--current CURRENT] [--frequency FREQUENCY] duration
```

### EPIC Data Processing

The EPIC folder contains tools for handling real power grid data from the EPIC testbed:

- Preprocessor utility to standardize real data format
- Ensures compatibility with simulated data format for analysis

```
Usage: python3 preprocessor.py csv_file
```

### Discriminator

Tools for analyzing power grid data:

#### Analyzer

```
usage: analyzer.py [-h] [--window WINDOW] [--output-dir OUTPUT_DIR] csv_file
```

- Outputs visual analysis of CSV data files
- Supports sliding window analysis

#### GALILEO

```
usage: galileo.py [-h] --input INPUT [--real [REAL ...]] [--simulated [SIMULATED ...]] [--model MODEL] {train,analyze}
```

- SVM-based classification system
- Two main modes of operation:
  1. Training: Build model using labeled real and simulated data
  2. Analysis: Classify new data as real or simulated

## Workflow

1. Generate synthetic data using either Mosaik or Pandapower generators
2. Process real EPIC testbed data using the preprocessor
3. Train GALILEO using both real and simulated datasets
   ```
   example: galileo.py train --input . --real real_data.csv --simulated fake_data.csv
   ```
4. Use the trained model to analyze and classify new power grid scenarios
   ```
   example: galileo.py analyze --input some_data.csv
   ```
