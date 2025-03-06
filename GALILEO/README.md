# Galileo: Signal Analysis and Classification Framework

Galileo is a Python framework for analyzing time series data, extracting features, and classifying signals as real or simulated.

## Features

- **Feature Extraction**: Extract statistical features from time series data
- **Noise Analysis**: Extract noise features from signals
- **Model Training**: Train SVM models to classify signals
- **Model Evaluation**: Evaluate model performance with detailed metrics
- **Visualization**: Generate plots for feature importance and model performance
- **Command-line Interface**: Unified CLI for all operations

## Installation

### Requirements

- Python 3.7+
- Required packages: scikit-learn, pandas, numpy, matplotlib, tsfresh, scipy, joblib

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/username/galileo.git
cd galileo
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Install the package in development mode:

```bash
pip install -e .
```

## Usage

Galileo provides a unified command-line interface that supports multiple operations.

### Basic Usage

```bash
python main.py [command] [options]
```

### Available Commands

- `extract`: Extract features from CSV files
- `train`: Train a classification model
- `evaluate`: Evaluate a trained model
- `analyze`: Analyze new data using a trained model

### Example Workflows

#### 1. Extract Features from CSV Files

```bash
python main.py extract --data-dir data/raw_signals --column V1 --output features_V1.csv
```

This command:

- Processes all CSV files in the `data/raw_signals` directory
- Extracts features from the `V1` column
- Extracts noise features by default
- Saves the combined features to `features_V1.csv`

Options:

- `--window`: Window size for feature extraction (default: 10)
- `--no-noise`: Disable noise feature extraction
- `--filter`: Filter type for noise extraction (default: savgol)
- More options available (run with `--help` to see all)

#### 2. Train a Model

```bash
python main.py train --input features_V1.csv --model model_V1.joblib
```

This command:

- Loads the features from `features_V1.csv`
- Splits the data into training and testing sets
- Trains an SVM model
- Calculates and plots feature importance
- Evaluates the model on the test set
- Saves the trained model to `model_V1.joblib`

Options:

- `--train-ratio`: Ratio of data to use for training (default: 0.8)
- `--seed`: Random seed for reproducibility (default: 42)
- `--no-eval`: Skip model evaluation
- `--report`: Path to save evaluation report (default: evaluation_report.csv)

#### 3. Evaluate a Model

```bash
python main.py evaluate --model model_V1.joblib --input features_test.csv --output-dir eval_results
```

This command:

- Loads the model from `model_V1.joblib`
- Evaluates it on the features in `features_test.csv`
- Generates evaluation visualizations and metrics
- Saves results to the `eval_results` directory

Options:

- `--test-only`: Use the entire input file as test set instead of splitting
- `--report`: Path to save evaluation report (default: evaluation_report.csv)
- `--column`: Name of the column being evaluated (for reporting)

#### 4. Analyze New Data

```bash
python main.py analyze --model model_V1.joblib --input new_signal.csv --output results.json
```

This command:

- Loads the model from `model_V1.joblib`
- Analyzes the signal in `new_signal.csv`
- Classifies it as REAL or SIMULATED
- Saves the analysis results to `results.json`

Options:

- `--column`: Specific column to analyze (overrides model settings)
- `--verbose`: Print detailed analysis information

## Programmatic Usage

You can also use Galileo as a library in your Python code:

```python
from galileo.features import process_csv_files
from galileo.model import train_with_features, analyze_with_model

# Extract features
process_csv_files(
    data_directory='data/signals',
    output_file='features.csv',
    target_column='V1',
    window_size=10
)

# Train a model
train_with_features(
    features_file='features.csv',
    model_path='model.joblib',
    train_ratio=0.8
)

# Analyze new data
results = analyze_with_model(
    file_path='new_signal.csv',
    model_path='model.joblib'
)
print(f"Classification: {results['classification']} with {results['confidence']}% confidence")
```

## License

MIT License
