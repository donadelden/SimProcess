# SimProcess: High Fidelity Simulation of Noisy ICS Physical Processes

SimProcess is a Python framework for analyzing time series data, extracting features, and classifying signals as real or simulated.

## Features

- **Feature Extraction**: Extract statistical features from time series data
- **Noise Analysis**: Extract noise features from signals using various filtering methods
- **Model Training**: Train SVM models to classify signals
- **Model Evaluation**: Evaluate model performance with detailed metrics
- **Model Transfer**: Apply trained models to new datasets (transferability)
- **Visualization**: Generate plots for feature importance, prediction distribution, and model performance
- **Command-line Interface**: Unified CLI for all operations

## Installation

### Requirements

- Python 3.7+
- Required packages: scikit-learn, pandas, numpy, matplotlib, tsfresh, scipy, joblib

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/donadelden/SimProcess.git
cd simprocess
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

SimProcess provides a unified command-line interface that supports multiple operations.

### Basic Usage

```bash
python main.py [command] [options]
```

### Available Commands

- `extract`: Extract features from CSV files
- `train`: Train a classification model
- `evaluate`: Evaluate a trained model
- `analyze`: Analyze new data using a trained model

## Detailed Command Instructions

### 1. Extract Features from CSV Files

The extract command processes raw signal data to extract features for model training.

#### Basic Command Structure:

```bash
python main.py extract --data-dir <DATA_DIRECTORY> --column <COLUMN_NAME> --output <OUTPUT_FILE>
```

#### Required Arguments:

- `--data-dir` or `-d`: Directory containing CSV files with raw signal data

#### Optional Arguments:

- `--column` or `-c`: Column to extract features from (if not specified, all columns will be used)
- `--output` or `-o`: Output CSV file path (default: combined\_<COLUMN>\_features.csv)
- `--window` or `-w`: Window size for feature extraction (default: 20)
- `--rename` or `-r`: Rename the column prefix in output features
- `--no-noise`: Disable noise feature extraction
- `--filter`: Filter type ('moving_average', 'butterworth', 'savgol', 'kalman') (default: kalman)
- `--cutoff`: Cutoff frequency for Butterworth filter (default: 0.1)
- `--fs`: Sampling frequency for Butterworth filter (default: 1.0)
- `--poly-order`: Polynomial order for Savitzky-Golay filter (default: 2)
- `--process-variance`: Process variance parameter for Kalman filter (default: 1e-5)
- `--measurement-variance`: Measurement variance parameter for Kalman filter (default: 1e-1)
- `--epsilon`: Epsilon value for filtering outliers (default: 0.10)
- `--all-columns`: Extract features from all suitable columns in the dataset

#### Example Commands:

Extract features from a specific column:

```bash
python main.py extract --data-dir data/raw_signals --column V1
```

Extract features from all suitable columns:

```bash
python main.py extract --data-dir data/raw_signals --all-columns --output combined_all_features.csv
```

Use Butterworth filter for noise extraction:

```bash
python main.py extract --data-dir data/raw_signals --column V1 --filter butterworth --cutoff 0.05 --fs 2.0
```

**Important Note:** Always specify the full path when using relative directories:

```bash
python main.py extract --data-dir ./data/raw_signals --column V1
```

### 2. Train a Classification Model

The train command creates a model that can classify signals as real or simulated using extracted features.

#### Basic Command Structure:

```bash
python main.py train --input <FEATURES_FILE> --model <OUTPUT_MODEL> [options]
```

#### Required Arguments:

- `--input` or `-i`: Input features CSV file for training

#### Optional Arguments:

- `--model` or `-m`: Path to save the trained model (default: simprocess_model.joblib)
- `--report` or `-r`: Path to save evaluation report (default: evaluation_report.csv)
- `--training-mode`: Type of training approach ('advanced', 'feature-reduction') (default: advanced)
- `--fast-mode`: Training speed mode (0=full, 1=reduced, 2=minimal) (default: 1)
- `--balancing-ratio`: Class balancing ratio for SMOTE (default: 0.9)
- `--features-to-keep`: Number of top features to keep (default: 11)
- `--max-features`: Maximum number of features to use in feature reduction mode (default: 20)
- `--dynamic`: Test on dynamic data
- `--no-eval`: Skip saving evaluation report

#### Example Commands:

Basic training with default parameters:

```bash
python main.py train --input ./combined_V1_features.csv
```

Save model to a specific location:

```bash
python main.py train --input ./combined_V1_features.csv --model ./models/V1_model.joblib
```

Train using feature reduction approach:

```bash
python main.py train --input ./combined_V1_features.csv --training-mode feature-reduction --max-features 15
```

Train with more extensive grid search (slower but potentially more accurate):

```bash
python main.py train --input ./combined_V1_features.csv --fast-mode 0
```

**Important Note:** Always include `./` when specifying files in the current directory to prevent path-related errors:

```bash
python main.py train --input ./combined_V1_features.csv
```

## Example Complete Workflow

### Step 1: Extract features from raw data

```bash
python main.py extract --data-dir ./data/signals --column V1 --window 20 --filter kalman
```

This will create a file named `combined_V1_features.csv` in the current directory.

### Step 2: Train a model using the extracted features

```bash
python main.py train --input ./combined_V1_features.csv --model ./models/V1_model.joblib
```

### Step 3: Evaluate the model on test data

```bash
python main.py evaluate --model ./models/V1_model.joblib --input ./test_features.csv --output-dir ./evaluation
```

### Step 4: Analyze new signals with the trained model

```bash
python main.py analyze --model ./models/V1_model.joblib --input ./new_signal.csv --column V1
```

## Additional Command Information

For detailed information on evaluate and analyze commands, refer to the remaining sections of the documentation.

## Programmatic Usage

You can also use SimProcess as a library in your Python code:

```python
from simprocess.features import process_csv_files
from simprocess.model import train_with_features, analyze_with_model

# Extract features
process_csv_files(
    data_directory='./data/signals',
    output_file='features.csv',
    target_column='V1',
    window_size=10
)

# Train a model
train_with_features(
    features_file='./features.csv',
    model_path='./model.joblib',
    train_ratio=0.8
)

# Analyze new data
is_real, confidence, metrics = analyze_with_model(
    model_path='./model.joblib',
    input_file='./new_signal.csv',
    target_column='V1',
    output_dir='./analysis_results',
    extract_noise=True,
    filter_type='savgol'
)

print(f"Classification: {'REAL' if is_real else 'SIMULATED'} with {confidence:.1f}% confidence")
print(f"Windows classified as real: {metrics['real_windows']}/{metrics['total_windows']} ({metrics['real_windows']/metrics['total_windows']*100:.1f}%)")
```

## License

MIT License
