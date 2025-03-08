# Galileo: Signal Analysis and Classification Framework

Galileo is a Python framework for analyzing time series data, extracting features, and classifying signals as real or simulated.

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
python -m galileo [command] [options]
```

### Available Commands

- `extract`: Extract features from CSV files
- `train`: Train a classification model
- `evaluate`: Evaluate a trained model
- `analyze`: Analyze new data using a trained model

### Example Workflows

#### 1. Extract Features from CSV Files

```bash
python -m galileo extract --data-dir data/raw_signals --column V1 --output features_V1.csv
```

This command:

- Processes all CSV files in the `data/raw_signals` directory
- Extracts features from the `V1` column
- Extracts noise features by default
- Saves the combined features to `features_V1.csv`

Options:

- `--window`: Window size for feature extraction (default: 10)
- `--no-noise`: Disable noise feature extraction
- `--filter`: Filter type for noise extraction ('moving_average', 'butterworth', 'savgol', default: savgol)
- `--cutoff`: Cutoff frequency for Butterworth filter (default: 0.1)
- `--fs`: Sampling frequency for Butterworth filter (default: 1.0)
- `--poly-order`: Polynomial order for Savitzky-Golay filter (default: 2)
- `--rename`: Rename the column prefix in output features

#### 2. Train a Model

```bash
python -m galileo train --input features_V1.csv --model model_V1.joblib
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
- `--noise-only`: Use only noise features for training
- `--report`: Path to save evaluation report (default: evaluation_report.csv)

#### 3. Evaluate a Model

```bash
python -m galileo evaluate --model model_V1.joblib --input features_test.csv --output-dir eval_results
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
python -m galileo analyze --model model_V1.joblib --input new_signal.csv --column V1 --output-dir analysis_results
```

This command:

- Loads the model from `model_V1.joblib`
- Extracts features from the `V1` column in `new_signal.csv`
- Uses the same feature extraction method as during training, including noise extraction
- Classifies each window as REAL or SIMULATED
- Provides an overall classification for the signal with confidence level
- Generates visualizations of the prediction distribution
- Saves detailed results to the `analysis_results` directory

Options:

- `--column`: Specific column to analyze (required)
- `--rename`: Rename the column to match model expectations (e.g., if model was trained on V1 but input uses voltage1)
- `--window`: Window size for feature extraction (default: 10)
- `--output-dir`: Directory to save analysis results (default: analysis)
- `--no-noise`: Disable noise feature extraction
- `--filter`: Filter type for noise extraction ('moving_average', 'butterworth', 'savgol', default: savgol)
- `--cutoff`: Cutoff frequency for Butterworth filter (default: 0.1)
- `--fs`: Sampling frequency for Butterworth filter (default: 1.0)
- `--poly-order`: Polynomial order for Savitzky-Golay filter (default: 2)

## Analyze Command Output Files

The analyze command generates the following output files in the specified output directory:

1. `window_predictions.csv`: CSV file containing window-by-window predictions, including:

   - Window index
   - Classification probability
   - Binary prediction (0 = simulated, 1 = real)
   - Classification label (Real or Simulated)

2. `prediction_distribution.svg`: Visualization of the prediction distribution

3. `analysis_summary.txt`: Text file containing a summary of the analysis, including:
   - Input file and column information
   - Model information
   - Number and percentage of windows classified as real/simulated
   - Overall classification and confidence

## Model Transferability

The analyze command enables transferability by allowing you to apply models trained on one dataset to new datasets. This is particularly useful when:

- Analyzing new signals with the same structure as the training data
- Working with data from different sources or equipment that may use different column names
- Evaluating signal authenticity in real-world applications

The `--rename` parameter is particularly useful for transferability scenarios where column names might differ between the training and test datasets.

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
is_real, confidence, metrics = analyze_with_model(
    model_path='model.joblib',
    input_file='new_signal.csv',
    target_column='V1',
    output_dir='analysis_results',
    extract_noise=True,
    filter_type='savgol'
)

print(f"Classification: {'REAL' if is_real else 'SIMULATED'} with {confidence:.1f}% confidence")
print(f"Windows classified as real: {metrics['real_windows']}/{metrics['total_windows']} ({metrics['real_windows']/metrics['total_windows']*100:.1f}%)")
```

## Understanding Analysis Results

The overall classification is determined by the proportion of windows classified as real. If more than 50% of windows are classified as real, the entire signal is classified as real. The confidence score represents how far the classification is from the decision boundary, with 100% indicating all windows had the same classification and 0% indicating an even split.

For best results, ensure that:

1. The column structure in your input file matches what the model expects
2. The signal characteristics (sampling rate, units, etc.) are similar to the training data
3. The window size is the same as what was used during training
4. If the model was trained with noise features, use the same noise extraction method during analysis

## License

MIT License
