"""
Core functionality and constants for the SimProcess framework.
"""

import os
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('simprocess')

# Constants
DEFAULT_WINDOW_SIZE = 20
DEFAULT_FILTER_TYPE = 'kalman'
DEFAULT_CUTOFF = 0.1
DEFAULT_FS = 1.0
DEFAULT_POLY_ORDER = 2
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_RANDOM_SEED = 42
DEFAULT_MODEL_PATH = 'simprocess_model.joblib'
DEFAULT_REPORT_PATH = 'evaluation_report.csv'
DEFAULT_PROCESS_VARIANCE = 1e-5
DEFAULT_MEASUREMENT_VARIANCE = 1e-1


# Signal types and their corresponding columns
SIGNAL_TYPES = {
    'current': ['C1', 'C2', 'C3'],
    'voltage': ['V1', 'V2', 'V3'],
    'power': ['power_real', 'power_reactive', 'power_apparent'],
    'frequency': ['frequency']
}

# Critical feature types that indicate signal quality
CRITICAL_FEATURES = [
    'std', 'variance', 'entropy', 'autocorr', 'kurtosis'
]

class SimProcess(Exception):
    """Base exception for SimProcess framework errors."""
    pass

class DataLoadError(SimProcess):
    """Exception raised when data loading fails."""
    pass

class FeatureExtractionError(SimProcess):
    """Exception raised when feature extraction fails."""
    pass

class ModelError(SimProcess):
    """Exception raised for model-related errors."""
    pass

def validate_file_path(file_path, extension='.csv'):
    """
    Validate that a file path exists and has the expected extension.
    
    Args:
        file_path (str): Path to validate
        extension (str): Expected file extension
        
    Returns:
        bool: True if valid, raises exception otherwise
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.lower().endswith(extension):
        raise ValueError(f"File must have {extension} extension: {file_path}")
    
    return True

def validate_directory(directory_path):
    """
    Validate that a directory exists.
    
    Args:
        directory_path (str): Directory to validate
        
    Returns:
        bool: True if valid, raises exception otherwise
    """
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Directory not found: {directory_path}")
    
    return True

def is_numeric_column(df, column):
    """
    Check if a column contains numeric data.
    
    Args:
        df (pandas.DataFrame): DataFrame to check
        column (str): Column name to check
        
    Returns:
        bool: True if the column is numeric
    """
    return np.issubdtype(df[column].dtype, np.number)

def get_column_type(column_name):
    """
    Determine the type of a column based on its name.
    
    Args:
        column_name (str): Name of the column
        
    Returns:
        str: Type of the column ('current', 'voltage', 'power', 'frequency', or 'unknown')
    """
    column_name = column_name.lower()
    
    if column_name.startswith('c') and len(column_name) == 2 and column_name[1].isdigit():
        return 'current'
    elif column_name.startswith('v') and len(column_name) == 2 and column_name[1].isdigit():
        return 'voltage'
    elif 'power' in column_name:
        return 'power'
    elif 'frequency' in column_name:
        return 'frequency'
    else:
        return 'unknown'