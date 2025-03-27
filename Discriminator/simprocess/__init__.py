"""
SimProcess - Signal Analysis and Classification Framework

A tool for analyzing time series data, extracting features, 
and classifying signals as real or simulated.
"""

__version__ = "1.0.0"

# Make core modules available at package level
from . import core
from . import data
from . import features
from . import model
from . import visualization

# Import command modules
from . import extract_command
from . import train_command
from . import evaluate_command

# Version information
__all__ = [
    'core',
    'data',
    'features',
    'model',
    'visualization',
    'extract_command',
    'train_command',
    'evaluate_command',
    'analyze_command',
    '__version__'
]