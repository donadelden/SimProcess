"""
Command-line interface functionality for the SimProcess framework.
"""

import os
import logging
import argparse

# Setup logger
logger = logging.getLogger('simprocess.cli')

# Import command implementation modules
from simprocess import extract_command
from simprocess import train_command
from simprocess import evaluate_command
from simprocess import analyze_command

# Re-export command modules
__all__ = ['extract_command', 'train_command', 'evaluate_command', 'analyze_command']