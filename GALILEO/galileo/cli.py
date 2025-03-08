"""
Command-line interface functionality for the Galileo framework.
"""

import os
import logging
import argparse

# Setup logger
logger = logging.getLogger('galileo.cli')

# Import command implementation modules
from galileo import extract_command
from galileo import train_command
from galileo import evaluate_command
from galileo import analyze_command

# Re-export command modules
__all__ = ['extract_command', 'train_command', 'evaluate_command', 'analyze_command']