"""
Command-line interface functionality for the SimDetector framework.
"""

import os
import logging
import argparse

# Setup logger
logger = logging.getLogger('simdetector.cli')

# Import command implementation modules
from simdetector import extract_command
from simdetector import train_command
from simdetector import evaluate_command
from simdetector import analyze_command

# Re-export command modules
__all__ = ['extract_command', 'train_command', 'evaluate_command', 'analyze_command']