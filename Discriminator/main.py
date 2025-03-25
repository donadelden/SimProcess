#!/usr/bin/env python3
"""
SimDetector - Signal Analysis and Classification Framework

A tool for analyzing time series data, extracting features, 
and classifying signals as real or simulated.
"""

import sys
import argparse
from simdetector.cli import (
    extract_command, 
    train_command, 
    evaluate_command,
    analyze_command
)


def create_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="SimDetector - Signal Analysis and Classification Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Version information
    parser.add_argument('--version', action='version', version='SimDetector v1.0.0')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Extract features command
    extract_parser = subparsers.add_parser(
        'extract', 
        help='Extract features from CSV files'
    )
    extract_command.setup_parser(extract_parser)
    
    # Train model command
    train_parser = subparsers.add_parser(
        'train', 
        help='Train a classification model'
    )
    train_command.setup_parser(train_parser)
    
    # Evaluate model command
    evaluate_parser = subparsers.add_parser(
        'evaluate', 
        help='Evaluate a trained model'
    )
    evaluate_command.setup_parser(evaluate_parser)
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze a CSV file using a trained model'
    )
    analyze_command.setup_parser(analyze_parser)
    
    return parser


def main():
    """Main entry point for the application."""
    parser = create_parser()
    args = parser.parse_args()
    
    # If no command is specified, show help
    if not args.command:
        parser.print_help()
        return 1
    
    # Dispatch to the appropriate command
    if args.command == 'extract':
        return extract_command.run(args)
    elif args.command == 'train':
        return train_command.run(args)
    elif args.command == 'evaluate':
        return evaluate_command.run(args)
    elif args.command == 'analyze':
        return analyze_command.run(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())