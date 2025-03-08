"""
Train command implementation for Galileo CLI.
"""

import logging
import os
from galileo.model import train_with_features
from galileo.core import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_RANDOM_SEED,
    DEFAULT_REPORT_PATH
)

logger = logging.getLogger('galileo.cli.train')

def setup_parser(parser):
    """
    Set up the argument parser for the train command.
    
    Args:
        parser (argparse.ArgumentParser): Parser to configure
    """
    parser.add_argument('--input', '-i',
                      required=True,
                      help='Input features CSV file for training')
    
    parser.add_argument('--model', '-m',
                      default=DEFAULT_MODEL_PATH,
                      help=f'Path to save the trained model (default: {DEFAULT_MODEL_PATH})')
    
    parser.add_argument('--train-ratio', '-t',
                      type=float,
                      default=DEFAULT_TRAIN_RATIO,
                      help=f'Ratio of data to use for training (default: {DEFAULT_TRAIN_RATIO})')
    
    parser.add_argument('--seed', '-s',
                      type=int,
                      default=DEFAULT_RANDOM_SEED,
                      help=f'Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})')
    
    parser.add_argument('--no-eval',
                      action='store_true',
                      help='Skip model evaluation')
    
    parser.add_argument('--noise-only',
                      action='store_true',
                      help='Use only noise features for training')

    parser.add_argument('--report', '-r',
                      default=DEFAULT_REPORT_PATH,
                      help=f'Path to save evaluation report (default: {DEFAULT_REPORT_PATH})')
    
    return parser


def run(args):
    """
    Run the train command with the specified arguments.
    
    Args:
        args (argparse.Namespace): Command arguments
        
    Returns:
        int: Exit code
    """
    logger.info(f"Training model with features file: {args.input}")
    
    # Ensure input is a CSV file
    if not args.input.lower().endswith('.csv'):
        logger.error(f"Input file must be a CSV file containing extracted features")
        return 1
    
    try:
        success = train_with_features(
            features_file=args.input,
            model_path=args.model,
            train_ratio=args.train_ratio,
            random_seed=args.seed,
            skip_evaluation=args.no_eval,
            report_file=args.report,
            noise_only=args.noise_only
        )
        
        if success:
            logger.info(f"Model training completed successfully")
            logger.info(f"Model saved to: {args.model}")
            if not args.no_eval:
                logger.info(f"Evaluation report saved to: {args.report}")
            return 0
        else:
            logger.error("Model training failed")
            return 1
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return 1