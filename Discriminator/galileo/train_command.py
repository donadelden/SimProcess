"""
Train command implementation for Galileo CLI.
"""

import logging
import os
from galileo.core import (
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORT_PATH
)
from galileo.ml import train_binary, train_reducing_features

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
    
    parser.add_argument('--report', '-r',
                      default=DEFAULT_REPORT_PATH,
                      help=f'Path to save evaluation report (default: {DEFAULT_REPORT_PATH})')
    
    # Training mode
    parser.add_argument('--training-mode',
                      choices=['advanced', 'feature-reduction'],
                      default='advanced',
                      help='Type of training approach to use (default: advanced)')
    
    # Advanced ML options
    parser.add_argument('--fast-mode',
                      type=int,
                      choices=[0, 1, 2],
                      default=1,
                      help='Training speed mode: 0=full, 1=reduced, 2=minimal (default: 1)')
    
    parser.add_argument('--balancing-ratio',
                      type=float,
                      default=0.9,
                      help='Class balancing ratio for SMOTE (default: 0.9)')
    
    parser.add_argument('--features-to-keep',
                      type=int,
                      default=11,
                      help='Number of top features to keep (default: 11)')
    
    parser.add_argument('--max-features',
                      type=int,
                      default=20,
                      help='Maximum number of features to use in feature reduction mode (default: 20)')
    
    parser.add_argument('--fluctuation',
                      action='store_true',
                      help='Test on fluctuating data')
    
    parser.add_argument('--no-eval',
                      action='store_true',
                      help='Skip saving evaluation report')
    
    return parser


def run(args):
    """
    Run the train command with the specified arguments.
    
    Args:
        args (argparse.Namespace): Command arguments
        
    Returns:
        int: Exit code
    """
    logger.info(f"Training with mode: {args.training_mode}")
    
    # Ensure input is a CSV file
    if not args.input.lower().endswith('.csv'):
        logger.error(f"Input file must be a CSV file containing extracted features")
        return 1
    
    try:
        # Set up report file path
        report_file = args.report if not args.no_eval else None
        
        if args.training_mode == 'feature-reduction':
            # Use feature reduction training method from ml.py
            logger.info(f"Using feature reduction training method with max_features={args.max_features}, balancing_ratio={args.balancing_ratio}")
            
            success = train_reducing_features(
                features_file=args.input,
                model_path=args.model,
                report_file=report_file,
                max_features=args.max_features,
                balRatio=args.balancing_ratio,
                fast_mode=args.fast_mode
            )
            
        else:  # args.training_mode == 'advanced'
            # Use advanced ML training method from ml.py
            logger.info(f"Using advanced ML training method with fast_mode={args.fast_mode}, balancing_ratio={args.balancing_ratio}")
            
            success = train_binary(
                features_file=args.input,
                model_path=args.model,
                report_file=report_file,
                features_to_keep=args.features_to_keep,
                dataset_balancing_ratio=args.balancing_ratio,
                fast_mode=args.fast_mode,
                fluctuation=args.fluctuation
            )
        
        if success:
            logger.info(f"Model training completed successfully")
            logger.info(f"Model saved to: {args.model}")
            if report_file:
                logger.info(f"Evaluation report saved to: {report_file}")
            return 0
        else:
            logger.error("Model training failed")
            return 1
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1