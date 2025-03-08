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
    
    # Model type selection
    parser.add_argument('--model-type',
                      choices=['svm', 'rf'],
                      default='svm',
                      help='Type of model to train (default: svm)')
    
    # SVM specific parameters
    parser.add_argument('--kernel',
                      choices=['rbf', 'linear', 'poly', 'sigmoid'],
                      default='rbf',
                      help='Kernel type for SVM (default: rbf)')
    
    parser.add_argument('--C',
                      type=float,
                      default=1.0,
                      help='Regularization parameter for SVM (default: 1.0)')
    
    # Random Forest specific parameters
    parser.add_argument('--n-estimators',
                      type=int,
                      default=100,
                      help='Number of trees in Random Forest (default: 100)')
    
    parser.add_argument('--max-depth',
                      type=int,
                      default=None,
                      help='Maximum depth of trees in Random Forest (default: None)')
    
    parser.add_argument('--min-samples-split',
                      type=int,
                      default=2,
                      help='Minimum samples required to split a node in Random Forest (default: 2)')
    
    parser.add_argument('--min-samples-leaf',
                      type=int,
                      default=1,
                      help='Minimum samples required at a leaf node in Random Forest (default: 1)')
    
    return parser


def run(args):
    """
    Run the train command with the specified arguments.
    
    Args:
        args (argparse.Namespace): Command arguments
        
    Returns:
        int: Exit code
    """
    logger.info(f"Training {args.model_type.upper()} model with features file: {args.input}")
    
    # Ensure input is a CSV file
    if not args.input.lower().endswith('.csv'):
        logger.error(f"Input file must be a CSV file containing extracted features")
        return 1
    
    try:
        # Prepare model-specific parameters
        model_params = {}
        
        if args.model_type == 'svm':
            model_params = {
                'kernel': args.kernel,
                'C': args.C,
                'probability': True
            }
            logger.info(f"SVM parameters: kernel={args.kernel}, C={args.C}")
        elif args.model_type == 'rf':
            model_params = {
                'n_estimators': args.n_estimators,
                'random_state': args.seed,
                'n_jobs': -1  # Use all available cores
            }
            
            # Add optional parameters if specified
            if args.max_depth is not None:
                model_params['max_depth'] = args.max_depth
            if args.min_samples_split != 2:
                model_params['min_samples_split'] = args.min_samples_split
            if args.min_samples_leaf != 1:
                model_params['min_samples_leaf'] = args.min_samples_leaf
                
            logger.info(f"Random Forest parameters: n_estimators={args.n_estimators}, "
                        f"max_depth={args.max_depth if args.max_depth else 'None'}, "
                        f"min_samples_split={args.min_samples_split}, "
                        f"min_samples_leaf={args.min_samples_leaf}")
        
        success = train_with_features(
            features_file=args.input,
            model_path=args.model,
            train_ratio=args.train_ratio,
            random_seed=args.seed,
            skip_evaluation=args.no_eval,
            report_file=args.report,
            noise_only=args.noise_only,
            model_type=args.model_type,
            **model_params
        )
        
        if success:
            logger.info(f"{args.model_type.upper()} model training completed successfully")
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