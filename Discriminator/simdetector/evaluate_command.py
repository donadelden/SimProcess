"""
Evaluate command implementation for SimDetector CLI.
"""

import logging
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from simdetector.model import split_features_dataset, evaluate_model
from simdetector.visualization import (
    plot_confusion_matrix,
    plot_prediction_distribution,
    generate_report_summary
)
from simdetector.core import (
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_TRAIN_RATIO
)

logger = logging.getLogger('simdetector.cli.evaluate')

def setup_parser(parser):
    """
    Set up the argument parser for the evaluate command.
    
    Args:
        parser (argparse.ArgumentParser): Parser to configure
    """
    parser.add_argument('--model', '-m',
                      default=DEFAULT_MODEL_PATH,
                      help=f'Path to the model to evaluate (default: {DEFAULT_MODEL_PATH})')
    
    parser.add_argument('--input', '-i',
                      required=True,
                      help='Input features CSV file for evaluation')
    
    parser.add_argument('--test-only', '-t',
                      action='store_true',
                      help='Use the entire input file as test set instead of splitting')
    
    parser.add_argument('--output-dir', '-o',
                      default='evaluation',
                      help='Directory to save evaluation results (default: evaluation)')
    
    parser.add_argument('--report', '-r',
                      default=DEFAULT_REPORT_PATH,
                      help=f'Path to save evaluation report (default: {DEFAULT_REPORT_PATH})')
    
    parser.add_argument('--column', '-c',
                      help='Name of the column being evaluated (for reporting)')
    
    return parser


def run(args):
    """
    Run the evaluate command with the specified arguments.
    
    Args:
        args (argparse.Namespace): Command arguments
        
    Returns:
        int: Exit code
    """
    logger.info(f"Evaluating model: {args.model}")
    
    try:
        # Ensure input is a CSV file
        if not args.input.lower().endswith('.csv'):
            logger.error(f"Input file must be a CSV file containing extracted features")
            return 1
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load model
        logger.info(f"Loading model from {args.model}")
        model_data = joblib.load(args.model)
        
        # Check model format
        if len(model_data) == 4:
            svm, scaler, feature_names, column_name = model_data
        else:
            svm, scaler, feature_names = model_data
            column_name = None
            
        # Use column name from arguments if provided
        if args.column:
            column_name = args.column
        
        # Log column information
        logger.info(f"Model trained on column: {column_name or 'unknown'}")
        
        # Load evaluation data
        if args.test_only:
            # Use entire file as test set
            logger.info(f"Loading test data from {args.input}")
            df = pd.read_csv(args.input)
            
            if 'real' not in df.columns:
                logger.error("'real' column not found in the test data")
                return 1
                
            X_test = df.drop('real', axis=1)
            y_test = df['real']
            
            logger.info(f"Test set: {len(X_test)} samples")
        else:
            # Split data into train/test
            logger.info(f"Splitting data from {args.input}")
            _, X_test, _, y_test, _ = split_features_dataset(
                args.input, train_ratio=1-DEFAULT_TRAIN_RATIO
            )
        
        # Check for missing features
        missing_features = [f for f in feature_names if f not in X_test.columns]
        if missing_features:
            logger.error(f"{len(missing_features)} features used in training were not found in test data")
            if len(missing_features) > 5:
                logger.error(f"First 5 missing features: {', '.join(missing_features[:5])}")
                logger.error(f"... and {len(missing_features) - 5} more")
            return 1
            
        # Evaluate model
        logger.info("Evaluating model...")
        has_noise = 1 if any("noise_" in feat for feat in feature_names) else 0
        
        results = evaluate_model(
            X_test, y_test, svm, scaler, 
            column_name=column_name, 
            report_file=args.report,
            has_noise=has_noise
        )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        # Confusion matrix plot
        cm_file = os.path.join(args.output_dir, f"confusion_matrix_{column_name or 'model'}.svg")
        plot_confusion_matrix(
            results['confusion_matrix'],
            output_file=cm_file,
            title=f"Confusion Matrix - {column_name or 'Model'}"
        )
        
        # Prediction distribution plot
        dist_file = os.path.join(args.output_dir, f"prediction_dist_{column_name or 'model'}.svg")
        plot_prediction_distribution(
            results['predictions'],
            results['probabilities'],
            output_file=dist_file
        )
        
        # Report summary if report file exists
        if os.path.exists(args.report):
            summary_file = os.path.join(args.output_dir, "report_summary.svg")
            generate_report_summary(args.report, output_file=summary_file)
        
        logger.info(f"Evaluation completed successfully")
        logger.info(f"Results saved to: {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        return 1