"""
Analyze command implementation for SimDetector CLI.
"""

import logging
import os
from simdetector.model import analyze_with_model
from simdetector.core import (
    DEFAULT_MODEL_PATH, 
    DEFAULT_WINDOW_SIZE,
    DEFAULT_FILTER_TYPE,
    DEFAULT_CUTOFF,
    DEFAULT_FS,
    DEFAULT_POLY_ORDER,
    ModelError
)

logger = logging.getLogger('simdetector.cli.analyze')

def setup_parser(parser):
    """
    Set up the argument parser for the analyze command.
    
    Args:
        parser (argparse.ArgumentParser): Parser to configure
    """
    parser.add_argument('--model', '-m',
                      default=DEFAULT_MODEL_PATH,
                      help=f'Path to the model to use for analysis (default: {DEFAULT_MODEL_PATH})')
    
    parser.add_argument('--input', '-i',
                      required=True,
                      help='Input CSV file for analysis')
    
    parser.add_argument('--column', '-c',
                      required=True,
                      help='Column to analyze')
    
    parser.add_argument('--rename', '-r',
                      help='Rename the column to match model expectations (e.g., if model was trained on V1 but input uses voltage1)')
    
    parser.add_argument('--output-dir', '-o',
                      default='analysis',
                      help='Directory to save analysis results (default: analysis)')
    
    parser.add_argument('--window', '-w',
                      type=int,
                      default=DEFAULT_WINDOW_SIZE,
                      help=f'Window size for feature extraction (default: {DEFAULT_WINDOW_SIZE})')
    
    # Noise extraction arguments
    parser.add_argument('--no-noise', 
                      action='store_true', 
                      help='Disable noise feature extraction')
    
    parser.add_argument('--filter', 
                       choices=['moving_average', 'butterworth', 'savgol', 'kalman'], 
                       default=DEFAULT_FILTER_TYPE,
                       help=f'Filter type for noise extraction (default: {DEFAULT_FILTER_TYPE})')
    
    # Add parameters for Kalman filter
    parser.add_argument('--process-variance',
                       type=float,
                       default=1e-5,
                       help='Process variance parameter for Kalman filter (default: 1e-5)')

    parser.add_argument('--measurement-variance',
                       type=float, 
                       default=1e-1,
                       help='Measurement variance parameter for Kalman filter (default: 0.1)')
    
  
    parser.add_argument('--cutoff', 
                      type=float, 
                      default=DEFAULT_CUTOFF,
                      help=f'Cutoff frequency for Butterworth filter (default: {DEFAULT_CUTOFF})')
    
    parser.add_argument('--fs', 
                      type=float, 
                      default=DEFAULT_FS,
                      help=f'Sampling frequency for Butterworth filter (default: {DEFAULT_FS})')
    
    parser.add_argument('--poly-order', 
                      type=int, 
                      default=DEFAULT_POLY_ORDER,
                      help=f'Polynomial order for Savitzky-Golay filter (default: {DEFAULT_POLY_ORDER})')
    
    return parser


def run(args):
    """
    Run the analyze command with the specified arguments.
    
    Args:
        args (argparse.Namespace): Command arguments
        
    Returns:
        int: Exit code
    """
    logger.info(f"Analyzing file {args.input} using model {args.model}")
    
    try:
        # Call the analyze_with_model function from model.py
        is_real, confidence, metrics = analyze_with_model(
            model_path=args.model,
            input_file=args.input,
            target_column=args.column,
            output_dir=args.output_dir,
            window_size=args.window,
            column_rename=args.rename,
            extract_noise=not args.no_noise,
            filter_type=args.filter,
            cutoff=args.cutoff,
            fs=args.fs,
            poly_order=args.poly_order,
            process_variance=args.process_variance,
            measurement_variance=args.measurement_variance
        )
        
        # Log the final results
        logger.info(f"\nAnalysis completed successfully.")
        logger.info(f"Signal is classified as: {'REAL' if is_real else 'SIMULATED'}")
        logger.info(f"Confidence: {confidence:.1f}%")
        logger.info(f"Analysis results saved to: {args.output_dir}")
        
        return 0
        
    except ModelError as e:
        logger.error(f"Error during analysis: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1