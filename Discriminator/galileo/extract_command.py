"""
Extract command implementation for Galileo CLI.
"""

import logging
from galileo.features import process_csv_files
from galileo.core import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_FILTER_TYPE,
    DEFAULT_CUTOFF,
    DEFAULT_FS,
    DEFAULT_POLY_ORDER
)

logger = logging.getLogger('galileo.cli.extract')

def setup_parser(parser):
    """
    Set up the argument parser for the extract command.
    
    Args:
        parser (argparse.ArgumentParser): Parser to configure
    """
    parser.add_argument('--data-dir', '-d', 
                       required=True,
                       help='Directory containing CSV files')
    
    parser.add_argument('--output', '-o', 
                       help='Output CSV file path (default: combined_COLUMN_features.csv)')
    
    parser.add_argument('--column', '-c', 
                       required=True,
                       help='Column to extract features from (default: V1)')
    
    parser.add_argument('--window', '-w', 
                       type=int, 
                       default=DEFAULT_WINDOW_SIZE,
                       help=f'Window size for feature extraction (default: {DEFAULT_WINDOW_SIZE})')
    
    parser.add_argument('--rename', '-r', 
                       help='Rename the column prefix in output features')
    
    # Noise extraction arguments
    parser.add_argument('--no-noise', 
                       action='store_true', 
                       help='Disable noise feature extraction')
    
    parser.add_argument('--filter', 
                       choices=['moving_average', 'butterworth', 'savgol', 'kalman'], 
                       default=DEFAULT_FILTER_TYPE,
                       help=f'Filter type for noise extraction (default: {DEFAULT_FILTER_TYPE})')

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
    
    parser.add_argument('--process-variance',
                       type=float,
                       default=1e-5,
                       help='Process variance parameter for Kalman filter (default: 1e-5)')

    parser.add_argument('--measurement-variance',
                       type=float, 
                       default=1e-1,
                       help='Measurement variance parameter for Kalman filter (default: 0.1)')


    return parser


def run(args):
    """
    Run the extract command with the specified arguments.
    
    Args:
        args (argparse.Namespace): Command arguments
        
    Returns:
        int: Exit code
    """
    logger.info(f"Extracting features from {args.data_dir}")
    
    try:
        success = process_csv_files(
            data_directory=args.data_dir,
            output_file=args.output,
            target_column=args.column,
            window_size=args.window,
            extract_noise=not args.no_noise,
            filter_type=args.filter,
            cutoff=args.cutoff,
            fs=args.fs,
            poly_order=args.poly_order,
            output_column_prefix=args.rename,
            process_variance=args.process_variance,
            measurement_variance=args.measurement_variance
        )

        
        if success:
            logger.info("Feature extraction completed successfully")
            return 0
        else:
            logger.error("Feature extraction failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}")
        return 1