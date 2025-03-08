"""
Analyze command implementation for Galileo CLI.
"""

import logging
import os
import joblib
import pandas as pd
import numpy as np
from galileo.data import load_data, extract_noise_signal
from galileo.features import extract_window_features, extract_features
from galileo.visualization import plot_prediction_distribution
from galileo.core import (
    DEFAULT_MODEL_PATH, 
    DEFAULT_WINDOW_SIZE,
    DEFAULT_FILTER_TYPE,
    DEFAULT_CUTOFF,
    DEFAULT_FS,
    DEFAULT_POLY_ORDER
)

logger = logging.getLogger('galileo.cli.analyze')

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
                      choices=['moving_average', 'butterworth', 'savgol'], 
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
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load model
        logger.info(f"Loading model from {args.model}")
        model_data = joblib.load(args.model)
        
        # Check model format
        if len(model_data) == 4:
            svm, scaler, feature_names, model_column = model_data
        else:
            svm, scaler, feature_names = model_data
            model_column = None
        
        logger.info(f"Model was trained on column: {model_column or 'unknown'}")
        
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = load_data(args.input)
        
        # Verify column exists
        if args.column not in df.columns:
            logger.error(f"Column '{args.column}' not found in the input file")
            logger.info(f"Available columns: {', '.join(df.columns)}")
            return 1
        
        # Determine if we need to rename the column
        target_column = args.column
        output_column_prefix = args.rename if args.rename else target_column
        
        # Check if the model has noise features
        has_noise_features = any('noise_' in feature for feature in feature_names)
        if has_noise_features and args.no_noise:
            logger.warning("Model was trained with noise features, but --no-noise flag is set")
            logger.warning("This may reduce classification accuracy")
        
        # Extract base features
        logger.info(f"Extracting features from column '{target_column}'")
        features_df, extracted_features = extract_window_features(
            df, 
            window_size=args.window, 
            target_column=target_column
        )
        
        if features_df.empty:
            logger.error(f"No features could be extracted from the data")
            return 1
            
        # Extract noise features if needed and not disabled
        if has_noise_features and not args.no_noise:
            logger.info(f"Extracting noise features using {args.filter} filter")
            
            # Process each window to extract noise features
            noise_features = []
            
            for i in range(0, len(df), args.window//2):  # 50% overlap
                window = df.iloc[i:i+args.window].copy()
                
                # Skip if window is too small
                if len(window) < args.window:
                    continue
                    
                # Calculate appropriate window size and poly order for noise extraction
                noise_window_size = max(args.window//5, 3)  # Ensure minimum window size of 3
                # Ensure poly order is always less than window size
                noise_poly_order = min(args.poly_order, noise_window_size - 1)
                
                # Extract noise from the window
                try:
                    noise_df = extract_noise_signal(
                        window, 
                        filter_type=args.filter,
                        window_size=noise_window_size,
                        cutoff=args.cutoff,
                        fs=args.fs,
                        poly_order=noise_poly_order,
                        keep_noise_only=True,
                        target_column=target_column
                    )
                except Exception as e:
                    logger.warning(f"Error extracting noise: {str(e)}")
                    continue
                
                # Skip if noise extraction failed
                if noise_df.empty or target_column not in noise_df.columns:
                    continue
                    
                # Extract features from the noise signal
                from galileo.features import extract_features
                noise_signal = noise_df[target_column].values
                if len(noise_signal) > 1:
                    # Explicitly indicate this is a noise signal
                    noise_feature_dict = extract_features(noise_signal, is_noise=True)
                    if noise_feature_dict:
                        # Add 'noise_' prefix to feature names
                        # If output_column_prefix is different from target_column, use it for noise features too
                        noise_feature_dict = {f"noise_{output_column_prefix}_{k}": v for k, v in noise_feature_dict.items()}
                        noise_features.append(noise_feature_dict)
            
            # If we have noise features, create a DataFrame and align with original features
            if noise_features:
                noise_df = pd.DataFrame(noise_features)
                
                # Make sure we have the same number of rows
                min_rows = min(len(features_df), len(noise_df))
                features_df = features_df.iloc[:min_rows].reset_index(drop=True)
                noise_df = noise_df.iloc[:min_rows].reset_index(drop=True)
                
                # Merge noise features with original features
                for col in noise_df.columns:
                    features_df[col] = noise_df[col]
                
                logger.info(f"Added {len(noise_df.columns)} noise features")
            else:
                logger.warning("No noise features could be extracted")
                if has_noise_features:
                    logger.warning("Model expects noise features, but none were extracted")
                    logger.warning("This may reduce classification accuracy")
        
        # Rename columns if needed to match the model's expected features
        if args.rename and target_column != output_column_prefix:
            rename_map = {}
            for col in features_df.columns:
                if col.startswith(target_column + '_'):
                    new_col = col.replace(target_column + '_', output_column_prefix + '_', 1)
                    rename_map[col] = new_col
            
            if rename_map:
                features_df = features_df.rename(columns=rename_map)
                logger.info(f"Renamed columns to use prefix '{output_column_prefix}'")
        
        # Check for missing features required by the model
        missing_features = [f for f in feature_names if f not in features_df.columns]
        if missing_features:
            logger.warning(f"{len(missing_features)} features used in training were not found in extracted features")
            logger.warning("These features will be filled with zeros")
            
            # Add missing features with zeros
            for feature in missing_features:
                features_df[feature] = 0
                
        # Ensure features are in the same order as expected by the model
        X = features_df[feature_names]
        
        # Replace NaNs with zeros
        X = X.fillna(0)
        
        # Apply the model
        logger.info("Analyzing signals...")
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        y_pred = svm.predict(X_scaled)
        y_prob = svm.predict_proba(X_scaled)[:, 1]  # Probability of being real
        
        # Calculate statistics
        total_windows = len(y_pred)
        real_windows = int(sum(y_pred))
        simulated_windows = total_windows - real_windows
        real_percentage = (real_windows / total_windows * 100) if total_windows > 0 else 0
        
        logger.info(f"\nAnalysis Results:")
        logger.info(f"Total windows analyzed: {total_windows}")
        logger.info(f"Windows classified as real: {real_windows} ({real_percentage:.1f}%)")
        logger.info(f"Windows classified as simulated: {simulated_windows} ({100 - real_percentage:.1f}%)")
        
        # Determine overall classification
        threshold = 0.5  # Threshold for considering the entire signal real
        overall_real_ratio = real_windows / total_windows if total_windows > 0 else 0
        is_real = overall_real_ratio >= threshold
        
        logger.info(f"\nOverall Classification:")
        logger.info(f"Signal is classified as: {'REAL' if is_real else 'SIMULATED'}")
        logger.info(f"Confidence: {abs(overall_real_ratio - 0.5) * 2 * 100:.1f}%")
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'window': range(total_windows),
            'probability': y_prob,
            'prediction': y_pred,
            'classification': ['Real' if p == 1 else 'Simulated' for p in y_pred]
        })
        
        results_file = os.path.join(args.output_dir, "window_predictions.csv")
        results_df.to_csv(results_file, index=False)
        logger.info(f"\nDetailed window predictions saved to: {results_file}")
        
        # Generate prediction distribution plot
        plot_file = os.path.join(args.output_dir, "prediction_distribution.svg")
        plot_prediction_distribution(y_pred, y_prob, output_file=plot_file)
        logger.info(f"Prediction distribution plot saved to: {plot_file}")
        
        # Create summary file
        summary_file = os.path.join(args.output_dir, "analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Galileo Analysis Summary\n")
            f.write(f"======================\n\n")
            f.write(f"Input file: {args.input}\n")
            f.write(f"Analyzed column: {target_column}\n")
            f.write(f"Model used: {args.model}\n")
            f.write(f"Model trained on: {model_column or 'unknown'}\n\n")
            
            f.write(f"Analysis Results:\n")
            f.write(f"Total windows analyzed: {total_windows}\n")
            f.write(f"Windows classified as real: {real_windows} ({real_percentage:.1f}%)\n")
            f.write(f"Windows classified as simulated: {simulated_windows} ({100 - real_percentage:.1f}%)\n\n")
            
            f.write(f"Overall Classification:\n")
            f.write(f"Signal is classified as: {'REAL' if is_real else 'SIMULATED'}\n")
            f.write(f"Confidence: {abs(overall_real_ratio - 0.5) * 2 * 100:.1f}%\n")
            
        logger.info(f"Summary report saved to: {summary_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1