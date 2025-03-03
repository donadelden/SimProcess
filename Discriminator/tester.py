#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import re

def load_windowed_data(file_path):
    """
    Load a pre-processed CSV file containing windowed feature data.
    
    Args:
        file_path (str): Path to the CSV file with windowed features
        
    Returns:
        pandas.DataFrame: Loaded data or None if error
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} windows with {len(df.columns)} features")
        
        # Detect available signal types in the data
        signal_types = detect_signal_types(df)
        if signal_types:
            print(f"Detected signal types: {', '.join(signal_types)}")
        
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def detect_signal_types(df):
    """
    Detect available signal types in the data.
    
    Args:
        df (pandas.DataFrame): Input data with feature columns
        
    Returns:
        list: Detected signal types (V1, V2, C1, power_real, etc.)
    """
    signal_types = set()
    
    # Common prefixes to look for
    patterns = [
        r'^(V\d+)_',      # Voltage signals: V1, V2, V3, etc.
        r'^(C\d+)_',      # Current signals: C1, C2, C3, etc.
        r'^(power_\w+)_', # Power signals: power_real, power_effective, etc.
        r'^(frequency)_'  # Frequency signal
    ]
    
    for column in df.columns:
        for pattern in patterns:
            match = re.match(pattern, column)
            if match:
                signal_types.add(match.group(1))
                break
    
    return sorted(list(signal_types))

def analyze_windowed_data(windowed_file, model_path, signal_type=None, output_file=None):
    """
    Analyze pre-processed windowed data using a trained model.
    
    Args:
        windowed_file (str): Path to the CSV file with windowed features
        model_path (str): Path to the saved model file
        signal_type (str): Signal type to analyze (V1, V2, C1, power_real, etc.)
        output_file (str): Path to save the analysis results (optional)
        
    Returns:
        dict: Analysis results
    """
    # Load the data
    data = load_windowed_data(windowed_file)
    if data is None:
        return None
        
    # Load the model
    try:
        model_data = joblib.load(model_path)
        
        # Check model data structure
        if isinstance(model_data, tuple) and len(model_data) >= 3:
            if len(model_data) == 4:
                # Model with 4 elements (svm, scaler, feature_names, target_column)
                svm, scaler, feature_names, model_signal_type = model_data
                print(f"Model was trained on column: {model_signal_type}")
            else:
                # Model with 3 elements (svm, scaler, feature_names)
                svm, scaler, feature_names = model_data
                model_signal_type = None
            
            print(f"Model loaded from {model_path}")
            print(f"Model expects {len(feature_names)} features")
        else:
            print(f"Error: Unexpected model structure")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Determine which signal type to use
    detected_signal_types = detect_signal_types(data)
    
    # Use the signal type from command line if provided
    if signal_type:
        # Verify signal type exists in data
        if signal_type not in detected_signal_types:
            print(f"Warning: Requested signal type '{signal_type}' not found in data")
            print(f"Available signal types: {', '.join(detected_signal_types)}")
            
            # Try to use model's signal type as fallback
            if model_signal_type and model_signal_type in detected_signal_types:
                print(f"Using model's signal type '{model_signal_type}' instead")
                active_signal_type = model_signal_type
            elif detected_signal_types:
                print(f"Using first available signal type '{detected_signal_types[0]}' instead")
                active_signal_type = detected_signal_types[0]
            else:
                print("No valid signal types found in data")
                return None
        else:
            active_signal_type = signal_type
    # Use model's signal type if available
    elif model_signal_type:
        if model_signal_type in detected_signal_types:
            print(f"Using model's signal type: {model_signal_type}")
            active_signal_type = model_signal_type
        else:
            print(f"Warning: Model's signal type '{model_signal_type}' not found in data")
            if detected_signal_types:
                print(f"Using first available signal type '{detected_signal_types[0]}' instead")
                active_signal_type = detected_signal_types[0]
            else:
                print("No valid signal types found in data")
                return None
    # Use first available signal type as default
    elif detected_signal_types:
        print(f"No signal type specified. Using first available: {detected_signal_types[0]}")
        active_signal_type = detected_signal_types[0]
    else:
        print("No valid signal types found in data")
        return None
        
    print(f"Analyzing with signal type: {active_signal_type}")
        
    # Prepare the features data
    X = data.copy()
    
    # Remove any column that might be a label column to avoid confusion
    for potential_label in ['real', 'label', 'class', 'target']:
        if potential_label in X.columns:
            print(f"Note: Removing column '{potential_label}' as it appears to be a label column")
            X = X.drop(potential_label, axis=1)
    
    # Filter feature_names to only include features for the active signal type
    # or keep all if we can't determine signal specificity
    active_features = []
    signal_specific_model = any(f.startswith(f"{active_signal_type}_") for f in feature_names)
    
    if signal_specific_model:
        # Model is signal-specific, only use matching features
        active_features = [f for f in feature_names if f.startswith(f"{active_signal_type}_") or 
                          f.startswith("noise_")]
        print(f"Using {len(active_features)} features specific to {active_signal_type}")
    else:
        # Model is not signal-specific or can't determine, use all features
        active_features = feature_names
        print(f"Using all {len(active_features)} features from model")
        
    # Check for missing features
    missing_features = [f for f in active_features if f not in X.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} features used in training are missing:")
        for feature in missing_features[:5]:  # Print first 5 missing features
            print(f"  - Missing feature: '{feature}'")
        if len(missing_features) > 5:
            print(f"  - ... and {len(missing_features) - 5} more")
        print("Using available features and filling missing ones with zeros")
        
    # Ensure all required features are present
    for feature in active_features:
        if feature not in X.columns:
            X[feature] = 0  # Fill missing features with zero
            
    # Ensure only model-relevant features are used
    X = X[active_features]
    
    # Handle NaN values
    X = X.fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = svm.predict(X_scaled)
    probabilities = svm.predict_proba(X_scaled)
    
    # Analyze results
    is_real = np.mean(predictions) > 0.5  # 1 for real, 0 for simulated
    confidence = np.mean(np.max(probabilities, axis=1)) * 100
    
    # Prepare results
    results = {
        'classification': "REAL" if is_real else "SIMULATED",
        'confidence': int(confidence),
        'window_predictions': predictions,
        'window_probabilities': probabilities
    }
    
    # Print summary
    print("\nAnalysis Results:")
    print(f"Classification: {results['classification']}")
    print(f"Confidence: {results['confidence']}%")
    
    total_windows = len(predictions)
    real_windows = sum(predictions)
    simulated_windows = total_windows - real_windows
    
    print(f"\nWindow Statistics:")
    print(f"Total windows analyzed: {total_windows}")
    print(f"Windows classified as real: {real_windows} ({real_windows/total_windows*100:.1f}%)")
    print(f"Windows classified as simulated: {simulated_windows} ({simulated_windows/total_windows*100:.1f}%)")
    
    # No evaluation against ground truth since we're just classifying
    
    # Save detailed results to file if requested
    if output_file:
        # Create results DataFrame with window-level predictions
        output_df = data.copy()
        output_df['prediction'] = predictions
        
        # Add probability columns
        if probabilities.shape[1] == 2:
            output_df['prob_simulated'] = probabilities[:, 0]
            output_df['prob_real'] = probabilities[:, 1]
        
        # Save to CSV
        output_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze pre-processed windowed data using a trained model')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file with windowed features')
    parser.add_argument('--model', '-m', required=True, help='Path to the trained model')
    parser.add_argument('--signal', '-s', help='Signal type to analyze (V1, V2, C1, power_real, etc.)')
    parser.add_argument('--output', '-o', help='Path to save detailed results (optional)')
    
    args = parser.parse_args()
    
    # Validate file extensions
    if not args.input.lower().endswith('.csv'):
        print("Error: Input file must be a CSV file")
        return
    
    if not args.model.lower().endswith('.joblib'):
        print("Warning: Model file doesn't have .joblib extension. Will attempt to load anyway.")
    
    # Run analysis
    analyze_windowed_data(
        args.input, 
        args.model,
        signal_type=args.signal,
        output_file=args.output
    )

if __name__ == "__main__":
    main()