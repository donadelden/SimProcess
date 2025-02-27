import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib
import os
from tsfresh.feature_extraction import feature_calculators
from preprocessor import load_data as prep_load_data
from preprocessor import filter_data

def load_data(file_path):
    """Load and preprocess the CSV file using preprocessor functions."""
    try:
        df = prep_load_data(file_path)
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def extract_features(signal):
    """Extract features from a signal."""
    if len(signal) < 2:
        return None
        
    std = feature_calculators.standard_deviation(signal)
    mean = feature_calculators.mean(signal)
    variance = feature_calculators.variance(signal)    
    std_perc = (std / mean * 100) if mean != 0 else 0
    skewness = feature_calculators.skewness(signal)
    approx_entropy = feature_calculators.approximate_entropy(signal,m=2,r=0.1)
    autocorr = feature_calculators.autocorrelation(signal,lag=1)
    
    features = {
        'std': std,
        'variance': variance,
        'std_perc': std_perc,
        'skewness': skewness,
        'approx_entropy':approx_entropy,
        'autocorr':autocorr,
    }
    
    return features

def extract_window_features(df, window_size=10, target_column=None):
    """Extract features from a sliding window of the data with preprocessing."""
    # Validate target_column exists in data
    if target_column and target_column not in df.columns:
        print(f"Error: Specified column '{target_column}' not found in data.")
        print(f"Available columns: {', '.join(df.columns)}")
        return pd.DataFrame(), []
        
    measurement_types = {
        'current': ['C1', 'C2', 'C3'],
        'voltage': ['V1', 'V2', 'V3'],
        'power': ['power_real', 'power_effective', 'power_apparent'],
        'frequency' : ['frequency']
    }
    
    all_features = []
    feature_names = [] 
    
    epsilon = 0.08 
    filtered_df = filter_data(df, window_size=window_size, epsilon=epsilon, target_column=target_column)
    
    for i in range(0, len(filtered_df), window_size//2):  # 50% overlap
        window = filtered_df.iloc[i:i+window_size].copy()
        
        # Skip if window is too small
        if len(window) < window_size:
            continue
            
        initial_size = window_size * len(window.columns)
                
        window = window.dropna()
        
        non_null_count_after = window.count().sum()
        
        # Skip window if it has less than 60% of its original data after removing nulls
        if non_null_count_after < 0.6 * initial_size:
            continue
            
        window_features = {}
        
        # If target_column is specified, only analyze that column
        if target_column:
            if target_column in filtered_df.columns:
                signal = window[target_column].values
                if len(signal) > 0:
                    features = extract_features(signal)
                    if features:
                        for fname, fval in features.items():
                            feature_name = f"{target_column}_{fname}"
                            window_features[feature_name] = fval
                            if feature_name not in feature_names:
                                feature_names.append(feature_name)
        else:
            # Otherwise, analyze all columns by category
            for category, columns in measurement_types.items():
                for col in columns:
                    if col in filtered_df.columns:
                        signal = window[col].values  
                        if len(signal) > 0:
                            features = extract_features(signal)
                            if features:
                                for fname, fval in features.items():
                                    feature_name = f"{col}_{fname}"
                                    window_features[feature_name] = fval
                                    if feature_name not in feature_names:
                                        feature_names.append(feature_name)
        
        if window_features:
            all_features.append(window_features)
    
    return pd.DataFrame(all_features), feature_names

def calculate_feature_importance(svm, X, y, feature_names):
    """Calculate feature importance using permutation importance."""
    result = permutation_importance(svm, X, y, n_repeats=10, random_state=42)
    importance_scores = result.importances_mean
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def plot_feature_importance(importance_df, title="Feature Importance"):
    """Plot feature importance scores."""
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance_df)), importance_df['Importance'])
    plt.xticks(range(len(importance_df)), importance_df['Feature'], rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def train_svm(training_files, labels, target_column=None):
    """Train SVM classifier on multiple files."""
    all_features = []
    all_labels = []
    feature_names = None
    
    for file_path, label in zip(training_files, labels):
        df = load_data(file_path)
        if df is not None:
            features_df, curr_feature_names = extract_window_features(df, target_column=target_column)
            if not features_df.empty:
                all_features.append(features_df)
                all_labels.extend([label] * len(features_df))
                if feature_names is None:
                    feature_names = curr_feature_names
    
    if not all_features:
        print("No valid features extracted from training files")
        return None, None, None
        
    X = pd.concat(all_features, ignore_index=True)
    y = np.array(all_labels)
    
    # Handle NaN values
    X = X.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_scaled, y)
    
    # Calculate feature importance
    importance_df = calculate_feature_importance(svm, X_scaled, y, feature_names)
    plot_feature_importance(importance_df)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    return svm, scaler, feature_names

def analyze_file(file_path, svm, scaler, feature_names, target_column=None):
    """Analyze a single file using trained SVM."""
    df = load_data(file_path)
    if df is None:
        return None
        
    features_df, extracted_feature_names = extract_window_features(df, target_column=target_column)
    if features_df.empty:
        return None
        
    # Handle NaN values
    features_df = features_df.fillna(0)
    
    # Check if any required features are missing
    missing_features = [f for f in feature_names if f not in features_df.columns]
    if missing_features:
        print(f"Error: {len(missing_features)} features used in training were not found in analysis data:")
        for feature in missing_features[:5]:  # Print first 5 missing features for clarity
            print(f"  - Missing feature: '{feature}'")
        if len(missing_features) > 5:
            print(f"  - ... and {len(missing_features) - 5} more")
        print("\nAnalysis cannot proceed. Please ensure you're using the same column(s) for analysis as used in training.")
        return None
        
    # Create aligned features dataframe with same structure as training data
    aligned_features = pd.DataFrame(index=features_df.index)
    for feature in feature_names:
        aligned_features[feature] = features_df[feature]
    
    # Scale features
    X_scaled = scaler.transform(aligned_features)
    
    predictions = svm.predict(X_scaled)
    probabilities = svm.predict_proba(X_scaled)
    
    is_real = np.mean(predictions) > 0.5  # 1 for real, 0 for simulated
    confidence = np.mean(np.max(probabilities, axis=1)) * 100
    
    return {
        'classification': "REAL" if is_real else "SIMULATED",
        'confidence': int(confidence),
        'window_predictions': predictions,
        'window_probabilities': probabilities
    }

def train_and_save_model(training_files, labels, model_path, target_column=None):
    """Train SVM model and save it to the specified path."""
    svm, scaler, feature_names = train_svm(training_files, labels, target_column=target_column)
    
    if svm is not None:
        joblib.dump((svm, scaler, feature_names, target_column), model_path)
        print(f"\nModel saved to {model_path}")
        print("Feature importance plot saved as 'feature_importance.png'")
        return True
    
    return False

def analyze_with_model(file_path, model_path, target_column=None):
    """Load model and analyze the specified file."""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found")
        return None
    
    # Load model
    model_data = joblib.load(model_path)
    
    # Check if the model includes target_column information (for backward compatibility)
    if len(model_data) == 4:
        svm, scaler, feature_names, trained_column = model_data
    else:
        svm, scaler, feature_names = model_data
        trained_column = None
    
    # Use command line column if specified, otherwise use the column the model was trained on
    analysis_column = target_column if target_column else trained_column
    
    if analysis_column:
        print(f"Analyzing only column: {analysis_column}")
    else:
        print("Analyzing all columns")
    
    results = analyze_file(file_path, svm, scaler, feature_names, target_column=analysis_column)
    return results