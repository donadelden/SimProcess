import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import argparse
import joblib
import os

def load_data(file_path):
    """Load and preprocess the CSV file."""
    try:
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def extract_features(signal):
    """Extract statistical features from a signal."""
    if len(signal) < 2:
        return None
        
    # Basic statistics 
    std = np.std(signal)
    mean = np.mean(signal)  
    variance = std ** 2
    diffs = np.diff(signal)
    diff_mean = np.mean(np.abs(diffs)) 
    diff_std = np.std(diffs)
    
    # Normalized measure (standard deviation as percentage of mean)
    std_perc = (std / mean * 100) if mean != 0 else 0
    diff_std_perc = diff_std / diff_mean if diff_mean != 0 else 0
   
    return {
        'std': std,
        'variance': variance,
        'std_perc': std_perc,
    }

def extract_window_features(df, window_size=10):
    """Extract features from a sliding window of the data."""
    measurement_types = {
        'current': ['C1', 'C2', 'C3'],
        'voltage': ['V1', 'V2', 'V3'],
        'power': ['power_real', 'power_effective', 'power_apparent'],
        'frequency' : ['frequency']
    }
    
    all_features = []
    feature_names = []  # Track feature names
    
    for i in range(0, len(df), window_size//2):  # 50% overlap
        window = df.iloc[i:i+window_size]
        if len(window) < window_size:
            continue
            
        window_features = {}
        
        for category, columns in measurement_types.items():
            for col in columns:
                if col in df.columns:
                    signal = window[col].dropna().values
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
    
    # Create DataFrame with feature names and importance scores
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

def train_svm(training_files, labels):
    """Train SVM classifier on multiple files."""
    all_features = []
    all_labels = []
    feature_names = None
    
    for file_path, label in zip(training_files, labels):
        df = load_data(file_path)
        if df is not None:
            features_df, curr_feature_names = extract_window_features(df)
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

def analyze_file(file_path, svm, scaler, feature_names):
    """Analyze a single file using trained SVM."""
    df = load_data(file_path)
    if df is None:
        return None
        
    features_df, _ = extract_window_features(df)
    if features_df.empty:
        return None
        
    # Handle NaN values
    features_df = features_df.fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(features_df)
    
    # Get predictions and probabilities
    predictions = svm.predict(X_scaled)
    probabilities = svm.predict_proba(X_scaled)
    
    # Get results
    is_real = np.mean(predictions) > 0.5  # 1 for real, 0 for simulated
    confidence = np.mean(np.max(probabilities, axis=1)) * 100
    
    return {
        'classification': "REAL" if is_real else "SIMULATED",
        'confidence': int(confidence),
        'window_predictions': predictions,
        'window_probabilities': probabilities
    }

def main():
    parser = argparse.ArgumentParser(description='Train or use SVM to discriminate between real and simulated power plant data.')
    parser.add_argument('mode', choices=['train', 'analyze'], help='Mode of operation')
    parser.add_argument('--input', '-i', required=True, help='Input file for analysis or directory containing training files')
    parser.add_argument('--real', '-r', nargs='*', help='List of real data files for training')
    parser.add_argument('--simulated', '-s', nargs='*', help='List of simulated data files for training')
    parser.add_argument('--model', '-m', default='svm_model.joblib', help='Path to save/load model')
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.real or not args.simulated:
            print("Need both real and simulated files for training")
            return
            
        # Prepare training data + labels
        training_files = args.real + args.simulated
        labels = [1] * len(args.real) + [0] * len(args.simulated)
        
        # Train model
        svm, scaler, feature_names = train_svm(training_files, labels)
        if svm is not None:
            # Save model
            joblib.dump((svm, scaler, feature_names), args.model)
            print(f"\nModel saved to {args.model}")
            print("Feature importance plot saved as 'feature_importance.png'")
    
    elif args.mode == 'analyze':
        if not os.path.exists(args.model):
            print(f"Model file {args.model} not found")
            return
            
        # Load model
        svm, scaler, feature_names = joblib.load(args.model)
        
        results = analyze_file(args.input, svm, scaler, feature_names)
        if results:
            print("\n=== Analysis Results ===")
            print(f"Classification: {results['classification']}")
            print(f"Confidence: {results['confidence']}%")
            
            window_results = pd.Series(results['window_predictions'])
            print(f"\nWindow Statistics:")
            print(f"Total windows analyzed: {len(window_results)}")
            real_windows = sum(window_results)
            simulated_windows = len(window_results) - real_windows
            print(f"Windows classified as real: {real_windows} ({real_windows/len(window_results)*100:.1f}%)")
            print(f"Windows classified as simulated: {simulated_windows} ({simulated_windows/len(window_results)*100:.1f}%)")

if __name__ == "__main__":
    main()