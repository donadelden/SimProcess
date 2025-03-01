import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
    approx_entropy = feature_calculators.approximate_entropy(signal, m=2, r=0.3)
    autocorr = feature_calculators.autocorrelation(signal,lag=1)
    kurtosis = feature_calculators.kurtosis(signal)
    #fourier_entropy = feature_calculators.fourier_entropy(signal, bins=20)
    lempev = feature_calculators.lempel_ziv_complexity(signal, bins=20)
    longest_above_mean = feature_calculators.longest_strike_above_mean(signal)
    longest_below_mean = feature_calculators.longest_strike_below_mean(signal)
    n_peaks = feature_calculators.number_peaks(signal, n=1)
    permutation_entropy = feature_calculators.permutation_entropy(signal, tau=1, dimension=4)

    features = {
        'std': std,
        'variance': variance,
        'std_perc': std_perc,
        'skewness': skewness,
        'approx_entropy':approx_entropy,
        'autocorr':autocorr,
        'kurtosis':kurtosis,
        #'fourier_entropy':fourier_entropy,
        'lempev':lempev,
        'longest_above_mean':longest_above_mean,
        'longest_below_mean':longest_below_mean,
        'n_peaks':n_peaks,
        'permutation_entropy':permutation_entropy,
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

# New functions for train-test split functionality

def split_features_dataset(features_file, train_ratio=0.8, random_seed=42):
    """
    Split a features dataset into training and testing sets.
    
    Args:
        features_file (str): Path to the CSV file containing extracted features
        train_ratio (float): Ratio of data to use for training
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    print(f"Loading features from {features_file}")
    try:
        df = pd.read_csv(features_file)
    except Exception as e:
        print(f"Error loading features file: {e}")
        return None, None, None, None, None
    
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Check if 'real' column exists
    if 'real' not in df.columns:
        print("Error: 'real' column not found in the data")
        return None, None, None, None, None
    
    # Extract features and labels
    X = df.drop('real', axis=1)
    y = df['real']
    
    # Report class distribution
    class_counts = y.value_counts()
    print("\nClass distribution in original data:")
    for label, count in class_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(df)*100:.2f}%)")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=random_seed, stratify=y
    )
    
    print(f"\nSplit complete:")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.2f}%)")
    print(f"  Testing set: {len(X_test)} samples ({len(X_test)/len(df)*100:.2f}%)")
    
    # Report class distribution in splits
    train_class_counts = y_train.value_counts()
    test_class_counts = y_test.value_counts()
    
    print("\nClass distribution in training set:")
    for label, count in train_class_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(X_train)*100:.2f}%)")
    
    print("\nClass distribution in testing set:")
    for label, count in test_class_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(X_test)*100:.2f}%)")
    
    feature_names = X.columns.tolist()
    return X_train, X_test, y_train, y_test, feature_names

def train_model_from_features(X_train, y_train, feature_names, model_path=None):
    """
    Train a model on the provided features and save it.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training labels
        feature_names (list): List of feature names
        model_path (str, optional): Path to save the model
        
    Returns:
        tuple: (svm, scaler, feature_names)
    """
    # Handle NaN values
    X_train = X_train.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Train SVM
    print("Training SVM classifier...")
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_scaled, y_train)
    
    # Calculate and plot feature importance
    importance_df = calculate_feature_importance(svm, X_scaled, y_train, feature_names)
    plot_feature_importance(importance_df)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save model if path is provided
    if model_path:
        joblib.dump((svm, scaler, feature_names), model_path)
        print(f"\nModel saved to {model_path}")
        print("Feature importance plot saved as 'feature_importance.png'")
    
    return svm, scaler, feature_names

def evaluate_model(X_test, y_test, svm, scaler, column_name=None, report_file="report.csv", has_noise=0):
    """
    Evaluate a trained model on test data.
    
    Args:
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test labels
        svm: Trained SVM model
        scaler: Trained feature scaler
        column_name (str, optional): Name of the column being analyzed
        report_file (str): CSV file to save the evaluation metrics
        has_noise (int): 1 if noise features were included, 0 otherwise
        
    Returns:
        dict: Evaluation metrics
    """
    # Handle NaN values
    X_test = X_test.fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = svm.predict(X_scaled)
    y_prob = svm.predict_proba(X_scaled)[:, 1]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Extract TP, TN, FP, FN from confusion matrix
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate rates and metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\nDetailed Confusion Matrix Metrics:")
        print(f"True Negatives (TN): {tn} ({tn/(tp+tn+fp+fn)*100:.2f}%) - Correctly classified as not real")
        print(f"False Positives (FP): {fp} ({fp/(tp+tn+fp+fn)*100:.2f}%) - Incorrectly classified as real")
        print(f"False Negatives (FN): {fn} ({fn/(tp+tn+fp+fn)*100:.2f}%) - Incorrectly classified as not real")
        print(f"True Positives (TP): {tp} ({tp/(tp+tn+fp+fn)*100:.2f}%) - Correctly classified as real")
        
        print("\nPerformance Metrics:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall (TPR): {recall:.4f} ({recall*100:.2f}%)")
        print(f"Specificity (TNR): {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"False Positive Rate: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
        print(f"False Negative Rate: {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
        print(f"F1 Score: {f1:.4f}")
        
        # Save metrics to CSV report file
        import os
        import csv
        
        total_windows = len(y_test)
        real_windows = int(sum(y_test))
        simulated_windows = total_windows - real_windows
        
        # Prepare the row to append
        metrics_row = {
            'column': column_name or "unknown",
            'f1score': f1,
            'precision': precision,
            'accuracy': accuracy,
            'TPR': recall,
            'TNR': specificity,
            'FPR': false_positive_rate,
            'FNR': false_negative_rate,
            'total_windows': total_windows,
            'real_windows': real_windows,
            'simulated_windows': simulated_windows,
            'has_noise': has_noise
        }
        
        # Check if report file exists
        file_exists = os.path.isfile(report_file)
        
        # Write header or append to file
        with open(report_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics_row)
        
        print(f"\nEvaluation metrics appended to {report_file}")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Calculate window statistics similar to analyze_with_model
    total_windows = len(y_pred)
    real_windows = sum(y_pred)
    simulated_windows = total_windows - real_windows
    
    print(f"\nWindow Statistics:")
    print(f"Total windows analyzed: {total_windows}")
    print(f"Windows classified as real: {real_windows} ({real_windows/total_windows*100:.1f}%)")
    print(f"Windows classified as simulated: {simulated_windows} ({simulated_windows/total_windows*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_prob
    }

def train_with_features(features_file, model_path, train_ratio=0.8, random_seed=42, skip_evaluation=False, report_file="report.csv"):
    """
    Train and evaluate a model using a features file with train-test split.
    
    Args:
        features_file (str): Path to the CSV file containing extracted features
        model_path (str): Path to save the trained model
        train_ratio (float): Ratio of data to use for training
        random_seed (int): Random seed for reproducibility
        skip_evaluation (bool): Whether to skip model evaluation
        report_file (str): CSV file to save the evaluation metrics
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Extract column name from features filename if possible
    import os
    file_basename = os.path.basename(features_file)
    column_name = None
    has_noise = 0
    
    # Try to extract column name from filename pattern like "combined_V1_features.csv"
    if "_" in file_basename:
        parts = file_basename.split("_")
        if len(parts) >= 2:
            potential_column = parts[1]  # Assume the column name is the second part
            if potential_column.startswith("V") or potential_column.startswith("C") or potential_column in ["frequency", "power"]:
                column_name = potential_column
                
        # Check if "noise" is in the filename
        if "noise" in file_basename.lower():
            has_noise = 1
    
    # Split the dataset
    X_train, X_test, y_train, y_test, feature_names = split_features_dataset(
        features_file, train_ratio, random_seed
    )
    
    if X_train is None:
        return False
    
    # Alternatively, check feature names for noise features
    if has_noise == 0 and any("noise_" in feat_name for feat_name in feature_names):
        has_noise = 1
    
    print(f"Detected: column={column_name}, has_noise={has_noise} ({'with' if has_noise else 'without'} noise features)")
    
    # Train the model
    svm, scaler, _ = train_model_from_features(X_train, y_train, feature_names, model_path)
    
    # Evaluate the model if not skipped
    if not skip_evaluation:
        print("\nEvaluating model on test set:")
        evaluate_model(X_test, y_test, svm, scaler, column_name, report_file, has_noise)
    
    return True