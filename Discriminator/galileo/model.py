"""
Model training, evaluation, and prediction for the Galileo framework.
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
from galileo.data import load_data, extract_noise_signal
from galileo.features import extract_window_features, extract_features
from galileo.core import ModelError, validate_file_path
from galileo.visualization import plot_feature_importance, plot_prediction_distribution

logger = logging.getLogger('galileo.model')

def split_features_dataset(features_file, train_ratio=0.8, random_seed=42, noise_only=False):
    """
    Split a features dataset into training and testing sets.
    
    Args:
        features_file (str): Path to the CSV file containing extracted features
        train_ratio (float): Ratio of data to use for training
        random_seed (int): Random seed for reproducibility
        noise_only (bool): Whether to use only noise features for training
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    logger.info(f"Loading features from {features_file}")
    try:
        validate_file_path(features_file, extension='.csv')
        df = pd.read_csv(features_file)
    except Exception as e:
        logger.error(f"Error loading features file: {str(e)}")
        raise ModelError(f"Failed to load features file: {str(e)}")
    
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Check if 'real' column exists
    if 'real' not in df.columns:
        logger.error("'real' column not found in the data")
        raise ModelError("Missing 'real' column in features file")
    
    # Extract features and labels
    X = df.drop('real', axis=1)
    y = df['real']
    
    # Filter for noise features only if requested
    if noise_only:
        noise_cols = [col for col in X.columns if 'noise_' in col]
        if not noise_cols:
            logger.error("No noise features found in the dataset. Make sure you extracted features with noise.")
            raise ModelError("No noise features found when noise_only=True")
        
        logger.info(f"Using only noise features for training ({len(noise_cols)} features)")
        X = X[noise_cols]
    
    # Report class distribution
    class_counts = y.value_counts()
    logger.info("Class distribution in original data:")
    for label, count in class_counts.items():
        logger.info(f"  Class {label}: {count} samples ({count/len(df)*100:.2f}%)")    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=random_seed, stratify=y
    )
    
    logger.info(f"Split complete:")
    logger.info(f"  Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.2f}%)")
    logger.info(f"  Testing set: {len(X_test)} samples ({len(X_test)/len(df)*100:.2f}%)")
    
    # Report class distribution in splits
    train_class_counts = y_train.value_counts()
    test_class_counts = y_test.value_counts()
    
    logger.info("Class distribution in training set:")
    for label, count in train_class_counts.items():
        logger.info(f"  Class {label}: {count} samples ({count/len(X_train)*100:.2f}%)")
    
    logger.info("Class distribution in testing set:")
    for label, count in test_class_counts.items():
        logger.info(f"  Class {label}: {count} samples ({count/len(X_test)*100:.2f}%)")
    
    feature_names = X.columns.tolist()
    return X_train, X_test, y_train, y_test, feature_names


def calculate_feature_importance(model, X, y, feature_names, model_type="svm"):
    """
    Calculate feature importance using appropriate method for the model type.
    
    Args:
        model: Trained model with predict method
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        model_type: Type of model ("svm", "rf", "ocsvm")
        
    Returns:
        pandas.DataFrame: DataFrame with feature importance scores
    """
    if model_type.lower() == "rf":
        # For Random Forest, use built-in feature importance
        importance_scores = model.feature_importances_
    elif model_type.lower() == "ocsvm":
        # For One-Class SVM, we need a custom approach since permutation importance
        # doesn't work well with one-class classifiers
        
        # Get the target class
        target_class = getattr(model, 'target_class', 1)
        
        # Filter X to only include samples from the target class
        X_target = X[y == target_class]
        if len(X_target) == 0:
            logger.warning("No samples from target class found for feature importance calculation")
            # Return zero importance for all features as fallback
            importance_scores = np.zeros(len(feature_names))
        else:
            # Define a scoring function that doesn't rely on y
            # We'll use the average decision function value as our score
            # (higher = better for inliers)
            def ocsvm_score_no_y(estimator, X_subset):
                decision_values = estimator.decision_function(X_subset)
                return np.mean(decision_values)
            
            # Define a permutation importance function for OCSVM
            # We'll manually implement a simplified version
            importance_scores = []
            baseline_score = ocsvm_score_no_y(model, X_target)
            
            # Calculate importance for each feature
            for i in range(X_target.shape[1]):
                # Create a copy of the data and permute one feature
                X_permuted = X_target.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                
                # Calculate new score
                permuted_score = ocsvm_score_no_y(model, X_permuted)
                
                # Importance is the decrease in score
                importance = baseline_score - permuted_score
                importance_scores.append(importance)
            
            importance_scores = np.array(importance_scores)
    else:
        # For other models (like SVM), use permutation importance
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importance_scores = result.importances_mean
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df


def train_model_from_features(X_train, y_train, feature_names, model_path=None, metric_name=None, 
                             model_type="svm", **model_params):
    """
    Train a model on the provided features and save it.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training labels
        feature_names (list): List of feature names
        model_path (str, optional): Path to save the model
        metric_name (str, optional): Name of the metric being analyzed
        model_type (str): Type of model to train ("svm", "rf", "ocsvm")
        **model_params: Additional parameters for the model
        
    Returns:
        tuple: (model, scaler, feature_names)
    """
    # Handle NaN values
    X_train = X_train.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Create model based on model_type
    logger.info(f"Training {model_type.upper()} classifier...")
    
    if model_type.lower() == "rf":
        # Default parameters for Random Forest
        params = {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }
        # Update with any user-provided parameters
        params.update(model_params)
        
        # Create and train the Random Forest
        model = RandomForestClassifier(**params)
        model.fit(X_scaled, y_train)
    elif model_type.lower() == "ocsvm":
        # Default parameters for One-Class SVM
        params = {
            'kernel': 'rbf',
            'nu': 0.1,
            'gamma': 'scale'
        }
        # Update with any user-provided parameters, excluding 'target_class'
        params.update({k: v for k, v in model_params.items() if k not in ['target_class']})
        
        # Extract target class (which class to consider as "normal")
        target_class = model_params.get('target_class', 1)  # Default to class 1 (real)
        
        # Filter training data to only include the target class
        X_target = X_scaled[y_train == target_class]
        
        if len(X_target) == 0:
            logger.error(f"No samples found for target class {target_class}")
            raise ModelError(f"No samples found for target class {target_class}")
        
        logger.info(f"Training One-Class SVM on {len(X_target)} samples of class {'real' if target_class == 1 else 'simulated'}")
        
        # Create and train the One-Class SVM
        model = OneClassSVM(**params)
        model.fit(X_target)
        
        # Store the target class with the model
        model.target_class = target_class
    else:  # Default to SVM
        # Default parameters for SVM
        params = {
            'kernel': 'rbf',
            'C': 1.0,
            'probability': True
        }
        # Update with any user-provided parameters
        params.update(model_params)
        
        # Create and train the SVM
        model = SVC(**params)
        model.fit(X_scaled, y_train)
    
    # Calculate feature importance
    importance_df = calculate_feature_importance(model, X_scaled, y_train, feature_names, model_type)
    
    logger.info("Top 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['Feature']}: {row['Importance']:.6f}")
    
    # Plot feature importance
    output_dir = os.path.dirname(model_path) if model_path else None
    if output_dir == '':
        output_dir = None
        
    plot_file = plot_feature_importance(
        importance_df, 
        metric_name=metric_name,
        output_dir=output_dir
    )
    
    if plot_file:
        logger.info(f"Feature importance plot saved as: {plot_file}")
    
    # Save model if path is provided
    if model_path:
        if model_type.lower() == "ocsvm":
            # Store model_type along with the model data and target_class
            joblib.dump((model, scaler, feature_names, metric_name, model_type, model_params.get('target_class', 1)), model_path)
        else:
            # Store model_type along with the model data
            joblib.dump((model, scaler, feature_names, metric_name, model_type), model_path)
        logger.info(f"{model_type.upper()} model saved to {model_path}")
        
    return model, scaler, feature_names

def evaluate_model(X_test, y_test, model, scaler, column_name=None, report_file="report.csv", has_noise=0, model_type="svm"):
    """
    Evaluate a trained model on test data.
    
    Args:
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test labels
        model: Trained model (SVM, RandomForest, etc.)
        scaler: Trained feature scaler
        column_name (str, optional): Name of the column being analyzed
        report_file (str): CSV file to save the evaluation metrics
        has_noise (int): 1 if noise features were included, 0 otherwise
        model_type (str): Type of model ("svm", "rf", "ocsvm")
        
    Returns:
        dict: Evaluation metrics
    """
    # Handle NaN values
    X_test = X_test.fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X_test)
    
    # Make predictions based on model type
    if model_type.lower() == "ocsvm":
        # Get the target class that was used for training
        target_class = getattr(model, 'target_class', 1)  # Default to 1 if not stored
        
        # For OCSVM, predictions are 1 for inliers and -1 for outliers
        raw_pred = model.predict(X_scaled)
        
        # Convert to binary classification format:
        # If target_class is 1 (real), then inliers (1) -> 1, outliers (-1) -> 0
        # If target_class is 0 (simulated), then inliers (1) -> 0, outliers (-1) -> 1
        if target_class == 1:
            y_pred = (raw_pred == 1).astype(int)  # 1 for inliers, 0 for outliers
        else:
            y_pred = (raw_pred == -1).astype(int)  # 1 for outliers, 0 for inliers
        
        # OCSVM doesn't provide probability estimates by default
        # We can use decision_function as a proxy for confidence
        decision_values = model.decision_function(X_scaled)
        
        # Normalize to [0, 1] range for consistency
        min_val, max_val = np.min(decision_values), np.max(decision_values)
        if max_val > min_val:
            normalized_values = (decision_values - min_val) / (max_val - min_val)
        else:
            normalized_values = np.zeros_like(decision_values)
        
        # For target_class=1, higher decision values -> higher probability of class 1
        # For target_class=0, lower decision values -> higher probability of class 1
        if target_class == 1:
            y_prob = normalized_values
        else:
            y_prob = 1 - normalized_values
            
        logger.info(f"One-Class SVM target class: {'Real' if target_class == 1 else 'Simulated'}")
    else:
        # Original prediction code for SVM and RF
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]  # Probability of being real
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:")
    logger.info(str(cm))
    
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
        
        logger.info("Detailed Confusion Matrix Metrics:")
        logger.info(f"True Negatives (TN): {tn} ({tn/(tp+tn+fp+fn)*100:.2f}%) - Correctly classified as not real")
        logger.info(f"False Positives (FP): {fp} ({fp/(tp+tn+fp+fn)*100:.2f}%) - Incorrectly classified as real")
        logger.info(f"False Negatives (FN): {fn} ({fn/(tp+tn+fp+fn)*100:.2f}%) - Incorrectly classified as not real")
        logger.info(f"True Positives (TP): {tp} ({tp/(tp+tn+fp+fn)*100:.2f}%) - Correctly classified as real")
        
        logger.info("Performance Metrics:")
        logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        logger.info(f"Recall (TPR): {recall:.4f} ({recall*100:.2f}%)")
        logger.info(f"Specificity (TNR): {specificity:.4f} ({specificity*100:.2f}%)")
        logger.info(f"False Positive Rate: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
        logger.info(f"False Negative Rate: {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Save metrics to CSV report file
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
            'has_noise': has_noise,
            'model_type': model_type
        }
        
        # Check if report file exists
        file_exists = os.path.isfile(report_file)
        
        # Write header or append to file
        with open(report_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics_row)
        
        logger.info(f"Evaluation metrics appended to {report_file}")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    logger.info("Classification Report:")
    logger.info(report)
    
    # Calculate window statistics similar to analyze_with_model
    total_windows = len(y_pred)
    real_windows = sum(y_pred)
    simulated_windows = total_windows - real_windows
    
    logger.info("Window Statistics:")
    logger.info(f"Total windows analyzed: {total_windows}")
    logger.info(f"Windows classified as real: {real_windows} ({real_windows/total_windows*100:.1f}%)")
    logger.info(f"Windows classified as simulated: {simulated_windows} ({simulated_windows/total_windows*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_prob
    }

def train_with_features(features_file, model_path, train_ratio=0.8, random_seed=42, 
                       skip_evaluation=False, report_file="report.csv", noise_only=False,
                       model_type="svm", **model_params):
    """
    Train and evaluate a model using a features file with train-test split.
    
    Args:
        features_file (str): Path to the CSV file containing extracted features
        model_path (str): Path to save the trained model
        train_ratio (float): Ratio of data to use for training
        random_seed (int): Random seed for reproducibility
        skip_evaluation (bool): Whether to skip model evaluation
        report_file (str): CSV file to save the evaluation metrics
        noise_only (bool): Whether to use only noise features for training
        model_type (str): Type of model to train ("svm", "rf")
        **model_params: Additional parameters for the model
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract column name from features filename if possible
        file_basename = os.path.basename(features_file)
        column_name = None
        has_noise = 0
        
        # Try to extract column name from filename pattern like "combined_V1_features.csv"
        if "_" in file_basename:
            parts = file_basename.split("_")
            
            # Handle voltage and current columns (V1, V2, V3, C1, C2, C3)
            if len(parts) >= 2:
                potential_column = parts[1]  # Assume the column name is the second part
                if potential_column.startswith("V") or potential_column.startswith("C"):
                    column_name = potential_column
                elif potential_column == "frequency":
                    column_name = potential_column
            
            # Handle power types (power_real, power_apparent, power_effective)
            if "power" in file_basename:
                if "power_real" in file_basename:
                    column_name = "power_real"
                elif "power_apparent" in file_basename:
                    column_name = "power_apparent"
                elif "power_effective" in file_basename:
                    column_name = "power_effective"
                elif len(parts) >= 2 and parts[1] == "power":
                    column_name = "power"  # Generic fallback if specific power type not identified
                    
            # Check if "noise" is in the filename
            if "noise" in file_basename.lower():
                has_noise = 1
        
        # Split the dataset
        X_train, X_test, y_train, y_test, feature_names = split_features_dataset(
            features_file, train_ratio, random_seed, noise_only
        )
        
        # Check feature names for noise features
        if has_noise == 0 and any("noise_" in feat_name for feat_name in feature_names):
            has_noise = 1
        
        logger.info(f"Detected: column={column_name}, has_noise={has_noise} ({'with' if has_noise else 'without'} noise features)")
        logger.info(f"Using model type: {model_type}")
        
        # Train the model
        model, scaler, _ = train_model_from_features(
            X_train, y_train, feature_names, model_path, 
            metric_name=column_name, model_type=model_type, **model_params
        )
        
        # Evaluate the model if not skipped
        if not skip_evaluation:
            logger.info("Evaluating model on test set:")
            evaluate_model(X_test, y_test, model, scaler, column_name, report_file, has_noise, model_type)
        
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise ModelError(f"Failed to train model: {str(e)}")

def analyze_with_model(model_path, input_file, target_column, output_dir="analysis", 
                      window_size=10, column_rename=None, extract_noise=True, 
                      filter_type="savgol", cutoff=0.1, fs=1.0, poly_order=2,
                      process_variance=1e-5, measurement_variance=1e-1):
    """
    Analyze a CSV file using a trained model to classify signals as real or simulated.
    
    Args:
        model_path (str): Path to the trained model file
        input_file (str): Path to the CSV file to analyze
        target_column (str): Column to analyze
        output_dir (str): Directory to save analysis results
        window_size (int): Window size for feature extraction
        column_rename (str, optional): Rename the column to match model expectations
        extract_noise (bool): Whether to extract noise features
        filter_type (str): Type of filter to use for noise extraction
        cutoff (float): Cutoff frequency for Butterworth filter
        fs (float): Sampling frequency for Butterworth filter
        poly_order (int): Polynomial order for Savitzky-Golay filter
        
    Returns:
        tuple: (is_real (bool), confidence (float), metrics (dict))
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)
        
        # Check model format
        if len(model_data) == 6:  # OCSVM format with target_class
            model, scaler, feature_names, model_column, model_type, target_class = model_data
            if model_type.lower() == "ocsvm":
                model.target_class = target_class
        elif len(model_data) == 5:  # New format with model_type
            model, scaler, feature_names, model_column, model_type = model_data
        elif len(model_data) == 4:  # Old format without model_type
            model, scaler, feature_names, model_column = model_data
            model_type = "svm"  # Default to SVM for backward compatibility
        else:
            model, scaler, feature_names = model_data
            model_column = None
            model_type = "svm"  # Default to SVM for backward compatibility
        
        logger.info(f"Model was trained on column: {model_column or 'unknown'}")
        logger.info(f"Model type: {model_type}")
        
        # Load data
        logger.info(f"Loading data from {input_file}")
        df = load_data(input_file)
        
        # Verify column exists
        if target_column not in df.columns:
            logger.error(f"Column '{target_column}' not found in the input file")
            logger.info(f"Available columns: {', '.join(df.columns)}")
            raise ModelError(f"Column '{target_column}' not found in the input file")
        
        # Determine if we need to rename the column
        output_column_prefix = column_rename if column_rename else target_column
        
        # Check if the model has noise features
        has_noise_features = any('noise_' in feature for feature in feature_names)
        
        # Extract noise from the entire dataset before windowing if needed
        noise_column = None
        if has_noise_features and extract_noise:
            logger.info(f"Extracting noise from entire dataset using {filter_type} filter")
            
            # Calculate appropriate window size and poly order for noise extraction
            noise_window_size = max(window_size//5, 3)  # Ensure minimum window size of 3
            # Ensure poly order is always less than window size
            noise_poly_order = min(poly_order, noise_window_size - 1)
            
            # Extract noise from the entire dataset
            noise_df = extract_noise_signal(
                df, 
                filter_type=filter_type,
                window_size=noise_window_size,
                cutoff=cutoff,
                fs=fs,
                poly_order=noise_poly_order,
                keep_noise_only=True,
                target_column=target_column
            )
            
            # Add noise column to original dataframe
            if not noise_df.empty and target_column in noise_df.columns:
                noise_column = f"{target_column}_noise_raw"
                df[noise_column] = noise_df[target_column]
                logger.info(f"Added noise column '{noise_column}' to dataframe")
            else:
                logger.warning("Noise extraction failed, proceeding without noise features")
        
        # Extract features
        logger.info(f"Extracting features from column '{target_column}'")
        features_df, extracted_features = extract_window_features(
            df, 
            window_size=window_size, 
            target_column=target_column,
            noise_column=noise_column
        )
        
        if features_df.empty:
            logger.error(f"No features could be extracted from the data")
            raise ModelError("No features could be extracted from the data")
            
        # Rename columns if needed to match the model's expected features
        if column_rename and target_column != output_column_prefix:
            rename_map = {}
            for col in features_df.columns:
                if col.startswith(target_column + '_'):
                    new_col = col.replace(target_column + '_', output_column_prefix + '_', 1)
                    rename_map[col] = new_col
                elif col.startswith('noise_' + target_column + '_'):
                    new_col = col.replace('noise_' + target_column + '_', 'noise_' + output_column_prefix + '_', 1)
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
        
        # Make predictions based on model type
        if model_type.lower() == "ocsvm":
            # Get the target class that was used for training
            target_class = getattr(model, 'target_class', 1)  # Default to 1 if not stored
            
            # For OCSVM, predictions are 1 for inliers and -1 for outliers
            raw_pred = model.predict(X_scaled)
            
            # Convert to binary classification format:
            # If target_class is 1 (real), then inliers (1) -> 1, outliers (-1) -> 0
            # If target_class is 0 (simulated), then inliers (1) -> 0, outliers (-1) -> 1
            if target_class == 1:
                y_pred = (raw_pred == 1).astype(int)  # 1 for inliers, 0 for outliers
            else:
                y_pred = (raw_pred == -1).astype(int)  # 1 for outliers, 0 for inliers
            
            # OCSVM doesn't provide probability estimates by default
            # We can use decision_function as a proxy for confidence
            decision_values = model.decision_function(X_scaled)
            
            # Normalize to [0, 1] range for consistency
            min_val, max_val = np.min(decision_values), np.max(decision_values)
            if max_val > min_val:
                normalized_values = (decision_values - min_val) / (max_val - min_val)
            else:
                normalized_values = np.zeros_like(decision_values)
            
            # For target_class=1, higher decision values -> higher probability of class 1
            # For target_class=0, lower decision values -> higher probability of class 1
            if target_class == 1:
                y_prob = normalized_values
            else:
                y_prob = 1 - normalized_values
                
            logger.info(f"One-Class SVM target class: {'Real' if target_class == 1 else 'Simulated'}")
        else:
            # Original prediction code for SVM and RF
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]  # Probability of being real
        
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
        confidence = abs(overall_real_ratio - 0.5) * 2 * 100  # Convert to confidence percentage
        
        logger.info(f"\nOverall Classification:")
        logger.info(f"Signal is classified as: {'REAL' if is_real else 'SIMULATED'}")
        logger.info(f"Confidence: {confidence:.1f}%")
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'window': range(total_windows),
            'probability': y_prob,
            'prediction': y_pred,
            'classification': ['Real' if p == 1 else 'Simulated' for p in y_pred]
        })
        
        results_file = os.path.join(output_dir, "window_predictions.csv")
        results_df.to_csv(results_file, index=False)
        logger.info(f"\nDetailed window predictions saved to: {results_file}")
        
        # Generate prediction distribution plot
        plot_file = os.path.join(output_dir, "prediction_distribution.svg")
        plot_prediction_distribution(y_pred, y_prob, output_file=plot_file)
        logger.info(f"Prediction distribution plot saved to: {plot_file}")
        
        # Create summary file
        summary_file = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Galileo Analysis Summary\n")
            f.write(f"======================\n\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Analyzed column: {target_column}\n")
            f.write(f"Model used: {model_path}\n")
            f.write(f"Model trained on: {model_column or 'unknown'}\n")
            f.write(f"Model type: {model_type}\n\n")
            
            f.write(f"Analysis Results:\n")
            f.write(f"Total windows analyzed: {total_windows}\n")
            f.write(f"Windows classified as real: {real_windows} ({real_percentage:.1f}%)\n")
            f.write(f"Windows classified as simulated: {simulated_windows} ({100 - real_percentage:.1f}%)\n\n")
            
            f.write(f"Overall Classification:\n")
            f.write(f"Signal is classified as: {'REAL' if is_real else 'SIMULATED'}\n")
            f.write(f"Confidence: {confidence:.1f}%\n")
        
        logger.info(f"Summary report saved to: {summary_file}")
        
        return is_real, confidence, {
            'total_windows': total_windows,
            'real_windows': real_windows,
            'simulated_windows': simulated_windows,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise ModelError(f"Analysis failed: {str(e)}")