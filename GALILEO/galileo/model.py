"""
Model training, evaluation, and prediction for the Galileo framework.
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
from galileo.data import load_data
from galileo.features import extract_window_features
from galileo.core import ModelError, validate_file_path
from galileo.visualization import plot_feature_importance

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


def calculate_feature_importance(model, X, y, feature_names):
    """
    Calculate feature importance using permutation importance.
    
    Args:
        model: Trained model with predict method
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        
    Returns:
        pandas.DataFrame: DataFrame with feature importance scores
    """
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importance_scores = result.importances_mean
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df


def train_model_from_features(X_train, y_train, feature_names, model_path=None, metric_name=None):
    """
    Train a model on the provided features and save it.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training labels
        feature_names (list): List of feature names
        model_path (str, optional): Path to save the model
        metric_name (str, optional): Name of the metric being analyzed
        
    Returns:
        tuple: (svm, scaler, feature_names)
    """
    # Handle NaN values
    X_train = X_train.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Train SVM
    logger.info("Training SVM classifier...")
    svm = SVC(kernel='rbf', C=1.0, probability=True)
    svm.fit(X_scaled, y_train)
    
    # Calculate feature importance
    importance_df = calculate_feature_importance(svm, X_scaled, y_train, feature_names)
    
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
        # Store metric_name with the model for later reference
        joblib.dump((svm, scaler, feature_names, metric_name), model_path)
        logger.info(f"Model saved to {model_path}")
        
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
                       skip_evaluation=False, report_file="report.csv", noise_only=False):
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
        
        # Train the model
        svm, scaler, _ = train_model_from_features(X_train, y_train, feature_names, model_path, metric_name=column_name)
        
        # Evaluate the model if not skipped
        if not skip_evaluation:
            logger.info("Evaluating model on test set:")
            evaluate_model(X_test, y_test, svm, scaler, column_name, report_file, has_noise)
        
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise ModelError(f"Failed to train model: {str(e)}")
