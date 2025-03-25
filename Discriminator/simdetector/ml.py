"""
Advanced machine learning models and training for the SimDetector framework.
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tsfresh.feature_selection.relevance import calculate_relevance_table
from imblearn.over_sampling import SMOTE
import logging

logger = logging.getLogger('simdetector.ml')

# Set random seed for reproducibility
np.random.seed(42)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Default configuration parameters
DEFAULT_FILTERS = ["kalman"]  # Other filters: "no_noise", "butterworth", "moving_average", "savgol"
DEFAULT_WINDOW_SIZES = [20, 50]
DEFAULT_APPLIED_NOISES = ["gaussian+uniform", "gaussian1", "gaussian2", "uniform", "pink", "laplace", 
                         "gmm", "laplace+poisson", "poisson", "autoencoder"]

# Training speed modes
# fast = 0: train on all models with full GridSearch
# fast = 1: all models with small GridSearch
# fast = 2: only small Random Forest
FAST_MODE = 0


def _train_model(X, y, model, parameters, dataset_balancing_ratio=1, fluctuating=False):
    """
    Internal function to train a model applying GridSearchCV on a set of parameters 
    and return the scores and other info.
    
    Args:
        X (pandas.DataFrame): Features dataframe
        y (pandas.DataFrame/Series): Target dataframe/series with 'real' and optionally 'source' columns
        model: Scikit-learn model to train
        parameters (dict): Parameters for GridSearchCV
        dataset_balancing_ratio (float): Ratio for SMOTE balancing
        fluctuating (bool): Whether to test on fluctuating data
        
    Returns:
        dict: Dictionary with results and model information
    """
    # Divide the data into training and testing sets 
    real_indices = y[y['real'] == 1].index
    non_real_indices = y[y['real'] != 1].index

    X_real = X.loc[real_indices]
    y_real = y.loc[real_indices]

    X_non_real = X.loc[non_real_indices]
    y_non_real = y.loc[non_real_indices]

    if fluctuating: 
        # Extract values where "source" starts with "fluctuating_"
        fluctuating_indices = y_non_real[y_non_real['source'].str.startswith("fluctuating_")].index
        X_fluctuating = X_non_real.loc[fluctuating_indices]
        y_fluctuating_all = y_non_real.loc[fluctuating_indices]
        y_fluctuating = y_fluctuating_all['real'].copy()

        # Remove fluctuating values from non_real
        X_non_real = X_non_real.drop(fluctuating_indices)
        y_non_real = y_non_real.drop(fluctuating_indices)

    # Generate training set and testing set for real data
    try:
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=0.1, shuffle=True, random_state=42, stratify=y_real['source'])
    except ValueError:
        logger.warning("Stratified split failed for real data, falling back to random split")
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=0.1, shuffle=True, random_state=42)
    
    # Generate training set and testing set for non-real data
    split = 0.7

    try:
        X_train_non_real, X_test_non_real, y_train_non_real, y_test_non_real = train_test_split(
            X_non_real, y_non_real, test_size=split, shuffle=True, random_state=42, stratify=y_non_real['source']
        )
    except ValueError:
        source_counts = y_non_real['source'].value_counts()
        logger.warning(f"Stratify failed for non-real data. Sources with less than 3 elements: {list(source_counts[source_counts < 3].index)}")
        X_train_non_real, X_test_non_real, y_train_non_real, y_test_non_real = train_test_split(
            X_non_real, y_non_real, test_size=split, shuffle=True, random_state=42
        )

    # Combine training real and non-real data back together
    X_train = pd.concat([X_train_real, X_train_non_real])
    y_train_all = pd.concat([y_train_real, y_train_non_real])
    y_train = y_train_all['real'].copy()

    # Select the test set
    if fluctuating:
        X_test = X_fluctuating
        y_test = y_fluctuating
        y_test_all = y_fluctuating_all
    else:
        # Reduce the testing set by a factor of 0.5
        X_test_non_real, _, y_test_non_real, _ = train_test_split(
            X_test_non_real, y_test_non_real, test_size=0.5, shuffle=True, random_state=42)
        X_test = pd.concat([X_test_real, X_test_non_real])
        y_test_all = pd.concat([y_test_real, y_test_non_real])
        y_test = y_test_all['real'].copy()
    
    # Apply balancing to the training dataset
    if dataset_balancing_ratio:
        smote = SMOTE(sampling_strategy=dataset_balancing_ratio, random_state=42)
        try:
            X_train, y_train = smote.fit_resample(X_train, y_train)
        except ValueError:
            dataset_balancing_ratio = y_train.value_counts()[1] / y_train.value_counts()[0]
            logger.warning(f"Error in SMOTE. Dataset balancing ratio: {dataset_balancing_ratio}")
    else:
        dataset_balancing_ratio = y_train.value_counts()[1] / y_train.value_counts()[0]
    
    logger.info(f"Training data size: {len(X_train)} ({y_train.value_counts()}), Testing data size: {len(X_test)}, ratio: {dataset_balancing_ratio}, fluctuating: {fluctuating}")

    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, parameters, refit=True, verbose=1, cv=5, n_jobs=-1, scoring="recall")
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    logger.info(f'F1 Score after GridSearchCV: {f1}')
    logger.info(f'Accuracy after GridSearchCV: {accuracy}')
    logger.info(f'Precision after GridSearchCV: {precision}')
    logger.info(f'Recall after GridSearchCV: {recall}')

    # Calculate mean probabilities for every source
    sources = y_test_all['source'].unique()
    # Avoid messing up the indices
    y_test_all.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    y_pred_proba = pd.DataFrame(grid_search.best_estimator_.predict_proba(X_test)).reset_index(drop=True)

    mean_probas = {}
    stdev_probas = {}
    sem_probas = {}
    
    for source in sources:
        source_indices = y_test_all[y_test_all['source'] == source].index
        mean_proba = y_pred_proba.loc[source_indices].mean(axis=0).values[1]   # 1 in our case is the class "real"
        stdev_proba = y_pred_proba.loc[source_indices].std(axis=0).values[1]   # the stdev is the same for both classes
        sem_proba = stdev_proba / math.sqrt(len(source_indices)) if len(source_indices) > 0 else -1  # calculate SEM
        mean_probas[source] = mean_proba
        stdev_probas[source] = stdev_proba
        sem_probas[source] = sem_proba
        
    # Fill NaN values with -1 for mean, stdev, and sem probas
    mean_probas = {k: -1 if np.isnan(v) else v for k, v in mean_probas.items()}
    stdev_probas = {k: -1 if np.isnan(v) else v for k, v in stdev_probas.items()}
    sem_probas = {k: -1 if np.isnan(v) else v for k, v in sem_probas.items()}

    return {
        'f1': f1, 
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'model': model.__class__.__name__, 
        'best_params': grid_search.best_params_, 
        'features_size': len(X.columns), 
        "DatasetBalancing": dataset_balancing_ratio, 
        "mean_probas": mean_probas, 
        "stdev_probas": stdev_probas, 
        "sem_probas": sem_probas, 
        "split": split, 
        "fluctuating": fluctuating,
        "best_estimator": grid_search.best_estimator_
    }


def _train_models_binary(df, y_column, fast_mode=FAST_MODE, dataset_balancing_ratio=1, fluctuation=False):
    """
    Train multiple models and return the results.
    
    Args:
        df (pandas.DataFrame): Features dataframe
        y_column (str or list): Target column(s) 
        fast_mode (int): Training mode (0=full, 1=reduced, 2=minimal)
        dataset_balancing_ratio (float): Ratio for SMOTE balancing
        fluctuation (bool): Whether to test on fluctuating data
        
    Returns:
        list: List of result dictionaries
    """
    results = []

    # Extract the "real" column from the dataframe
    y = df[y_column]
    if isinstance(y_column, list):
        df = df.drop(columns=y_column)
    else:
        df = df.drop(columns=[y_column])

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier()
    if fast_mode != 0:
        rf_param_grid = {
            'n_estimators': [50, 200],
            'max_depth': [None],
        }
    else:
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    results.append(_train_model(df, y, rf_classifier, rf_param_grid, dataset_balancing_ratio=dataset_balancing_ratio, fluctuating=fluctuation))

    # Only train other models if not in fast mode 2
    if fast_mode <= 1:
        # Create a simple Neural Network classifier
        nn_classifier = MLPClassifier()
        if fast_mode == 1:
            nn_param_grid = {
                'hidden_layer_sizes': [(50,), (100,)],
                'activation': ['relu'],
            }
        else:
            nn_param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (50, 100)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd'],
                'alpha': [0.0001, 0.001, 0.01, 0.05],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000],
                'early_stopping': [True]
            }
        results.append(_train_model(df, y, nn_classifier, nn_param_grid, dataset_balancing_ratio=dataset_balancing_ratio, fluctuating=fluctuation))

        # Create a Logistic Regression classifier
        lr_classifier = LogisticRegression()
        if fast_mode == 1:
            lr_param_grid = {
                'C': [0.1, 1],
                'solver': ['newton-cg']
            }
        else:
            lr_param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['newton-cg', 'lbfgs', 'liblinear']
            }
        results.append(_train_model(df, y, lr_classifier, lr_param_grid, dataset_balancing_ratio=dataset_balancing_ratio, fluctuating=fluctuation))

        # Create an AdaBoost classifier
        ada_classifier = AdaBoostClassifier()
        if fast_mode == 1:
            ada_param_grid = {
                'n_estimators': [50, 200],
                'learning_rate': [0.5]
            }
        else:
            ada_param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1, 10]
            }
        results.append(_train_model(df, y, ada_classifier, ada_param_grid, dataset_balancing_ratio=dataset_balancing_ratio, fluctuating=fluctuation))

        # Create a Decision Tree classifier
        dt_classifier = DecisionTreeClassifier()
        if fast_mode == 1:
            dt_param_grid = {
                'criterion': ['gini'],
                'max_depth': [None],
                'min_samples_split': [2],
                'min_samples_leaf': [1]
            }
        else:
            dt_param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        results.append(_train_model(df, y, dt_classifier, dt_param_grid, dataset_balancing_ratio=dataset_balancing_ratio, fluctuating=fluctuation))

        # Create a K-Nearest Neighbors classifier
        knn_classifier = KNeighborsClassifier()

        if fast_mode == 1:
            knn_param_grid = {
                'n_neighbors': [3],
                'weights': ['uniform'],
                'algorithm': ['auto'],
                'leaf_size': [30]
            }
        else:
            knn_param_grid = {
                'n_neighbors': [2, 3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 30, 50]
            }
        results.append(_train_model(df, y, knn_classifier, knn_param_grid, dataset_balancing_ratio=dataset_balancing_ratio, fluctuating=fluctuation))

    return results


def _train_reducing_features(X, y_column, features_list, max_features=10, balRatio=1, fast_mode=FAST_MODE):
    """
    Get a features list ordered by importance, train a model increasing each time by one feature.
    
    Args:
        X (pandas.DataFrame): Features dataframe
        y_column (str or list): Target column(s)
        features_list (list): List of features ordered by importance
        max_features (int): Maximum number of features to use
        balRatio (float): Ratio for SMOTE balancing
        fast_mode (int): Training mode (0=full, 1=reduced, 2=minimal)
        
    Returns:
        list: List of result dictionaries
    """
    results = []

    # Extract the "real" column from the dataframe
    if isinstance(y_column, list):
        y = X[y_column]
        X = X.drop(columns=y_column)
    else:
        y = X[y_column]
        X = X.drop(columns=[y_column])

    for i in range(1, max_features+1):
        logger.info(f"Training with {i} features")

        features_to_train = features_list[:i]
        X_tmp = X.copy()
        X_tmp = X_tmp[features_to_train] 

        # Create a Random Forest classifier
        rf_classifier = RandomForestClassifier()
        if fast_mode >= 1:
            rf_param_grid = {
                'n_estimators': [50, 200],
                'max_depth': [None, 30],
            }            
        else:
            rf_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        tmp_result = _train_model(X_tmp, y, rf_classifier, rf_param_grid, dataset_balancing_ratio=balRatio)
        tmp_result["last_added_features"] = features_to_train[-1]
        results.append(tmp_result)

    return results


def train_binary(features_file, model_path, report_file=None, features_to_keep=None, 
                dataset_balancing_ratio=1, fast_mode=FAST_MODE, fluctuation=False):
    """
    Train multiple models on binary classification and save the best model.
    
    Args:
        features_file (str): Path to the CSV file containing extracted features
        model_path (str): Path to save the trained model
        report_file (str, optional): Path to save evaluation report
        features_to_keep (int, optional): Number of top features to keep
        dataset_balancing_ratio (float): Ratio for SMOTE balancing
        fast_mode (int): Training mode (0=full, 1=reduced, 2=minimal)
        fluctuation (bool): Whether to test on fluctuating data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load features file
        logger.info(f"Loading features from {features_file}")
        df = pd.read_csv(features_file)
        df = df.dropna()
        
        # Extract column name from filename
        file_basename = os.path.basename(features_file)
        column_name = None
        
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
            
            # Handle power types (power_real, power_apparent, power_reactive)
            if "power" in file_basename:
                if "power_real" in file_basename:
                    column_name = "power_real"
                elif "power_apparent" in file_basename:
                    column_name = "power_apparent"
                elif "power_reactive" in file_basename:
                    column_name = "power_reactive"
                elif len(parts) >= 2 and parts[1] == "power":
                    column_name = "power"  # Generic fallback
        
        logger.info(f"Detected column name: {column_name or 'unknown'}")
        
        # Apply feature selection if specified
        if features_to_keep is not None and features_to_keep != "all":
            # extract features importance
            importance_path = os.path.join(os.path.dirname(features_file), 
                                        f"feature_importance_{column_name or 'features'}.csv")
            
            if not os.path.exists(importance_path):
                logger.info(f"Calculating feature importance for {column_name or 'features'}")
                feat_relevance = calculate_relevance_table(df.drop(columns=['real', 'source']), df['real'])
                df_feat = feat_relevance[['feature', 'p_value']]
                df_feat.columns = ['Feature', 'Importance']
                
                # Save feature importance
                os.makedirs(os.path.dirname(importance_path), exist_ok=True)
                df_feat.to_csv(importance_path, index=False)
            else:
                logger.info(f"Loading feature importance from {importance_path}")
                df_feat = pd.read_csv(importance_path)

            # Sort by importance
            df_feat = df_feat.sort_values(by='Importance', ascending=False)
            features_list = df_feat['Feature'].tolist()
            
            # Add required columns
            features_list = ['real', 'source'] + features_list[:features_to_keep]
            
            # Keep only selected features
            df = df[features_list]
            logger.info(f"Using top {features_to_keep} features")
        
        # Drop rows where 'source' contains the word "Gan"
        if 'source' in df.columns:
            df = df[~df['source'].str.contains("Gan", case=False, na=False)]
            logger.info("Removed GAN-generated samples")
            
            # Drop or keep rows with fluctuating data
            if not fluctuation and 'source' in df.columns:
                df = df[~df['source'].str.contains("fluctuating_", case=False, na=False)]
                logger.info("Removed fluctuating samples")
        
        df = df.reset_index(drop=True)
        
        # Train multiple models
        logger.info(f"Training models with balancing ratio: {dataset_balancing_ratio}")
        results = _train_models_binary(df, ['real', 'source'], 
                                     fast_mode=fast_mode,
                                     dataset_balancing_ratio=dataset_balancing_ratio,
                                     fluctuation=fluctuation)
        
        # Find the best model based on F1 score
        best_result = max(results, key=lambda x: x['f1'])
        
        logger.info(f"Best model: {best_result['model']} with F1 score: {best_result['f1']}")
        logger.info(f"Parameters: {best_result['best_params']}")
        
        # Save the best model
        import joblib
        
        # Get the best estimator
        best_estimator = best_result['best_estimator']
        
        # Create a standardized object for saving, compatible with SimDetector's format
        # Need to include the model type so the predict functions can handle it
        model_type = best_result['model'].lower()
        if model_type == 'randomforestclassifier':
            model_type = 'rf'
        elif model_type == 'logisticregression':
            model_type = 'lr'
        elif model_type == 'mlpclassifier':
            model_type = 'mlp'
        elif model_type == 'adaboostclassifier':
            model_type = 'adaboost'
        elif model_type == 'decisiontreeclassifier':
            model_type = 'dt'
        elif model_type == 'kneighborsclassifier':
            model_type = 'knn'
        
        # Create a mock StandardScaler (SimDetector expects this format)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        # Get feature names
        feature_names = df.drop(columns=['real', 'source']).columns.tolist()
        
        # Save in SimDetector's format
        joblib.dump((best_estimator, scaler, feature_names, column_name, model_type), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save detailed results if report_file is provided
        if report_file:
            for result in results:
                result["noise"] = "kalman"  # Default filter
                result["filename"] = column_name or "unknown"
                # Remove window_size as requested
                result["num_rows"] = len(df)
                # Set the features used to the actual feature names
                result["features_used"] = str(feature_names)
                
                # Clean up the best_params for CSV saving
                if 'best_params' in result:
                    result['best_params'] = str(result['best_params'])
                
                # Remove the estimator object before saving to CSV
                if 'best_estimator' in result:
                    del result['best_estimator']
            
            # Create detailed report
            results_df = pd.DataFrame(results)
            results_df.to_csv(report_file, index=False)
            logger.info(f"Detailed report saved to {report_file}")
            
        return True
            
    except Exception as e:
        logger.error(f"Error training models with feature reduction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False 

def train_reducing_features(features_file, model_path, report_file=None, max_features=10, balRatio=1, fast_mode=FAST_MODE):
    """
    Train models with progressively more features to analyze feature importance.
    
    Args:
        features_file (str): Path to the CSV file containing extracted features
        model_path (str): Path to save the trained model
        report_file (str, optional): Path to save evaluation report
        max_features (int): Maximum number of features to use
        balRatio (float): Ratio for SMOTE balancing
        fast_mode (int): Training mode (0=full, 1=reduced, 2=minimal)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load features file
        logger.info(f"Loading features from {features_file}")
        df = pd.read_csv(features_file)
        df = df.dropna()
        
        # Extract column name from filename
        file_basename = os.path.basename(features_file)
        column_name = None
        
        # Try to extract column name from filename pattern
        if "_" in file_basename:
            parts = file_basename.split("_")
            
            if len(parts) >= 2:
                potential_column = parts[1]
                if potential_column.startswith("V") or potential_column.startswith("C"):
                    column_name = potential_column
                elif potential_column == "frequency":
                    column_name = potential_column
            
            if "power" in file_basename:
                if "power_real" in file_basename:
                    column_name = "power_real"
                elif "power_apparent" in file_basename:
                    column_name = "power_apparent"
                elif "power_reactive" in file_basename:
                    column_name = "power_reactive"
                elif len(parts) >= 2 and parts[1] == "power":
                    column_name = "power"
        
        # Extract features importance
        importance_path = os.path.join(os.path.dirname(features_file), 
                                    f"feature_importance_{column_name or 'features'}.csv")
        
        if not os.path.exists(importance_path):
            logger.info(f"Calculating feature importance for {column_name or 'features'}")
            feat_relevance = calculate_relevance_table(df.drop(columns=['real', 'source']), df['real'])
            df_feat = feat_relevance[['feature', 'p_value']]
            df_feat.columns = ['Feature', 'Importance']
            
            # Save feature importance
            os.makedirs(os.path.dirname(importance_path), exist_ok=True)
            df_feat.to_csv(importance_path, index=False)
        else:
            logger.info(f"Loading feature importance from {importance_path}")
            df_feat = pd.read_csv(importance_path)

        # Sort by importance
        df_feat = df_feat.sort_values(by='Importance', ascending=False)
        features_list = df_feat['Feature'].tolist()
        
        # Drop rows where 'source' contains the word "Gan" or "fluctuating_"
        if 'source' in df.columns:
            df = df[~df['source'].str.contains("Gan", case=False, na=False)]
            df = df[~df['source'].str.contains("fluctuating_", case=False, na=False)]
            
        df = df.reset_index(drop=True)
        
        # Train with progressively more features
        logger.info(f"Training with feature reduction up to {max_features} features")
        results = _train_reducing_features(df, ['real', 'source'], features_list, 
                                          max_features=max_features, 
                                          balRatio=balRatio,
                                          fast_mode=fast_mode)
        
        # Find the best model based on F1 score
        best_result = max(results, key=lambda x: x['f1'])
        
        logger.info(f"Best model has {len(best_result['best_params'])} features")
        logger.info(f"F1 score: {best_result['f1']}")
        
        # Save the best model
        import joblib
        
        # Get the best estimator
        best_estimator = best_result['best_estimator']
        
        # Create a standardized object for saving, compatible with SimDetector's format
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        # Get feature names (only the ones used for training)
        feature_count = best_result.get('features_size', max_features)
        feature_names = features_list[:feature_count]
        
        # Save in SimDetector's format
        joblib.dump((best_estimator, scaler, feature_names, column_name, 'rf'), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save detailed results if report_file is provided
        if report_file:
            for i, result in enumerate(results):
                result["noise"] = "kalman"  # Default filter
                result["filename"] = column_name or "unknown"
                # Remove window_size as requested
                result["num_rows"] = len(df)
                
                # Set the features used - properly tracking which features were used in this iteration
                current_features = features_list[:i+1]  # First i+1 features
                result["features_used"] = str(current_features)
                
                # Clean up the best_params for CSV saving
                if 'best_params' in result:
                    result['best_params'] = str(result['best_params'])
                
                # Remove the estimator object before saving to CSV
                if 'best_estimator' in result:
                    del result['best_estimator']
            
            # Create detailed report
            results_df = pd.DataFrame(results)
            results_df.to_csv(report_file, index=False)
            logger.info(f"Feature reduction report saved to {report_file}")
            
        return True
            
    except Exception as e:
        logger.error(f"Error training models with feature reduction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False