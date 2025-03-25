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
np.random.seed(42)
warnings.filterwarnings("ignore", category=RuntimeWarning)

filters = ["kalman"]  # Other filters: "no_noise", "butterworth", "moving_average", "savgol"
window_sizes = [5,10,15,20,30,40,50,60,70,80,90,100]
applied_noises = ["gaussian+uniform", "gaussian1", "gaussian2", "uniform", "pink", "laplace", "gmm", "laplace+poisson", "poisson", "autoencoder"]

# fast = 0: train on all the things
# fast = 1: all models small GridSerach
# fast = 2: only small RF
fast = 0


def _train_model(X, y, model, parameters, dataset_balancing_ratio=1, fluctuating=False):
    """Internal function to train a model applying GridSearchCV on a set of parameters and return the scores and other infos."""

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
        print(f"Stratify failed. Sources with less than 3 elements: {list(source_counts[source_counts < 3].index)}")
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
            print(f"Error in SMOTE. Dataset balancing ratio: {dataset_balancing_ratio}")
    else:
        dataset_balancing_ratio = y_train.value_counts()[1] / y_train.value_counts()[0]
    
    print(f"Training data size: {len(X_train)} ({y_train.value_counts()}), Testing data size: {len(X_test)}, ratio: {dataset_balancing_ratio}, fluctuating: {fluctuating}")

    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, parameters, refit=True, verbose=1, cv=5, n_jobs=-1, scoring="recall")
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f'F1 Score after GridSearchCV: {f1}')
    print(f'Accuracy after GridSearchCV: {accuracy}')
    print(f'Precision after GridSearchCV: {precision}')
    print(f'Recall after GridSearchCV: {recall}')


    # Calculate mean probabilities for every source
    sources = y_test_all['source'].unique()
    # avoid messing up the indices
    y_test_all.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    y_pred_proba = pd.DataFrame(grid_search.best_estimator_.predict_proba(X_test)).reset_index(drop=True)

    mean_probas = {}
    stdev_probas = {}
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

    return {'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'model': model.__class__.__name__, 'best_params': grid_search.best_params_, 'features_size': len(X.columns), "DatasetBalancin": dataset_balancing_ratio, "mean_probas": mean_probas, "stdev_probas": stdev_probas, "sem_probas": sem_probas , "split": split, "fluctuating": fluctuating}


def _train_models_binary(df, y_column, dataset_balancing_ratio=1, fluctuation=False):
    """Train multiple models and return the results"""

    results = []

    # Extract the "real" column from the dataframe
    y = df[y_column]
    if isinstance(y_column, list):
        df = df.drop(columns=y_column)
    else:
        df = df.drop(columns=[y_column])

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier()
    if fast != 0:
        rf_param_grid = {
            'n_estimators': [50, 200],
            'max_depth': [None],
            #'min_samples_split': [2, 10],
            #'min_samples_leaf': [1, 4]
        }
    else:
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    results += [_train_model(df, y, rf_classifier, rf_param_grid, dataset_balancing_ratio=dataset_balancing_ratio, fluctuating=fluctuation)]

    if fast <= 1:
        # Create a simple Neural Network classifier
        nn_classifier = MLPClassifier()
        if fast == 1:
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
        results += [_train_model(df, y, nn_classifier, nn_param_grid, dataset_balancing_ratio=dataset_balancing_ratio, fluctuating=fluctuation)]

        # Create a Logistic Regression classifier
        lr_classifier = LogisticRegression()
        if fast == 1:
            lr_param_grid = {
                'C': [0.1, 1],
                'solver': ['newton-cg']
            }
        else:
            lr_param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['newton-cg', 'lbfgs', 'liblinear']
            }
        results += [_train_model(df, y, lr_classifier, lr_param_grid, dataset_balancing_ratio=dataset_balancing_ratio, fluctuating=fluctuation)]

        # Create an AdaBoost classifier
        ada_classifier = AdaBoostClassifier()
        if fast == 1:
            ada_param_grid = {
                'n_estimators': [50, 200],
                'learning_rate': [0.5]
            }
        else:
            ada_param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1, 10]
            }
        results += [_train_model(df, y, ada_classifier, ada_param_grid, dataset_balancing_ratio=dataset_balancing_ratio, fluctuating=fluctuation)]

        # Create a Decision Tree classifier
        dt_classifier = DecisionTreeClassifier()
        if fast == 1:
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
        results += [_train_model(df, y, dt_classifier, dt_param_grid, dataset_balancing_ratio=dataset_balancing_ratio, fluctuating=fluctuation)]

        # Create a K-Nearest Neighbors classifier
        knn_classifier = KNeighborsClassifier()

        if fast == 1:
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
        results += [_train_model(df, y, knn_classifier, knn_param_grid, dataset_balancing_ratio=dataset_balancing_ratio, fluctuating=fluctuation)]

    return results


def _train_reducing_features(X, y_column, features_list, max_features=10, balRatio=1):
    """Get a features list ordered by importance, train a model increasing each time by one feature."""

    results = []

    # Extract the "real" column from the dataframe
    if isinstance(y_column, list):
        y = X[y_column]
        X = X.drop(columns=y_column)
    else:
        y = X[y_column]
        X = X.drop(columns=[y_column])

    features_to_train = features_list.copy()
    for i in range(1, max_features+1):
        print(f"Training with {i} features")

        features_to_train = features_list[:i]
        X_tmp = X.copy()
        X_tmp = X_tmp[features_to_train] 

        # Create a Random Forest classifier
        rf_classifier = RandomForestClassifier()
        if fast >= 1:
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
        results += [tmp_result]

    return results


def train_binary(base_path, features_to_keep=None, dataset_balancing_ratio=1, windows=[20,50], fluctuation=False):

    for filter in filters:
        results = []
        for window_size in windows: 

            file_path = os.path.join(base_path, f"window{window_size}", filter)
            measurements_names = [f.split("combined_")[1].split("_features.csv")[0]  for f in os.listdir(file_path) if f.endswith('.csv') and f.startswith("combined_")]
            
            for f in measurements_names:
                
                print(f"------------\nLoading {f} data")

                # open combined file
                path = os.path.join(file_path, f"combined_{f}_features.csv")
                df = pd.read_csv(path)
                df = df.dropna()

                if features_to_keep is not None and features_to_keep != "all":
                    # extract features importance
                    path = os.path.join(file_path, f"feature_importance_{f}.csv")
                    if not os.path.exists(path):
                        feat_relevance = calculate_relevance_table(df.drop(columns=['real', 'source']), df['real'])
                        df_feat = feat_relevance[['feature', 'p_value']]
                        df_feat.columns = ['Feature', 'Importance']
                    else:
                        df_feat = pd.read_csv(path)

                    # sort by importance
                    df_feat = df_feat.sort_values(by='Importance', ascending=False)
                    features_list = df_feat['Feature'].tolist()
                    # need also "real" column
                    features_list = ['real','source'] + features_list[:features_to_keep]
                    # keep only selected features
                    df = df[features_list]
                
                # Drop rows where 'source' contains the word "Gan"
                df = df[~df['source'].str.contains("Gan", case=False, na=False)]

                # Drop rows where 'source' contains the word "fluctuating_"
                if not fluctuation: 
                    df = df[~df['source'].str.contains("fluctuating_", case=False, na=False)]

                df = df.reset_index(drop=True)

                temp_results = _train_models_binary(df, ['real','source'], dataset_balancing_ratio=dataset_balancing_ratio, fluctuation=fluctuation)

                for result in temp_results:
                    result["noise"] = filter
                    result["filename"] = f
                    result["window_size"] = window_size
                    result["num_rows"] = len(df)
                
                results += temp_results

                # Convert results to a DataFrame and save to CSV
                results_df = pd.DataFrame(results)
                results_csv_path = os.path.join(base_path, f'final_results_filtered_{filter}_{"all" if features_to_keep is None else features_to_keep}_balRatio{dataset_balancing_ratio}{"_fluctuation" if fluctuation else ""}.csv')
                results_df.to_csv(results_csv_path, index=False)


def train_reducing_features(base_path, max_features=10, balRatio=1, windows=[20,50]):
    """Train models with a reduced number of features"""

    for noise in ["kalman"]:
        results = []
        for window_size in windows:
            # get folder path
            file_path = os.path.join(base_path, f"window{window_size}", noise)

            # Get all CSV files in the directory (really bad way but it works)
            measurements_names = [f.split("combined_")[1].split("_features.csv")[0]  for f in os.listdir(file_path) if f.endswith('.csv') and f.startswith("combined_")]
            
            for f in ["C1", "V1", "frequency", "allcolumns"]: #measurements_names:
                print("Loading ", f)

                # open combined file
                path = os.path.join(file_path, f"combined_{f}_features.csv")
                df = pd.read_csv(path)
                df = df.dropna()

                # extract features importance
                path = os.path.join(file_path, f"feature_importance_{f}.csv")
                if not os.path.exists(path):
                    feat_relevance = calculate_relevance_table(df.drop(columns=['real', 'source']), df['real'])
                    df_feat = feat_relevance[['feature', 'p_value']]
                    df_feat.columns = ['Feature', 'Importance']
                else:
                    df_feat = pd.read_csv(path)

                # sort by importance
                df_feat = df_feat.sort_values(by='Importance', ascending=False)
                features_list = df_feat['Feature'].tolist()

                # Drop rows where 'source' contains the word "Gan"
                df = df[~df['source'].str.contains("Gan", case=False, na=False)]
                # Drop rows where 'source' contains the word "fluctuating_"
                df = df[~df['source'].str.contains("fluctuating_", case=False, na=False)]
                df = df.reset_index(drop=True)


                temp_results = _train_reducing_features(df, ['real','source'], features_list, 
                                                        max_features=max_features, balRatio=balRatio)

                for result in temp_results:
                    result["noise"] = noise
                    result["filename"] = f
                    result["window_size"] = window_size
                    result["num_rows"] = len(df)
                    
                results += temp_results

        # Convert results to a DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(base_path, f'final_results_{noise}_{max_features}feature_selection_balRatio{balRatio}.csv')
        results_df.to_csv(results_csv_path, index=False)


if __name__ == '__main__':

    # Base path containing the dataset, and place where results will be saved.
    base_path = 'dataset-newFilter/'

    # Train the models
    train_binary(base_path, features_to_keep=11, dataset_balancing_ratio=0.9, windows = [20])

