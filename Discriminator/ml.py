import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
import pickle



def _train_model(X, y, model, parameters, save=False):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, parameters, refit=True, verbose=1, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.best_estimator_.predict(X_test)

    if save:
        model_filename = f"{model.__class__.__name__}_best_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)
        print(f"Model saved to {model_filename}")

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f'F1 Score after GridSearchCV: {f1}')
    print(f'Accuracy after GridSearchCV: {accuracy}')
    print(f'Precision after GridSearchCV: {precision}')
    print(f'Recall after GridSearchCV: {recall}')
    return {'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'model': model.__class__.__name__, 'best_params': grid_search.best_params_, 'features_size': len(X.columns)}


def _train_models_binary(df, y_column):

    results = []

    # Extract the "real" column from the dataframe
    y = df[y_column]
    df = df.drop(columns=[y_column])

    # Create a simple Neural Network classifier
    nn_classifier = MLPClassifier(max_iter=1000)
    nn_param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['adaptive'],
    }
    results += [_train_model(df, y, nn_classifier, nn_param_grid)]

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier()
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    results += [_train_model(df, y, rf_classifier, rf_param_grid, save=True)]


    # Create a Logistic Regression classifier
    lr_classifier = LogisticRegression()
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    }
    results += [_train_model(df, y, lr_classifier, lr_param_grid)]

    # Create an AdaBoost classifier
    ada_classifier = AdaBoostClassifier()
    ada_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1, 10]
    }
    results += [_train_model(df, y, ada_classifier, ada_param_grid)]

    # Create a Decision Tree classifier
    dt_classifier = DecisionTreeClassifier()
    dt_param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    results += [_train_model(df, y, dt_classifier, dt_param_grid)]

    return results


def _train_models_oneClass(df, y_column, label_to_train_on = 1, contamination = 0.5):

    results = []

    # train on real data
    the_other_label = label_to_train_on ^ 1  # test on simulated data
    train_data_real = df[df[y_column] == label_to_train_on]
    test_data_simulated = df[df[y_column] == the_other_label]

    # Split real data into training and testing sets
    X_train, X_test_real = train_test_split(train_data_real, test_size=0.2, shuffle=True, random_state=42)
    _, X_test_simulated = train_test_split(test_data_simulated, test_size=0.2, shuffle=True, random_state=42) 
    X_test = pd.concat([X_test_real, X_test_simulated])

    y_train = [label_to_train_on] * len(X_train)
    y_test = [label_to_train_on] * len(X_test_real) + [the_other_label] * len(X_test_simulated)
    y_test = [-1 if x == 0 else 1 for x in y_test]

    # Drop the label column from the training and testing data
    X_train = X_train.drop(columns=[y_column])
    X_test = X_test.drop(columns=[y_column])
    
    model = OneClassSVM(kernel='rbf', gamma=0.1, nu=contamination)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy and F1 score
    f1 = f1_score(y_test, y_pred, average='macro')#, pos_label=-1)
    print(f'F1 Score for One-Class SVM: {f1}')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    results += [{'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'model': model.__class__.__name__,
                 'features_size': len(X_train.columns)}]


    # Use Local Outlier Factor for novelty detection
    model = LocalOutlierFactor(novelty=True, contamination=contamination)
    model.fit(X_train.values, y=None)
    y_pred = model.predict(X_test.values)
    
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'F1 Score for Local Outlier Factor: {f1}')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    results += [{'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'model': model.__class__.__name__,
                 'features_size': len(X_train.columns)}]
    
    # Use Gaussian Mixture Model for novelty detection
    model = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == 0, -1, 1)

    # Calculate the accuracy and F1 score
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'F1 Score for Gaussian Mixture Model: {f1}')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    results += [{'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'model': 'GaussianMixture',
                 'features_size': len(X_train.columns)}]
    

    # Use Isolation Forest for novelty detection
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_train.values)
    y_pred = model.predict(X_test.values)
    y_pred = np.where(y_pred == 1, 1, -1)

    # Calculate the accuracy and F1 score
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'F1 Score for Isolation Forest: {f1}')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    results += [{'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'model': 'IsolationForest',
                 'features_size': len(X_train.columns)}]
    
    return results


def _train_reducing_features(X, y_column, features_list, max_features=10):
    """Get a features list ordered by importance, train a model with all features and then reduce the number of features"""

    results = []

    # Extract the "real" column from the dataframe
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
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        tmp_result = _train_model(X_tmp, y, rf_classifier, rf_param_grid)
        tmp_result["last_added_features"] = features_to_train[-1]
        results += [tmp_result]

    return results


def train_binary(base_path, features_to_keep=None):

    for noise in ["no_noise", "butterworth", "moving_average", "savgol"]:
        # Load the CSV file
        results = []
        
        for window_size in [5,10,15,20,30,40,50]: #

            if noise == "savgol" and window_size <= 15: 
                # not enough data in these cases, skipping
                continue

            file_path = os.path.join(base_path, f"window{window_size}", noise)

            # Get all CSV files in the directory
            measurements_names = [f.split("combined_")[1].split("_features.csv")[0]  for f in os.listdir(file_path) if f.endswith('.csv') and f.startswith("combined_")]
            
            for f in measurements_names:
                
                print(f"Loading {f} data")

                # open combined file
                path = os.path.join(file_path, f"combined_{f}_features.csv")
                df = pd.read_csv(path)
                df = df.dropna()

                if features_to_keep:
                    # extract features importance
                    path = os.path.join(file_path, f"feature_importance_{f}.csv")
                    df_feat = pd.read_csv(path)
                    # sort by importance
                    df_feat = df_feat.sort_values(by='Importance', ascending=False)
                    features_list = df_feat['Feature'].tolist()
                    # need also "real" column
                    features_list = ['real'] + features_list
                    # keep only selected features
                    df = df[features_list[:features_to_keep]]

                temp_results = _train_models_binary(df, 'real')

                for result in temp_results:
                    result["noise"] = noise
                    result["filename"] = f
                    result["window_size"] = window_size
                    result["num_rows"] = len(df)
                
                results += temp_results

                # Convert results to a DataFrame and save to CSV
                results_df = pd.DataFrame(results)
                results_csv_path = os.path.join(base_path, f'final_results_filtered_{noise}.csv')
                results_df.to_csv(results_csv_path, index=False)


def train_oneClass(base_path, window_size=50, contamination=0.5, features_to_keep=None):
    """Train one class models"""
    
    for noise in ["no_noise", "butterworth", "moving_average", "savgol"]:
        # Load the CSV file
        results = []
        
        for window_size in [window_size]: #[5, 10, 15, 20, 30, 40, 50]:

            file_path = os.path.join(base_path, f"window{window_size}", noise)

            # Get all CSV files in the directory
            measurements_names = [f.split("combined_")[1].split("_features.csv")[0]  for f in os.listdir(file_path) if f.endswith('.csv') and f.startswith("combined_")]
            
            for f in measurements_names:
                
                print(f"Loading {f} data")

                # open combined file
                path = os.path.join(file_path, f"combined_{f}_features.csv")
                df = pd.read_csv(path)
                df = df.dropna()

                if features_to_keep:
                    # extract features importance
                    path = os.path.join(file_path, f"feature_importance_{f}.csv")
                    df_feat = pd.read_csv(path)
                    # sort by importance
                    df_feat = df_feat.sort_values(by='Importance', ascending=False)
                    features_list = df_feat['Feature'].tolist()
                    # need also "real" column
                    features_list = ['real'] + features_list
                    # keep only selected features
                    df = df[features_list[:features_to_keep]]

                temp_results = _train_models_oneClass(df, 'real', contamination=contamination)

                for result in temp_results:
                    result["noise"] = noise
                    result["filename"] = f
                    result["window_size"] = window_size
                    result["num_rows"] = len(df)
                
                results += temp_results

                # Convert results to a DataFrame and save to CSV
                results_df = pd.DataFrame(results)
                results_csv_path = os.path.join(base_path, f'final_results_filtered_OneClass_{noise}.csv')
                results_df.to_csv(results_csv_path, index=False)


def train_reducing_features(base_path, max_features=10):

    for noise in ["no_noise", "butterworth", "moving_average", "savgol"]:
        results = []
        for window_size in [50]:
            # get folder path
            file_path = os.path.join(base_path, f"window{window_size}", noise)

            # Get all CSV files in the directory (really bad way but it works)
            measurements_names = [f.split("combined_")[1].split("_features.csv")[0]  for f in os.listdir(file_path) if f.endswith('.csv') and f.startswith("combined_")]
            
            for f in measurements_names:
                # extract features importance
                path = os.path.join(file_path, f"feature_importance_{f}.csv")
                df = pd.read_csv(path)
                # sort by importance
                df = df.sort_values(by='Importance', ascending=False)
                features_list = df['Feature'].tolist()

                # open comnbined file
                path = os.path.join(file_path, f"combined_{f}_features.csv")
                df = pd.read_csv(path)
                df = df.dropna()

                temp_results = _train_reducing_features(df, 'real', features_list, 
                                                        max_features=max_features)

                for result in temp_results:
                    result["noise"] = noise
                    result["filename"] = f
                    result["window_size"] = window_size
                    result["num_rows"] = len(df)
                    
                results += temp_results

        # Convert results to a DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(base_path, f'final_results_{noise}_{max_features}feature_selection.csv')
        results_df.to_csv(results_csv_path, index=False)



if __name__ == '__main__':

    base_path = 'dataset-features/'

    train_binary(base_path, features_to_keep=10)

    train_oneClass(base_path, contamination=0.3, features_to_keep=10)

    train_reducing_features(base_path, max_features=10)
