from matplotlib.axis import Axis
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 17})
marker_size = 10
line_width = 1.5
column_order = ["I1", "I2", "I3", "V1", "V2", "V3", "frequency", "power_apparent", "power_reactive", "power_real", "allvalues"]


def plot_delta_accuracies_on_each_source(results_path, window = 20, filename="V1", model="RandomForestClassifier", features_number=11, balRatio=0.9):

    # load normal data
    df = pd.read_csv(results_path)
    df['filename'] = df['filename'].replace('C1', 'I1').replace('C2', 'I2').replace('C3', 'I3')
    df = df[df['features_size'] == features_number]
    df = df[df['DatasetBalancin'] == balRatio]
    df = df[df['model'] == model]
    df = df[df['window_size'] == window]
    df['filename'] = df['filename'].str.replace('allcolumns', 'allvalues')

    mean_probas = df['mean_probas'].apply(eval).apply(pd.Series)
    sem_probas = df['sem_probas'].apply(eval).apply(pd.Series)
    mean_probas.index = df['filename']
    sem_probas.index = df['filename']

    # Just to be sure
    mean_probas = mean_probas.replace(-1, 0)
    sem_probas = sem_probas.replace(-1, 0)

    # Drop columns named EPIC{i}.csv
    for i in range(9):
        column_name = f'EPIC{i}.csv'
        if column_name in mean_probas.columns:
            mean_probas.drop(columns=[column_name], inplace=True)
            sem_probas.drop(columns=[column_name], inplace=True)
    
    mean_probas.columns = mean_probas.columns.str[:-4]
    sem_probas.columns = sem_probas.columns.str[:-4]

    mean_probas_selected = mean_probas.loc[filename].sort_values(ascending=False)
    sem_probas_selected = sem_probas.loc[filename].reindex(mean_probas_selected.index)

    # Load dynamic data
    file_path_dynamic = results_path[:-4] + "_dynamic.csv"
    df_dynamic = pd.read_csv(file_path_dynamic)
    df_dynamic['filename'] = df_dynamic['filename'].replace('C1', 'I1').replace('C2', 'I2').replace('C3', 'I3')
    df_dynamic = df_dynamic[df_dynamic['model'] == model]
    df_dynamic = df_dynamic[df_dynamic['window_size'] == window]
    df_dynamic = df_dynamic[df_dynamic['features_size'] == features_number]
    df_dynamic = df_dynamic[df_dynamic['DatasetBalancin'] == balRatio]
    df_dynamic['filename'] = df_dynamic['filename'].str.replace('allcolumns', 'allvalues')

    mean_probas_dynamic = df_dynamic['mean_probas'].apply(eval).apply(pd.Series)
    sem_probas_dynamic = df_dynamic['sem_probas'].apply(eval).apply(pd.Series)
    mean_probas_dynamic.index = df_dynamic['filename']
    sem_probas_dynamic.index = df_dynamic['filename']

    # for now replacing -1 (which means NaN) with 0 so we throw them away, but maybe should be handled differently TODO
    mean_probas_dynamic = mean_probas_dynamic.replace(-1, 0)
    sem_probas_dynamic = sem_probas_dynamic.replace(-1, 0)

    # Drop columns named EPIC{i}.csv
    for i in range(9):
        column_name = f'EPIC{i}.csv'
        if column_name in mean_probas_dynamic.columns:
            mean_probas_dynamic.drop(columns=[column_name], inplace=True)
            sem_probas_dynamic.drop(columns=[column_name], inplace=True)
    
    # Compute delta between normal and dynamic data
    # Adjust the indices of mean_probas_fluc to match mean_probas
    mean_probas_dynamic.columns = mean_probas_dynamic.columns.str.replace('fluctuating_', '').str.replace(".csv", "")
    sem_probas_dynamic.columns = sem_probas_dynamic.columns.str.replace('fluctuating_', '').str.replace(".csv", "")

    # Compute delta between normal and dynamic data
    mean_probas_dynamic_selected = mean_probas_dynamic.loc[filename].reindex(mean_probas_selected.index)
    sem_probas_dynamic_selected = sem_probas_dynamic.loc[filename].reindex(mean_probas_selected.index)

    # Select only the first X elements
    how_many_to_plot = 13
    mean_probas_selected = mean_probas_selected.iloc[:how_many_to_plot]
    sem_probas_selected = sem_probas_selected.iloc[:how_many_to_plot]
    mean_probas_dynamic_selected = mean_probas_dynamic_selected.iloc[:how_many_to_plot]
    sem_probas_dynamic_selected = sem_probas_dynamic_selected.iloc[:how_many_to_plot]

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'mean_probas': mean_probas_selected.values,
        'sem_probas': sem_probas_selected.values,
        'mean_probas_fluc': mean_probas_dynamic_selected.values,
        'sem_probas_fluc': sem_probas_dynamic_selected.values
    }, index=mean_probas_selected.index)

    # ---- printing some stuff
    max_mean_probas = mean_probas_selected.max()
    max_mean_probas_dynamic = mean_probas_dynamic_selected.loc[mean_probas_selected.idxmax()]
    difference = max_mean_probas_dynamic - max_mean_probas
    print(f"Max mean probability with {mean_probas_selected.idxmax()}: {max_mean_probas:.3f}. Dynamic: {max_mean_probas_dynamic:.3f}. Delta: {difference:.3f}")
    #-----------

    # Plot the mean probabilities for each column with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(plot_data))

    # Plot normal bars
    ax.bar(index, plot_data['mean_probas'], bar_width, yerr=plot_data['sem_probas'], label='Normal', capsize=4, zorder=5)

    # Plot dynamic bars
    ax.bar([i + bar_width for i in index], plot_data['mean_probas_fluc'], bar_width, yerr=plot_data['sem_probas_fluc'], label='Dynamic', capsize=4, zorder=5)

    plt.xlabel('Source')
    plt.ylabel('Mean Probability')

    plt.ylim(0, 1.01)
    short_index = [col.replace("Mosaik+", "M+").replace("Panda+", "P+") for col in plot_data.index]
    plt.xticks([i + bar_width / 2 for i in index], short_index, rotation=45, ha='right')
    plt.legend(fontsize=plt.rcParams['font.size'] - 1)
    ax.grid(True, which='both', zorder=0, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"./Results/mean_probabilities_{filename}.pdf")
    plt.show()


def plot_binary_models(results_path, window = 50, balRatio=0.9, features_number=11): 
    
    df = pd.read_csv(results_path)
    df['filename'] = df['filename'].replace('C1', 'I1').replace('C2', 'I2').replace('C3', 'I3')
    df = df[df['features_size'] == features_number]
    df = df[df['DatasetBalancin'] == balRatio]
    df = df[df['window_size'] == window]
    rename_dict = {
            'LogisticRegression': 'LR',
            'RandomForestClassifier': 'RF',
            'DecisionTreeClassifier': 'DT',
            'AdaBoostClassifier': 'AB',
            'MLPClassifier': 'NN',
            'KNeighborsClassifier': 'KNN'
        }
    
    df['model'] = df['model'].map(rename_dict)
    df['filename'] = df['filename'].str.replace('allcolumns', 'allvalues')

    for metric in ['recall']:
        print(f"Metric: {metric}")

        df_pivot = df.pivot(index='filename', columns='model', values=metric).reset_index().melt(id_vars='filename', var_name='model', value_name=metric)

        # Ensure the filename column follows the specified column order
        df_pivot['filename'] = pd.Categorical(df_pivot['filename'], categories=column_order, ordered=True)
        df_pivot = df_pivot.sort_values('filename')

        # Create a bar plot with seaborn
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_pivot, x='filename', y=metric, hue='model', ax=ax, zorder=5)
        ax.grid(True, which='both', zorder=0, alpha=0.5)

        ax.legend(title='Model', loc='lower center', ncol=6, fontsize=plt.rcParams['font.size'] - 1)
        plt.xlabel('Value')
        plt.ylabel(metric)
        plt.ylim(0, 1.01)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Display the plot
        plt.tight_layout()
        plt.savefig(f"./Results/binary_{metric}.pdf")
        plt.show()


def plot_window(results_path, model="RandomForestClassifier", balRatio=0.9, features=11):

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(results_path)
    df['filename'] = df['filename'].replace('C1', 'I1').replace('C2', 'I2').replace('C3', 'I3')
    df = df[df['model'] == model]
    df = df[df['DatasetBalancin'] == balRatio]
    df = df[df['features_size'] == features]
    df = df[df['filename'].isin(['V1', 'I1', 'frequency', 'allcolumns', 'allvalues'])]
    df['filename'] = df['filename'].str.replace('allcolumns', 'allvalues') # they are the same thing and only one is available

    for metric in ["recall"]:
        # Initialize lists to store scores for each filename
        accuracies_by_filename = {filename: [] for filename in df['filename'].unique()}

        for filename in accuracies_by_filename.keys():
            for window_size in sorted(df['window_size'].unique()):
                accuracy = df[(df['filename'] == filename) & (df['window_size'] == window_size)][metric]
                if not accuracy.empty:
                    accuracies_by_filename[filename].append(accuracy.values[0])

        fig, ax = plt.subplots(figsize=(13, 7))

        data = []
        for filename, accuracies in accuracies_by_filename.items():
            for window_size, accuracy in zip(sorted(df['window_size'].unique()), accuracies):
                data.append({'filename': filename, 'window_size': window_size, metric: accuracy})
        df_plot = pd.DataFrame(data)

        df_plot['filename'] = pd.Categorical(df_plot['filename'], categories=['I1', 'V1', 'frequency', 'allvalues'], ordered=True)
        df_plot = df_plot.sort_values('filename')

        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
        for i, (filename, group) in enumerate(df_plot.groupby('filename')):
            sns.lineplot(data=group, x='window_size', y=metric, marker=markers[i % len(markers)], markersize=marker_size, linewidth=line_width, label=filename, ax=ax, zorder=5)

        plt.xlabel('Window Size')
        plt.ylabel(metric)
        ax.legend(title='Value', loc='lower center', ncol=5)
        ax.grid(True, which='both', zorder=0, alpha=0.5)
        plt.ylim(0, 1.01)
        plt.xlim(5, 50)
        plt.xticks([5, 10, 15, 20, 30, 40, 50])

        plt.tight_layout()
        plt.savefig(f'./Results/windows_{metric}.pdf')
        plt.show()


if __name__ == "__main__":
    # Run from the folder with the result file or change it in the following:
    results_file = "./evaluation_report.csv"

    if not os.path.exists("./Results"):
        os.makedirs("./Results")
    
    #results_file = "./sample_results/results_kalman_11_balRatio0.9_rf.csv"
    plot_window(results_file)    

    #results_file = "./sample_results/results_kalman_11_balRatio0.9_all.csv"
    plot_binary_models(results_file, window=20)   

    #results_file = "./sample_results/results_kalman_11_balRatio0.9_all.csv"
    for filename in column_order:
        print(f"Filename: {filename}")
        plot_delta_accuracies_on_each_source(results_file, filename=filename, window=20, model="RandomForestClassifier", features_number=11, balRatio=0.9)


