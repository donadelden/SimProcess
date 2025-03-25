"""
Visualization functions for the SimDetector framework.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import os

logger = logging.getLogger('simdetector.visualization')

def plot_feature_importance(importance_df, title="Feature Importance", metric_name=None, output_dir=None):
    """
    Plot feature importance scores and save as SVG with metric name in filename.
    
    Args:
        importance_df (pandas.DataFrame): DataFrame with 'Feature' and 'Importance' columns
        title (str): Plot title
        metric_name (str, optional): Name of the metric being analyzed (e.g., 'V1', 'power_real')
        output_dir (str, optional): Directory to save the output files
    """
    try:
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importance_df)), importance_df['Importance'])
        plt.xticks(range(len(importance_df)), importance_df['Feature'], rotation=45, ha='right')
        
        # Create filename with metric name if provided
        if metric_name:
            title_with_metric = f"Feature Importance - {metric_name}"
            plt.title(title_with_metric)
            filename = f"feature_importance_{metric_name}.svg"
        else:
            plt.title(title)
            filename = "feature_importance.svg"
        
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        
        # Add output directory to filename if provided
        if output_dir:
            filename = os.path.join(output_dir, filename)
        
        # Save as SVG
        plt.savefig(filename, format='svg')
        logger.info(f"Feature importance plot saved as '{filename}'")
        plt.close()
        return filename
        
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {str(e)}")
        return None


def plot_confusion_matrix(cm, classes=['Not Real', 'Real'], title="Confusion Matrix", output_file=None, output_dir=None):
    """
    Plot confusion matrix and save as SVG.
    
    Args:
        cm (numpy.ndarray): Confusion matrix from sklearn
        classes (list): Class labels
        title (str): Plot title
        output_file (str, optional): Filename for saved plot
        output_dir (str, optional): Directory to save the output files
    """
    try:
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        
        # Add axis ticks
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Set default filename if not provided
        if output_file is None:
            output_file = "confusion_matrix.svg"
        
        # Add output directory to filename if provided
        if output_dir:
            output_file = os.path.join(output_dir, output_file)
        
        # Save as SVG
        plt.savefig(output_file, format='svg')
        logger.info(f"Confusion matrix plot saved as '{output_file}'")
        plt.close()
        return output_file
        
    except Exception as e:
        logger.error(f"Error creating confusion matrix plot: {str(e)}")
        return None


def plot_prediction_distribution(predictions, probabilities, output_file=None, output_dir=None):
    """
    Plot the distribution of predictions and their confidence.
    
    Args:
        predictions (numpy.ndarray): Binary predictions (0 or 1)
        probabilities (numpy.ndarray): Prediction probabilities
        output_file (str, optional): Filename for saved plot
        output_dir (str, optional): Directory to save the output files
    """
    try:
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Extract probabilities for the positive class
        if probabilities.ndim > 1 and probabilities.shape[1] > 1:
            pos_probs = probabilities[:, 1]
        else:
            pos_probs = probabilities
        
        plt.figure(figsize=(10, 6))
        
        # Create subplot 1: Prediction distribution
        plt.subplot(1, 2, 1)
        plt.hist([0, 1], bins=[0, 0.5, 1.5], weights=[sum(predictions==0), sum(predictions==1)])
        plt.xticks([0.25, 0.75], ['Not Real', 'Real'])
        plt.title('Prediction Distribution')
        plt.ylabel('Count')
        
        # Create subplot 2: Confidence histogram
        plt.subplot(1, 2, 2)
        plt.hist(pos_probs, bins=10)
        plt.title('Confidence Distribution')
        plt.xlabel('Probability of Real')
        plt.ylabel('Count')
        
        plt.tight_layout()
        
        # Set default filename if not provided
        if output_file is None:
            output_file = "prediction_distribution.svg"
        
        # Add output directory to filename if provided
        if output_dir:
            output_file = os.path.join(output_dir, output_file)
        
        # Save as SVG
        plt.savefig(output_file, format='svg')
        logger.info(f"Prediction distribution plot saved as '{output_file}'")
        plt.close()
        return output_file
        
    except Exception as e:
        logger.error(f"Error creating prediction distribution plot: {str(e)}")
        return None


def generate_report_summary(report_file, output_file=None, output_dir=None):
    """
    Generate a summary plot from evaluation report CSV file.
    
    Args:
        report_file (str): Path to the CSV report file
        output_file (str, optional): Filename for saved plot
        output_dir (str, optional): Directory to save the output files
    """
    try:
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Read the report file
        df = pd.read_csv(report_file)
        
        # Check if there's data to plot
        if df.empty:
            logger.warning(f"No data found in report file: {report_file}")
            return None
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: F1 score by column
        ax1 = axes[0, 0]
        df.sort_values('f1score', ascending=False).plot(x='column', y='f1score', kind='bar', ax=ax1)
        ax1.set_title('F1 Score by Column')
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('')
        
        # Plot 2: Precision and Recall by column
        ax2 = axes[0, 1]
        df.sort_values('precision', ascending=False).plot(x='column', y=['precision', 'TPR'], kind='bar', ax=ax2)
        ax2.set_title('Precision and Recall (TPR) by Column')
        ax2.set_ylim([0, 1])
        ax2.set_xlabel('')
        ax2.legend(['Precision', 'Recall'])
        
        # Plot 3: Accuracy by column
        ax3 = axes[1, 0]
        df.sort_values('accuracy', ascending=False).plot(x='column', y='accuracy', kind='bar', ax=ax3)
        ax3.set_title('Accuracy by Column')
        ax3.set_ylim([0, 1])
        
        # Plot 4: FPR and FNR by column
        ax4 = axes[1, 1]
        df.sort_values('FPR', ascending=True).plot(x='column', y=['FPR', 'FNR'], kind='bar', ax=ax4)
        ax4.set_title('Error Rates by Column')
        ax4.set_ylim([0, 1])
        ax4.legend(['False Positive Rate', 'False Negative Rate'])
        
        plt.tight_layout()
        
        # Set default filename if not provided
        if output_file is None:
            output_file = "report_summary.svg"
        
        # Add output directory to filename if provided
        if output_dir:
            output_file = os.path.join(output_dir, output_file)
        
        # Save as SVG
        plt.savefig(output_file, format='svg')
        logger.info(f"Report summary plot saved as '{output_file}'")
        plt.close()
        return output_file
        
    except Exception as e:
        logger.error(f"Error creating report summary plot: {str(e)}")
        return None