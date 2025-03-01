import argparse
import os
import galileo

def main():
    parser = argparse.ArgumentParser(description='Train or analyze using SVM with train-test split on extracted features.')
    parser.add_argument('--input', '-i', required=True, help='Input features CSV file for training or file to analyze')
    parser.add_argument('--model', '-m', default='svm_model.joblib', help='Path to save/load model')
    parser.add_argument('--column', '-c', help='Specific column to analyze (for analyze mode)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no-eval', action='store_true', help='Skip model evaluation')
    
    args = parser.parse_args()
    
    # Ensure input is a CSV file
    if not args.input.lower().endswith('.csv'):
        print(f"Error: Input file must be a CSV file containing extracted features")
        return
        
    print(f"Training with features file using train-test split: {args.input}")
    galileo.train_with_features(
        features_file=args.input,
        model_path=args.model,
        train_ratio=0.8,
        random_seed=args.seed,
        skip_evaluation=args.no_eval
    )
if __name__ == "__main__":
    main()