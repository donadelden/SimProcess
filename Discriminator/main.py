import argparse
import os
import galileo

def main():
    parser = argparse.ArgumentParser(description='Train or use SVM to discriminate between real and simulated power plant data.')
    parser.add_argument('mode', choices=['train', 'analyze'], help='Mode of operation')
    parser.add_argument('--input', '-i', required=True, help='Input file for analysis or directory containing training files')
    parser.add_argument('--real', '-r', nargs='*', help='List of real data files for training')
    parser.add_argument('--simulated', '-s', nargs='*', help='List of simulated data files for training')
    parser.add_argument('--model', '-m', default='svm_model.joblib', help='Path to save/load model')
    parser.add_argument('--column', '-c', help='Specific column to analyze (e.g., V1, C2, frequency)')
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.real or not args.simulated:
            print("Need both real and simulated files for training")
            return
            
        # Prepare training data + labels
        training_files = args.real + args.simulated
        labels = [1] * len(args.real) + [0] * len(args.simulated)
        
        # Train model
        if args.column:
            print(f"Training model using only column: {args.column}")
            galileo.train_and_save_model(training_files, labels, args.model, target_column=args.column)
        else:
            print("Training model using all columns")
            galileo.train_and_save_model(training_files, labels, args.model)
    
    elif args.mode == 'analyze':
        if not os.path.exists(args.model):
            print(f"Model file {args.model} not found")
            return
            
        # Analyze file
        results = galileo.analyze_with_model(args.input, args.model, target_column=args.column)
        
        if results:
            print("\n=== Analysis Results ===")
            print(f"Classification: {results['classification']}")
            print(f"Confidence: {results['confidence']}%")
            
            window_results = results['window_predictions']
            real_windows = sum(window_results)
            simulated_windows = len(window_results) - real_windows
            
            print(f"\nWindow Statistics:")
            print(f"Total windows analyzed: {len(window_results)}")
            print(f"Windows classified as real: {real_windows} ({real_windows/len(window_results)*100:.1f}%)")
            print(f"Windows classified as simulated: {simulated_windows} ({simulated_windows/len(window_results)*100:.1f}%)")
        else:
            print("\nAnalysis could not be completed. Please check the messages above for details.")

if __name__ == "__main__":
    main()