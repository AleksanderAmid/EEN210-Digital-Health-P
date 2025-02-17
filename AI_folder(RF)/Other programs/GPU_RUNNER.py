import cudf
import cupy as cp
import numpy as np
from cuml.ensemble import RandomForestClassifier as cuRF  # GPU-accelerated RF
from cuml.model_selection import train_test_split      # Works with cuDF DataFrames
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from model_trainer_evaluator import load_data, calculate_magnitudes, extract_features

def process_data_with_windows(data, window_size, overlap_percent):
    """
    Process data using sliding windows with specified overlap.
    Uses cuDF for data manipulation. If extract_features expects a pandas DataFrame,
    we convert the window to pandas before passing it.
    """
    processed_data = []
    data = data.sort_values('timestamp')
    
    step_size = int(window_size * (1 - overlap_percent / 100))
    
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i:i + window_size]
        # Convert to pandas if needed by extract_features (or update extract_features for cuDF)
        features = extract_features(window.to_pandas())
        # Update label conversion as in your original code
        features['label'] = 1 if features['label'] == 5 else 0
        processed_data.append(features)
    
    # Convert list of dicts to a cuDF DataFrame
    return cudf.DataFrame(processed_data)

def evaluate_configuration(X_train, X_test, y_train, y_test):
    """
    Train and evaluate the model with current configuration using cuML's RandomForestClassifier.
    Note: cuML's RF may not support all parameters (e.g., class_weight) so adjustments were made.
    """
    # Initialize GPU-accelerated RandomForestClassifier
    model = cuRF(
        n_estimators=500,
        max_depth=100,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Ensure predictions and labels are on CPU for evaluation using scikit-learn metrics
    if isinstance(y_pred, cp.ndarray):
        y_pred_cpu = cp.asnumpy(y_pred)
    else:
        y_pred_cpu = y_pred

    if isinstance(y_test, cp.ndarray):
        y_test_cpu = cp.asnumpy(y_test)
    else:
        y_test_cpu = y_test

    accuracy = accuracy_score(y_test_cpu, y_pred_cpu)
    precision = precision_score(y_test_cpu, y_pred_cpu, zero_division=0)
    recall = recall_score(y_test_cpu, y_pred_cpu, zero_division=0)
    f1 = f1_score(y_test_cpu, y_pred_cpu, zero_division=0)
    
    # Calculate fall-specific accuracy
    true_falls = (y_test_cpu == 1).sum()
    correct_falls = ((y_test_cpu == 1) & (y_pred_cpu == 1)).sum()
    fall_accuracy = (correct_falls / true_falls) if true_falls > 0 else 0
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'fall_accuracy': fall_accuracy * 100
    }

def test_window_configurations():
    # Create results directory
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model_Result")
    os.makedirs(result_dir, exist_ok=True)
    
    # Load the data (ensure load_data returns a DataFrame that can be converted to cuDF)
    print("Please select your training data CSV file...")
    data = load_data("Select training data CSV file")
    if data is None:
        return

    # Convert to cuDF DataFrame if not already
    if not isinstance(data, cudf.DataFrame):
        data = cudf.DataFrame(data)
    
    # Define configurations to test
    window_sizes = [10, 25, 50, 70, 100, 110, 125, 130, 140]
    overlap_percentages = [5, 15, 25, 50, 75]
    
    results = []
    
    # Data for plotting
    plot_data = {
        'window_size': [],
        'overlap': [],
        'accuracy': [],
        'fall_accuracy': [],
        'f1': []
    }
    
    total_combinations = len(window_sizes) * len(overlap_percentages)
    current_combination = 0
    
    for window_size in window_sizes:
        for overlap in overlap_percentages:
            current_combination += 1
            print(f"\nTesting combination {current_combination}/{total_combinations}")
            print(f"Window Size: {window_size}, Overlap: {overlap}%")
            
            # Process data with current configuration
            processed_data = process_data_with_windows(data, window_size, overlap)
            
            if len(processed_data) == 0:
                print(f"Error: No features extracted for window_size={window_size}, overlap={overlap}")
                continue
            
            X = processed_data.drop('label', axis=1)
            y = processed_data['label']
            
            # Print class distribution (convert to pandas for printing)
            y_cpu = y.to_pandas() if isinstance(y, cudf.Series) else y
            fall_count = (y_cpu == 1).sum()
            no_fall_count = (y_cpu == 0).sum()
            print("Class distribution:")
            print(f"Fall samples: {fall_count}")
            print(f"No-fall samples: {no_fall_count}")
            
            # Split the data using cuML's train_test_split (works with cuDF DataFrames)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Evaluate current configuration
            metrics = evaluate_configuration(X_train, X_test, y_train, y_test)
            
            # Store results
            result = {
                'window_size': window_size,
                'overlap': overlap,
                **metrics
            }
            results.append(result)
            
            # Store for plotting
            plot_data['window_size'].append(window_size)
            plot_data['overlap'].append(overlap)
            plot_data['accuracy'].append(metrics['accuracy'])
            plot_data['fall_accuracy'].append(metrics['fall_accuracy'])
            plot_data['f1'].append(metrics['f1'])
            
            print("Results:")
            print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
            print(f"Fall Detection Accuracy: {metrics['fall_accuracy']:.2f}%")
            print(f"F1-Score: {metrics['f1']:.2f}%")
    
    # Identify best configuration based on fall detection accuracy
    best_result = max(results, key=lambda x: x['fall_accuracy'])
    
    print("\n=== BEST CONFIGURATION ===")
    print(f"Window Size: {best_result['window_size']}")
    print(f"Overlap: {best_result['overlap']}%")
    print(f"Overall Accuracy: {best_result['accuracy']:.2f}%")
    print(f"Fall Detection Accuracy: {best_result['fall_accuracy']:.2f}%")
    print(f"F1-Score: {best_result['f1']:.2f}%")
    
    # Save results to JSON
    results_path = os.path.join(result_dir, 'window_overlap_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create heatmaps for visualizing metrics
    metrics_to_plot = {
        'accuracy': 'Overall Accuracy',
        'fall_accuracy': 'Fall Detection Accuracy',
        'f1': 'F1-Score'
    }
    
    # Use pandas for plotting (conversion from cuDF if necessary)
    import pandas as pd
    for metric, title in metrics_to_plot.items():
        results_df = pd.DataFrame([
            {
                'Window Size': r['window_size'],
                'Overlap': r['overlap'],
                'Value': r[metric]
            }
            for r in results
        ])
        
        pivot_table = results_df.pivot(index='Window Size', columns='Overlap', values='Value')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title(f'{title} for Different Window Sizes and Overlaps')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'window_overlap_{metric}_heatmap.png'))
        plt.close()
    
    # Create 3D surface plot for fall detection accuracy
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X_plot = np.array(sorted(list(set(plot_data['window_size']))))
    Y_plot = np.array(sorted(list(set(plot_data['overlap']))))
    X_plot, Y_plot = np.meshgrid(X_plot, Y_plot)
    
    Z = np.zeros_like(X_plot, dtype=float)
    for i, overlap in enumerate(sorted(list(set(plot_data['overlap'])))):
        for j, window in enumerate(sorted(list(set(plot_data['window_size'])))):
            matching_results = [r for r in results if r['window_size'] == window and r['overlap'] == overlap]
            if matching_results:
                Z[i, j] = matching_results[0]['fall_accuracy']
    
    surf = ax.plot_surface(X_plot, Y_plot, Z, cmap='viridis')
    ax.set_xlabel('Window Size')
    ax.set_ylabel('Overlap %')
    ax.set_zlabel('Fall Detection Accuracy %')
    plt.colorbar(surf)
    plt.title('Fall Detection Accuracy Surface Plot')
    plt.savefig(os.path.join(result_dir, 'window_overlap_3d_surface.png'))
    plt.close()
    
    print(f"\nResults and visualizations saved to {result_dir}")
    return best_result

if __name__ == "__main__":
    test_window_configurations()
