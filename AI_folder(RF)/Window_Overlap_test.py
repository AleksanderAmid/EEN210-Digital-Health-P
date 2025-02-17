import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from RF_MODEL import load_data, calculate_magnitudes, extract_features

def process_data_with_windows(data, window_size, overlap_percent):
    """Process data using sliding windows with specified overlap"""
    processed_data = []
    data = data.sort_values('timestamp')
    
    step_size = int(window_size * (1 - overlap_percent/100))
    
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i:i + window_size]
        features = extract_features(window)
        features['label'] = 1 if features['label'] == 5 else 0
        processed_data.append(features)
    
    return pd.DataFrame(processed_data)

def evaluate_configuration(X_train, X_test, y_train, y_test):
    """Train and evaluate model with current configuration"""
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=100,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate fall-specific accuracy
    true_falls = (y_test == 1).sum()
    correct_falls = sum((y_test == 1) & (y_pred == 1))
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
    
    # Load the data
    print("Please select your training data CSV file...")
    data = load_data("Select training data CSV file")
    if data is None:
        return
    
    # Define configurations to test
    window_sizes = [10, 25, 50, 70, 100,110, 125,135,150]
    overlap_percentages = [5, 15, 25, 50]
    
    results = []
    
    # Store results for plotting
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
            
            # Print class distribution
            fall_count = (y == 1).sum()
            no_fall_count = (y == 0).sum()
            print(f"Class distribution:")
            print(f"Fall samples: {fall_count}")
            print(f"No-fall samples: {no_fall_count}")
            
            # Split the data
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
            
            print(f"Results:")
            print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
            print(f"Fall Detection Accuracy: {metrics['fall_accuracy']:.2f}%")
            print(f"F1-Score: {metrics['f1']:.2f}%")
    
    # Find best configuration
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
    
    # Create heatmaps
    metrics_to_plot = {
        'accuracy': 'Overall Accuracy',
        'fall_accuracy': 'Fall Detection Accuracy',
        'f1': 'F1-Score'
    }
    
    for metric, title in metrics_to_plot.items():
        results_df = pd.DataFrame([
            {
                'Window Size': r['window_size'],
                'Overlap': r['overlap'],
                'Value': r[metric]
            }
            for r in results
        ])
        
        pivot_table = results_df.pivot(
            index='Window Size',
            columns='Overlap',
            values='Value'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title(f'{title} for Different Window Sizes and Overlaps')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'window_overlap_{metric}_heatmap.png'))
        plt.close()
    
    # Create 3D surface plot for fall detection accuracy
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X = np.array(sorted(list(set(plot_data['window_size']))))
    Y = np.array(sorted(list(set(plot_data['overlap']))))
    X, Y = np.meshgrid(X, Y)
    
    Z = np.zeros_like(X)
    for i, overlap in enumerate(sorted(list(set(plot_data['overlap'])))):
        for j, window in enumerate(sorted(list(set(plot_data['window_size'])))):
            matching_results = [r for r in results 
                              if r['window_size'] == window and r['overlap'] == overlap]
            if matching_results:
                Z[i, j] = matching_results[0]['fall_accuracy']
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
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