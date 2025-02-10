import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import json
from model_trainer_evaluator import load_data, process_data_with_windows
import seaborn as sns
from joblib import dump

def custom_precision_score(y_true, y_pred):
    return precision_score(y_true, y_pred, average='weighted', zero_division=0)

def custom_recall_score(y_true, y_pred):
    return recall_score(y_true, y_pred, average='weighted', zero_division=0)

def custom_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)

def test_all_parameters():
    # Create Model_Result directory if it doesn't exist
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model_Result")
    os.makedirs(result_dir, exist_ok=True)
    
    # Load the data
    print("Please select your training data CSV file...")
    data = load_data("Select training data CSV file")
    if data is None:
        return

    # Define parameter combinations to test
    window_sizes = [10, 20, 30, 40, 50, 60, 70, 80]
    overlap_sizes = [12, 25, 50]  # in percentage
    
    # Store results for all combinations
    all_results = []
    
    for window_size in window_sizes:
        for overlap in overlap_sizes:
            print(f"\nProcessing with window_size={window_size}, overlap={overlap}%")
            
            # Process data with current window size and overlap
            processed_data = process_data_with_windows(data, window_size=window_size)
            
            if len(processed_data) == 0:
                print(f"Error: No features extracted for window_size={window_size}, overlap={overlap}")
                continue
            
            X = processed_data.drop('label', axis=1)
            y = processed_data['label']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Define the parameter grid for RandomForest
            param_grid = {
                'n_estimators': [10, 50, 100, 200, 300, 400, 500],
                'max_depth': [2, 5, 10, 15, 20, 25, 30, None]
            }
            
            # Update scoring metrics with custom scorers
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(custom_precision_score),
                'recall': make_scorer(custom_recall_score),
                'f1': make_scorer(custom_f1_score)
            }
            
            # Create and run GridSearchCV
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=5,
                scoring=scoring,
                refit='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            print("Running GridSearchCV...")
            grid_search.fit(X_train, y_train)
            
            # Get best parameters and scores
            result = {
                'window_size': window_size,
                'overlap': overlap,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_score': grid_search.score(X_test, y_test),
                'cv_results': {
                    metric: grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]
                    for metric in scoring.keys()
                }
            }
            
            all_results.append(result)
            
            print(f"\nResults for window_size={window_size}, overlap={overlap}:")
            print(f"Best parameters: {result['best_params']}")
            print(f"Best CV accuracy: {result['best_score']:.4f}")
            print(f"Test accuracy: {result['test_score']:.4f}")
            print("CV Metrics:")
            for metric, score in result['cv_results'].items():
                print(f"  {metric}: {score:.4f}")
    
    # Find overall best configuration
    best_result = max(all_results, key=lambda x: x['best_score'])
    
    print("\n=== BEST OVERALL CONFIGURATION ===")
    print(f"Window Size: {best_result['window_size']}")
    print(f"Overlap: {best_result['overlap']}%")
    print(f"Random Forest Parameters: {best_result['best_params']}")
    print(f"Best CV Accuracy: {best_result['best_score']:.4f}")
    print(f"Test Accuracy: {best_result['test_score']:.4f}")
    print("\nCV Metrics:")
    for metric, score in best_result['cv_results'].items():
        print(f"  {metric}: {score:.4f}")
    
    # Create visualizations for the best configuration
    # Process data with best window size and overlap
    best_processed_data = process_data_with_windows(
        data, 
        window_size=best_result['window_size']
    )
    
    X = best_processed_data.drop('label', axis=1)
    y = best_processed_data['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model with best parameters
    best_model = RandomForestClassifier(
        random_state=42,
        **best_result['best_params']
    )
    best_model.fit(X_train, y_train)
    
    # Save best model
    model_path = os.path.join(result_dir, "best_model.joblib")
    dump(best_model, model_path)
    
    # Save all results to JSON
    results_path = os.path.join(result_dir, 'gridsearch_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        json_results = []
        for result in all_results:
            json_result = {
                'window_size': int(result['window_size']),
                'overlap': int(result['overlap']),
                'best_params': {
                    k: (int(v) if isinstance(v, np.int64) else v)
                    for k, v in result['best_params'].items()
                },
                'best_score': float(result['best_score']),
                'test_score': float(result['test_score']),
                'cv_results': {
                    k: float(v) for k, v in result['cv_results'].items()
                }
            }
            json_results.append(json_result)
        json.dump(json_results, f, indent=4)
    
    # Create heatmap of window size vs overlap
    results_df = pd.DataFrame([
        {
            'Window Size': r['window_size'],
            'Overlap': r['overlap'],
            'Accuracy': r['best_score']
        }
        for r in all_results
    ])
    
    pivot_table = results_df.pivot(
        index='Window Size',
        columns='Overlap',
        values='Accuracy'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd')
    plt.title('Accuracy for Different Window Sizes and Overlaps')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'parameter_heatmap.png'))
    plt.close()
    
    print(f"\nResults and visualizations saved to {result_dir}")
    print(f"Best model saved as {model_path}")

if __name__ == "__main__":
    test_all_parameters()