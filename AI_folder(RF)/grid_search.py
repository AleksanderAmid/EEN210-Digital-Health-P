import cudf
import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import json
from GPU_RUNNER import process_windows_vectorized
import seaborn as sns
from joblib import dump
from RF_MODEL_GPU import load_data
import gc

#--------------------
def balance_training_data(X, y, random_state=42):
    """
    Balances the training data by oversampling the minority class to match the majority class count.
    This mimics scikit-learn's class_weight='balanced' behavior.
    """
    # Combine X and y into one DataFrame
    df = X.copy()
    df['label'] = y
    # Compute class counts and convert to pandas Series for iteration
    counts = df['label'].value_counts().to_pandas()
    max_count = counts.max()
    balanced_df_list = []
    
    for label, count in counts.items():
        label_df = df[df['label'] == label]
        # Oversample if needed
        if count < max_count:
            label_df_balanced = label_df.sample(n=int(max_count), replace=True, random_state=random_state)
        else:
            label_df_balanced = label_df
        balanced_df_list.append(label_df_balanced)
    
    balanced_df = cudf.concat(balanced_df_list)
    # Shuffle the balanced DataFrame
    balanced_df = balanced_df.sample(frac=1, random_state=random_state)
    
    # Separate features and label
    y_balanced = balanced_df['label']
    X_balanced = balanced_df.drop('label', axis=1)
    return X_balanced, y_balanced

def evaluate_configuration(X_train, X_test, y_train, y_test, params):
    """
    Train and evaluate the model with the current configuration using cuML's RandomForestClassifier.
    Data remains on the GPU until final metric computations.
    """
    # Ensure data remains in cuDF format.
    X_train = cudf.DataFrame(X_train)
    y_train = cudf.Series(y_train)
    X_test = cudf.DataFrame(X_test)
    y_test = cudf.Series(y_test)
    
    # Balance the training data to mimic class_weight="balanced"------------------
    X_train, y_train = balance_training_data(X_train, y_train, random_state=42)
    
    # Set random_state for reproducibility and use the given hyperparameters.
    model = cuRF(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        max_features="sqrt",
        n_bins=16,
        bootstrap=True,
        n_streams=1,
        split_criterion="gini",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Convert predictions and labels to CPU for metric calculations.
    y_pred_cpu = y_pred.to_pandas()
    y_test_cpu = y_test.to_pandas()

    # Map labels to binary (if your fall label is '5', map it to 1; everything else to 0)
    y_test_cpu = y_test_cpu.apply(lambda x: 1 if x == 5 else 0)
    y_pred_cpu = y_pred_cpu.apply(lambda x: 1 if x == 5 else 0)

    accuracy = accuracy_score(y_test_cpu, y_pred_cpu)
    precision = precision_score(y_test_cpu, y_pred_cpu, zero_division=0)
    recall = recall_score(y_test_cpu, y_pred_cpu, zero_division=0)
    f1 = f1_score(y_test_cpu, y_pred_cpu, zero_division=0)
    
    # Calculate fall-specific accuracy.
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

def test_all_parameters():
    # Optimize GPU usage by setting persistent mode, power limit, and disabling auto-boost.
    os.system("sudo nvidia-smi -pm 1")  # Enable persistent mode.
    os.system("sudo nvidia-smi -pl 75")  # Set power limit.
    os.system("sudo nvidia-smi --auto-boost-default=0")

    # Create Model_Result directory if it doesn't exist
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model_Result")
    os.makedirs(result_dir, exist_ok=True)
    
    # Load the data
    print("Please select your training data CSV file...")
    data = load_data("Select training data CSV file")
    if data is None:
        return

    # Convert to cuDF DataFrame if not already.
    if not isinstance(data, cudf.DataFrame):
        data = cudf.DataFrame(data)
    
    # Define parameter combinations to test
    window_sizes = [145, 150,160,170]
    overlap_sizes = [25,50,75]  # in percentage
    n_estimators = [50,75,100,150,200,300]
    max_depths = [30,50,70,100]
    
    # Store results for all combinations
    all_results = []
    
    for window_size in window_sizes:
        for overlap in overlap_sizes:
            print(f"\nProcessing with window_size={window_size}, overlap={overlap}%")
            
            # Process data with current window size and overlap
            processed_data = process_windows_vectorized(data, window_size=window_size, overlap_percent=overlap)
            
            if len(processed_data) == 0:
                print(f"Error: No features extracted for window_size={window_size}, overlap={overlap}")
                continue
            
            X = processed_data.drop('label', axis=1)
            y = processed_data['label']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            for n_estimator in n_estimators:
                for max_depth in max_depths:
                    params = {'n_estimators': n_estimator, 'max_depth': max_depth}
                    print(f"Testing with params: {params}")
                    
                    # Evaluate current configuration
                    metrics = evaluate_configuration(X_train, X_test, y_train, y_test, params)
                    
                    # Store results
                    result = {
                        'window_size': window_size,
                        'overlap': overlap,
                        'params': params,
                        **metrics
                    }
                    all_results.append(result)

                    print(f"Accuracy: {metrics['accuracy']:.2f}%, F1-Score: {metrics['f1']:.2f}%, Fall Detection Accuracy: {metrics['fall_accuracy']:.2f}%")
            
            # Free memory after processing each window_size and overlap combination.
            del processed_data, X, y, X_train, X_test, y_train, y_test
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()
    
    # Find overall best configuration
    best_result = max(all_results, key=lambda x: x['fall_accuracy'])
    
    print("\n=== BEST OVERALL CONFIGURATION ===")
    print(f"Window Size: {best_result['window_size']}")
    print(f"Overlap: {best_result['overlap']}%")
    print(f"Random Forest Parameters: {best_result['params']}")
    print(f"Overall Accuracy: {best_result['accuracy']:.2f}%")
    print(f"Fall Detection Accuracy: {best_result['fall_accuracy']:.2f}%")
    print(f"F1-Score: {best_result['f1']:.2f}%")
    
    # Save results
    with open(os.path.join(result_dir, 'gridsearch_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResults saved to {result_dir}")

if __name__ == "__main__":
    test_all_parameters()
