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
from RF_MODEL_GPU import load_data

# Expose a global batch size parameter for model training.
BATCH_SIZE = 10000  # Adjust as needed

def process_windows_vectorized(data, window_size, overlap_percent):
    """
    Create a vectorized sliding window view and compute aggregated features in batch.
    This approach uses cupy's stride tricks to avoid launching many small GPU kernels.
    """
    # Ensure data is sorted by timestamp.
    data = data.sort_values('timestamp')
    step_size = int(window_size * (1 - overlap_percent / 100))
    n_rows = len(data)
    
    if n_rows < window_size:
        return cudf.DataFrame()  # Not enough data
    
    num_windows = (n_rows - window_size) // step_size + 1

    # Define sensor columns.
    sensor_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                   'gyroscope_x', 'gyroscope_y', 'gyroscope_z']

    # Convert each sensor column to a cupy array with a consistent type.
    sensor_arrays = {col: cp.asarray(data[col].to_numpy(), dtype=cp.float32)
                     for col in sensor_cols}

    # Create sliding windows using cupy's stride_tricks.
    windows = {}
    for col, arr in sensor_arrays.items():
        # arr.strides returns strides in bytes. The new shape and strides are computed
        shape = (num_windows, window_size)
        strides = (arr.strides[0] * step_size, arr.strides[0])
        windows[col] = cp.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    
    # Compute aggregated features for each sensor column.
    feature_dict = {}
    for col in sensor_cols:
        col_windows = windows[col]
        feature_dict[f'{col}_mean'] = cp.mean(col_windows, axis=1)
        feature_dict[f'{col}_std']  = cp.std(col_windows, axis=1)
        feature_dict[f'{col}_max']  = cp.max(col_windows, axis=1)
        feature_dict[f'{col}_min']  = cp.min(col_windows, axis=1)
    
    # Compute magnitudes for accelerometer and gyroscope using batch operations.
    acc_stack = cp.stack([windows[col] for col in ['acceleration_x', 'acceleration_y', 'acceleration_z']], axis=2)
    gyro_stack = cp.stack([windows[col] for col in ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']], axis=2)
    
    # Euclidean norm along the sensor axis.
    acc_mag = cp.linalg.norm(acc_stack, axis=2)
    gyro_mag = cp.linalg.norm(gyro_stack, axis=2)
    
    feature_dict['acc_mag_mean'] = cp.mean(acc_mag, axis=1)
    feature_dict['acc_mag_std']  = cp.std(acc_mag, axis=1)
    feature_dict['acc_mag_max']  = cp.max(acc_mag, axis=1)
    feature_dict['acc_mag_min']  = cp.min(acc_mag, axis=1)
    
    feature_dict['gyro_mag_mean'] = cp.mean(gyro_mag, axis=1)
    feature_dict['gyro_mag_std']  = cp.std(gyro_mag, axis=1)
    feature_dict['gyro_mag_max']  = cp.max(gyro_mag, axis=1)
    feature_dict['gyro_mag_min']  = cp.min(gyro_mag, axis=1)
    
    # Process labels: fill nulls (e.g. with 0) before creating sliding windows.
    labels = cp.asarray(data['label'].fillna(0).to_numpy())
    shape_label = (num_windows, window_size)
    strides_label = (labels.strides[0] * step_size, labels.strides[0])
    label_windows = cp.lib.stride_tricks.as_strided(labels, shape=shape_label, strides=strides_label)
    # For simplicity, take the first label in each window.
    feature_dict['label'] = label_windows[:, 0]
    
    # Convert cupy arrays to numpy arrays, then build a pandas DataFrame.
    import pandas as pd
    pd_features = {k: cp.asnumpy(v) for k, v in feature_dict.items()}
    features_df = pd.DataFrame(pd_features)
    
    # Return a cuDF DataFrame.
    return cudf.DataFrame.from_pandas(features_df)



def evaluate_configuration(X_train, X_test, y_train, y_test):
    """
    Train and evaluate the model with the current configuration using cuML's RandomForestClassifier.
    Data remains on the GPU until final metric computations.
    """
    # Ensure data remains in cuDF format.
    X_train = cudf.DataFrame(X_train)
    y_train = cudf.Series(y_train)
    X_test = cudf.DataFrame(X_test)
    y_test = cudf.Series(y_test)
    
    model = cuRF(
        n_estimators=500,
        max_depth=20,          # Reduced depth for faster training.
        max_features="sqrt",   # Helps GPU parallelization.
        n_bins=16,             # Reduces memory usage.
        bootstrap=True,
        n_streams=8,           # Increase parallelism.
        split_criterion="gini" # Faster than "entropy".
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



def test_window_configurations():
    # Optimize GPU usage by setting persistent mode, power limit, and disabling auto-boost.
    os.system("sudo nvidia-smi -pm 1")  # Enable persistent mode.
    os.system("sudo nvidia-smi -pl 75")  # Set power limit.
    os.system("sudo nvidia-smi --auto-boost-default=0")

    # Create results directory.
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model_Result")
    os.makedirs(result_dir, exist_ok=True)
    
    # Load the data (ensuring load_data returns a DataFrame that can be converted to cuDF).
    print("Please select your training data CSV file...")
    data = load_data("Select training data CSV file")
    if data is None:
        return

    # Convert to cuDF DataFrame if not already.
    if not isinstance(data, cudf.DataFrame):
        data = cudf.DataFrame(data)
    
    # Define configurations to test.
    window_sizes = [10, 25, 50, 70, 100, 110, 125, 130, 140]
    overlap_percentages = [5, 15, 25, 50, 75]
    
    results = []
    
    for window_size in window_sizes:
        for overlap in overlap_percentages:
            print(f"\nTesting Window Size: {window_size}, Overlap: {overlap}%")

            # Process data with current configuration using the vectorized approach.
            processed_data = process_windows_vectorized(data, window_size, overlap)
            if len(processed_data) == 0:
                print(f"Error: No features extracted for window_size={window_size}, overlap={overlap}")
                continue
            
            X = processed_data.drop('label', axis=1)
            y = processed_data['label']

            # Print class distribution.
            y_cpu = y.to_pandas()
            print(f"Fall samples: {(y_cpu == 1).sum()}, No-fall samples: {(y_cpu == 0).sum()}")
            
            # Split the data using cuML's train_test_split.
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Evaluate current configuration.
            metrics = evaluate_configuration(X_train, X_test, y_train, y_test)
            
            # Store results.
            result = {
                'window_size': window_size,
                'overlap': overlap,
                **metrics
            }
            results.append(result)

            print(f"Accuracy: {metrics['accuracy']:.2f}%, F1-Score: {metrics['f1']:.2f}%, Fall Detection Accuracy: {metrics['fall_accuracy']:.2f}%")

    # Identify best configuration based on fall detection accuracy.
    best_result = max(results, key=lambda x: x['fall_accuracy'])
    
    print("\n=== BEST CONFIGURATION ===")
    print(f"Window Size: {best_result['window_size']}")
    print(f"Overlap: {best_result['overlap']}%")
    print(f"Overall Accuracy: {best_result['accuracy']:.2f}%")
    print(f"Fall Detection Accuracy: {best_result['fall_accuracy']:.2f}%")
    print(f"F1-Score: {best_result['f1']:.2f}%")

    # Save results.
    with open(os.path.join(result_dir, 'window_overlap_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {result_dir}")

    return best_result

if __name__ == "__main__":
    test_window_configurations()
