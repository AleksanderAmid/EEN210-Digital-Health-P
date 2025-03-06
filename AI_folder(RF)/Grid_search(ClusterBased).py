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
import numpy as np
from datetime import datetime

def cluster_events(timestamps, labels, gap_tolerance=2.0):
    """
    Clusters consecutive windows (with label==1) into events.
    Returns a list of representative event timestamps (using the median time of each cluster).
    """
    # Select timestamps where label is 1
    event_times = [t for t, lab in zip(timestamps, labels) if lab == 1]
    if not event_times:
        return []
    
    # Convert datetime strings to numeric timestamps (seconds since epoch)
    try:
        # First, try direct float conversion (in case timestamps are already numeric)
        numeric_times = [float(t) for t in event_times]
    except ValueError:
        # If that fails, try parsing as datetime strings
        try:
            numeric_times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').timestamp() for t in event_times]
        except ValueError:
            # If that fails too, try another common format
            numeric_times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f').timestamp() for t in event_times]
    
    # Ensure the event times are sorted
    numeric_times = sorted(numeric_times)
    clusters = []
    current_cluster = [numeric_times[0]]
    for t in numeric_times[1:]:
        if t - current_cluster[-1] <= gap_tolerance:
            current_cluster.append(t)
        else:
            clusters.append(np.median(current_cluster))
            current_cluster = [t]
    clusters.append(np.median(current_cluster))
    return clusters

def match_events(true_events, predicted_events, match_tolerance=2.0):
    """
    Matches predicted events with true events using a time tolerance.
    Returns the number of matched (true detected) events.
    """
    # Both true_events and predicted_events should already be numeric timestamps from cluster_events
    matched = 0
    used_preds = set()
    for te in true_events:
        for i, pe in enumerate(predicted_events):
            if i in used_preds:
                continue
            if abs(pe - te) <= match_tolerance:
                matched += 1
                used_preds.add(i)
                break
    return matched

def evaluate_configuration(X_train, X_test, y_train, y_test, params, timestamps=None):
    """
    Train and evaluate the model with the current configuration using cuML's RandomForestClassifier.
    Data remains on the GPU until final metric computations.
    Includes both window-level and event-level (cluster-based) evaluation if timestamps are provided.
    """
    # Ensure data remains in cuDF format.
    X_train = cudf.DataFrame(X_train)
    y_train = cudf.Series(y_train)
    X_test = cudf.DataFrame(X_test)
    y_test = cudf.Series(y_test)
    
    model = cuRF(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        max_features="sqrt",
        n_bins=16,
        bootstrap=True,
        n_streams=8,
        split_criterion="gini"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Convert predictions and labels to CPU for metric calculations.
    y_pred_cpu = y_pred.to_pandas()
    y_test_cpu = y_test.to_pandas()

    # Map labels to binary (if your fall label is '5', map it to 1; everything else to 0)
    y_test_cpu = y_test_cpu.apply(lambda x: 1 if x == 5 else 0)
    y_pred_cpu = y_pred_cpu.apply(lambda x: 1 if x == 5 else 0)

    # Window-level metrics
    accuracy = accuracy_score(y_test_cpu, y_pred_cpu)
    precision = precision_score(y_test_cpu, y_pred_cpu, zero_division=0)
    recall = recall_score(y_test_cpu, y_pred_cpu, zero_division=0)
    f1 = f1_score(y_test_cpu, y_pred_cpu, zero_division=0)
    
    # Calculate fall-specific accuracy.
    true_falls = (y_test_cpu == 1).sum()
    correct_falls = ((y_test_cpu == 1) & (y_pred_cpu == 1)).sum()
    fall_accuracy = (correct_falls / true_falls) if true_falls > 0 else 0
    
    result = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'fall_accuracy': fall_accuracy * 100
    }
    
    # Event-level (cluster-based) evaluation if timestamps are provided
    if timestamps is not None:
        # Parameters for event clustering and matching
        gap_tolerance = 2.0      # seconds: windows closer than this are clustered together
        match_tolerance = 2.0    # seconds: predicted event must fall within this window of a true event
        event_duration = 2.0     # seconds: assumed duration of an event (used to approximate TN)
        
        # Convert timestamps to pandas Series if it's a cudf Series
        if isinstance(timestamps, cudf.Series):
            timestamps = timestamps.to_pandas()
        
        # Cluster events
        true_events = cluster_events(timestamps, y_test_cpu, gap_tolerance)
        predicted_events = cluster_events(timestamps, y_pred_cpu, gap_tolerance)
        
        # Match events
        TP = match_events(true_events, predicted_events, match_tolerance)
        FN = len(true_events) - TP
        FP = len(predicted_events) - TP
        
        # Estimate True Negatives (TN)
        try:
            # Convert timestamps to numeric values for duration calculation
            try:
                # First try direct conversion
                min_time = float(timestamps.min())
                max_time = float(timestamps.max())
            except ValueError:
                # If that fails, try parsing as datetime strings
                try:
                    min_time = datetime.strptime(timestamps.min(), '%Y-%m-%d %H:%M:%S').timestamp()
                    max_time = datetime.strptime(timestamps.max(), '%Y-%m-%d %H:%M:%S').timestamp()
                except ValueError:
                    # If that fails too, try another common format
                    min_time = datetime.strptime(timestamps.min(), '%Y-%m-%d %H:%M:%S.%f').timestamp()
                    max_time = datetime.strptime(timestamps.max(), '%Y-%m-%d %H:%M:%S.%f').timestamp()
            
            total_duration = max_time - min_time
            n_segments = int(total_duration / event_duration)
            TN = max(n_segments - (TP + FN + FP), 0)
            
            # Calculate event-level metrics
            event_accuracy = ((TP + TN) / (TP + TN + FP + FN) * 100) if (TP + TN + FP + FN) > 0 else 0
            event_recall = (TP / (TP + FN) * 100) if (TP + FN) > 0 else 0
            event_precision = (TP / (TP + FP) * 100) if (TP + FP) > 0 else 0
            event_f1 = (2 * event_precision * event_recall / (event_precision + event_recall)) if (event_precision + event_recall) > 0 else 0
            
            # Add event-level metrics to result
            result.update({
                'event_accuracy': event_accuracy,
                'event_recall': event_recall,
                'event_precision': event_precision,
                'event_f1': event_f1,
                'true_events': len(true_events),
                'predicted_events': len(predicted_events),
                'TP': TP,
                'FN': FN,
                'FP': FP,
                'TN': TN
            })
        except Exception as e:
            print(f"Error in event-level evaluation: {e}")
            # If event-level evaluation fails, continue with window-level metrics only
    
    return result


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
    window_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 100, 110, 120, 130, 145, 150]
    overlap_sizes = [75,80,90]  # in percentage
    n_estimators = [300, 400, 500]
    max_depths = [100, 300, 500]
    
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
            
            # Extract timestamps before dropping them for model training
            timestamps = processed_data['timestamp'] if 'timestamp' in processed_data.columns else None
            
            X = processed_data.drop(['label'], axis=1)
            if 'timestamp' in X.columns:
                X = X.drop(['timestamp'], axis=1)
            y = processed_data['label']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # If timestamps exist, split them the same way as the data
            if timestamps is not None:
                # Convert to numpy for indexing if it's a cudf Series
                if isinstance(timestamps, cudf.Series):
                    timestamps_np = timestamps.to_numpy()
                else:
                    timestamps_np = timestamps
                
                # Get the indices of the test set
                test_indices = y_test.index.to_pandas() if isinstance(y_test, cudf.Series) else y_test.index
                
                # Extract timestamps for test set
                test_timestamps = [timestamps_np[i] for i in test_indices]
            else:
                test_timestamps = None
            
            for n_estimator in n_estimators:
                for max_depth in max_depths:
                    params = {'n_estimators': n_estimator, 'max_depth': max_depth}
                    print(f"Testing with params: {params}")
                    
                    # Evaluate current configuration
                    metrics = evaluate_configuration(X_train, X_test, y_train, y_test, params, test_timestamps)
                    
                    # Store results
                    result = {
                        'window_size': window_size,
                        'overlap': overlap,
                        'params': params,
                        **metrics
                    }
                    all_results.append(result)

                    print(f"Window-Level - Accuracy: {metrics['accuracy']:.2f}%, F1-Score: {metrics['f1']:.2f}%, Fall Detection Accuracy: {metrics['fall_accuracy']:.2f}%")
                    
                    # Print event-level metrics if available
                    if 'event_accuracy' in metrics:
                        print(f"Event-Level - Accuracy: {metrics['event_accuracy']:.2f}%, Recall: {metrics['event_recall']:.2f}%, Precision: {metrics['event_precision']:.2f}%, F1-Score: {metrics['event_f1']:.2f}%")
                        print(f"Event-Level - True Events: {metrics['true_events']}, Predicted Events: {metrics['predicted_events']}, TP: {metrics['TP']}, FN: {metrics['FN']}, FP: {metrics['FP']}")
            
            # Free memory after processing each window_size and overlap combination.
            del processed_data, X, y, X_train, X_test, y_train, y_test
            if timestamps is not None:
                del timestamps, test_timestamps
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()
    
    # Find overall best configuration based on event-level F1 score if available, otherwise fall_accuracy
    if any('event_f1' in result for result in all_results):
        best_result = max([r for r in all_results if 'event_f1' in r], key=lambda x: x['event_f1'])
        print("\n=== BEST OVERALL CONFIGURATION (Based on Event-Level F1 Score) ===")
    else:
        best_result = max(all_results, key=lambda x: x['fall_accuracy'])
        print("\n=== BEST OVERALL CONFIGURATION (Based on Window-Level Fall Accuracy) ===")
    
    print(f"Window Size: {best_result['window_size']}")
    print(f"Overlap: {best_result['overlap']}%")
    print(f"Random Forest Parameters: {best_result['params']}")
    print(f"Window-Level - Overall Accuracy: {best_result['accuracy']:.2f}%")
    print(f"Window-Level - Fall Detection Accuracy: {best_result['fall_accuracy']:.2f}%")
    print(f"Window-Level - F1-Score: {best_result['f1']:.2f}%")
    
    if 'event_f1' in best_result:
        print(f"Event-Level - Accuracy: {best_result['event_accuracy']:.2f}%")
        print(f"Event-Level - Recall: {best_result['event_recall']:.2f}%")
        print(f"Event-Level - Precision: {best_result['event_precision']:.2f}%")
        print(f"Event-Level - F1-Score: {best_result['event_f1']:.2f}%")
        print(f"Event-Level - True Events: {best_result['true_events']}, Predicted Events: {best_result['predicted_events']}")
        print(f"Event-Level - TP: {best_result['TP']}, FN: {best_result['FN']}, FP: {best_result['FP']}")
    
    # Save results
    with open(os.path.join(result_dir, 'gridsearch_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResults saved to {result_dir}")

if __name__ == "__main__":
    test_all_parameters()