import cudf
import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import json
from GPU_RUNNER import process_windows_vectorized
import gc
import numpy as np
from datetime import datetime

def cluster_events(timestamps, labels, gap_tolerance=2.0):
    """
    Clusters consecutive windows with label==1 into events.
    Returns a list of event timestamps (median time per cluster).
    """
    event_times = [t for t, lab in zip(timestamps, labels) if lab == 1]
    if not event_times:
        return []
    
    numeric_times = sorted([datetime.strptime(t, '%Y-%m-%d %H:%M:%S').timestamp() for t in event_times])
    clusters, current_cluster = [], [numeric_times[0]]
    
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
    Matches predicted events to true events based on a time tolerance.
    """
    matched, used_preds = 0, set()
    for te in true_events:
        for i, pe in enumerate(predicted_events):
            if i in used_preds:
                continue
            if abs(pe - te) <= match_tolerance:
                matched += 1
                used_preds.add(i)
                break
    return matched

def evaluate_cluster_based(X_train, X_test, y_train, y_test, params, timestamps):
    """
    Train and evaluate using cluster-based fall detection.
    """
    X_train, X_test = cudf.DataFrame(X_train), cudf.DataFrame(X_test)
    y_train, y_test = cudf.Series(y_train), cudf.Series(y_test)
    
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
    y_pred = model.predict(X_test).to_pandas()
    y_test = y_test.to_pandas()
    
    # Map labels to binary fall classification
    y_test, y_pred = y_test.apply(lambda x: 1 if x == 5 else 0), y_pred.apply(lambda x: 1 if x == 5 else 0)
    
    # Cluster actual and predicted fall events
    true_events = cluster_events(timestamps, y_test)
    predicted_events = cluster_events(timestamps, y_pred)
    
    # Match events and calculate cluster-based metrics
    TP = match_events(true_events, predicted_events)
    FN, FP = len(true_events) - TP, len(predicted_events) - TP
    
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'event_precision': precision, 'event_recall': recall, 'event_f1': f1, 'true_events': len(true_events), 'predicted_events': len(predicted_events)}

def test_all_parameters():
    """
    Runs grid search over hyperparameters using cluster-based evaluation.
    """
    os.system("sudo nvidia-smi -pm 1")
    os.system("sudo nvidia-smi -pl 75")
    os.system("sudo nvidia-smi --auto-boost-default=0")
    
    data = process_windows_vectorized(load_data("Select training data CSV file"))
    data = cudf.DataFrame(data) if not isinstance(data, cudf.DataFrame) else data
    
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model_Result")
    os.makedirs(result_dir, exist_ok=True)
    
    window_sizes, overlap_sizes = [10, 20, 30, 50], [75, 80, 90]
    n_estimators, max_depths = [300, 400], [100, 300]
    all_results = []
    
    for window_size in window_sizes:
        for overlap in overlap_sizes:
            processed_data = process_windows_vectorized(data, window_size, overlap)
            timestamps = processed_data['timestamp'].to_pandas() if 'timestamp' in processed_data.columns else None
            X, y = processed_data.drop(['label', 'timestamp'], axis=1), processed_data['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            for n_estimator in n_estimators:
                for max_depth in max_depths:
                    params = {'n_estimators': n_estimator, 'max_depth': max_depth}
                    metrics = evaluate_cluster_based(X_train, X_test, y_train, y_test, params, timestamps)
                    all_results.append({'window_size': window_size, 'overlap': overlap, 'params': params, **metrics})
                    print(f"Event-Level - Precision: {metrics['event_precision']:.2f}%, Recall: {metrics['event_recall']:.2f}%, F1-Score: {metrics['event_f1']:.2f}%")
            
            del processed_data, X, y, X_train, X_test, y_train, y_test, timestamps
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
    
    best_result = max(all_results, key=lambda x: x['event_f1'])
    print("\n=== BEST CONFIGURATION ===")
    print(f"Window Size: {best_result['window_size']}, Overlap: {best_result['overlap']}%")
    print(f"Event Precision: {best_result['event_precision']:.2f}%, Recall: {best_result['event_recall']:.2f}%, F1: {best_result['event_f1']:.2f}%")
    
    with open(os.path.join(result_dir, 'gridsearch_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"Results saved to {result_dir}")
    
if __name__ == "__main__":
    test_all_parameters()
