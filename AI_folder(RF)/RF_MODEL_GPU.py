import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import tkinter as tk
from tkinter import filedialog
from joblib import dump, load
import seaborn as sns
import matplotlib.pyplot as plt
import os


# VANLIGA
'''def load_data(title="Select CSV file"):
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("CSV files", "*.csv")]
    )
    
    if not file_path:
        print("No data file selected. Exiting...")
        return None
        
    return pd.read_csv(file_path)'''

def load_data(filepath=None):
    if filepath is None:
        print("No CSV file path provided. Exiting...")
        return None
    return pd.read_csv(filepath)

def calculate_magnitudes(window_data):
    """Calculate magnitudes for accelerometer and gyroscope data"""
    acc_mag = np.sqrt(
        window_data['acceleration_x']**2 + 
        window_data['acceleration_y']**2 + 
        window_data['acceleration_z']**2
    )
    gyro_mag = np.sqrt(
        window_data['gyroscope_x']**2 + 
        window_data['gyroscope_y']**2 + 
        window_data['gyroscope_z']**2
    )
    return acc_mag, gyro_mag

def extract_features(window_data):
    """Extract features from a time window using individual aggregated calls."""
    # Define sensor columns.
    acc_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z']
    gyro_cols = ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    
    # Cast sensor data to a consistent type (float32).
    acc_data = window_data[acc_cols].astype('float32')
    gyro_data = window_data[gyro_cols].astype('float32')
    
    # Compute aggregated statistics for accelerometer data.
    acc_mean = acc_data.mean().astype('float32')
    acc_std = acc_data.std().astype('float32')
    acc_max = acc_data.max().astype('float32')
    acc_min = acc_data.min().astype('float32')
    
    # Compute aggregated statistics for gyroscope data.
    gyro_mean = gyro_data.mean().astype('float32')
    gyro_std = gyro_data.std().astype('float32')
    gyro_max = gyro_data.max().astype('float32')
    gyro_min = gyro_data.min().astype('float32')
    
    # Calculate magnitudes in a vectorized way.
    acc_mag = (acc_data ** 2).sum(axis=1).pow(0.5)
    gyro_mag = (gyro_data ** 2).sum(axis=1).pow(0.5)
    
    # Compute aggregated statistics for magnitudes.
    acc_mag_mean = acc_mag.mean()
    acc_mag_std = acc_mag.std()
    acc_mag_max = acc_mag.max()
    acc_mag_min = acc_mag.min()
    
    gyro_mag_mean = gyro_mag.mean()
    gyro_mag_std = gyro_mag.std()
    gyro_mag_max = gyro_mag.max()
    gyro_mag_min = gyro_mag.min()
    
    # Determine the most common label in the window.
    label = window_data['label'].mode().iloc[0]
    
    # Combine all results into a flat dictionary.
    features = {}
    for col in acc_cols:
        features[f'{col}_mean'] = acc_mean[col]
        features[f'{col}_std'] = acc_std[col]
        features[f'{col}_max'] = acc_max[col]
        features[f'{col}_min'] = acc_min[col]
    for col in gyro_cols:
        features[f'{col}_mean'] = gyro_mean[col]
        features[f'{col}_std'] = gyro_std[col]
        features[f'{col}_max'] = gyro_max[col]
        features[f'{col}_min'] = gyro_min[col]
    
    features['acc_mag_mean'] = acc_mag_mean
    features['acc_mag_std'] = acc_mag_std
    features['acc_mag_max'] = acc_mag_max
    features['acc_mag_min'] = acc_mag_min
    
    features['gyro_mag_mean'] = gyro_mag_mean
    features['gyro_mag_std'] = gyro_mag_std
    features['gyro_mag_max'] = gyro_mag_max
    features['gyro_mag_min'] = gyro_mag_min
    
    features['label'] = label
    
    return features



def process_data_with_windows(data, window_size=10):
    """Process data using sliding windows"""
    processed_data = []
    
    # Ensure data is sorted by timestamp
    data = data.sort_values('timestamp')
    
    # Create windows with 50% overlap
    step_size = 3*window_size // 4
    
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i:i + window_size]
        features = extract_features(window)
        # Convert labels to binary (fall=1, no-fall=0)
        features['label'] = 1 if features['label'] == 5 else 0
        processed_data.append(features)
    
    return pd.DataFrame(processed_data)

def train_model():
    # Load the data
    print("Please select your training data CSV file...")
    data = load_data("Select training data CSV file")
    
    if data is None:
        return None
    
    # Process data with time windows
    print("Processing data with time windows...")
    processed_data = process_data_with_windows(data)
    
    if len(processed_data) == 0:
        print("Error: No features extracted from the data!")
        return None
    
    # Split features and labels
    X = processed_data.drop('label', axis=1)
    y = processed_data['label']
    
    # Check class distribution
    fall_count = (y == 1).sum()
    no_fall_count = (y == 0).sum()
    print(f"\nClass distribution:")
    print(f"Fall samples: {fall_count}")
    print(f"No-fall samples: {no_fall_count}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train the Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=300,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    rf_model.fit(X_train, y_train)
    
    # Calculate training accuracy
    train_accuracy = rf_model.score(X_train, y_train)
    test_accuracy = rf_model.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "fall_detection_model.joblib")
    dump(rf_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return rf_model, X_test, y_test

def evaluate_model(model=None, test_data=None, y_true=None):
    if model is None:
        # Load the model
        print("Please select the model.joblib file...")
        root = tk.Tk()
        root.withdraw()
        model_path = filedialog.askopenfilename(
            title="Select model.joblib file",
            filetypes=[("Joblib files", "*.joblib")]
        )
        if not model_path:
            print("No model file selected. Exiting...")
            return
        model = load(model_path)
        
        # Load test data
        print("Please select the test data CSV file...")
        data = load_data("Select test data CSV file")
        if data is None:
            return
        
        # Process data with time windows
        print("Processing test data with time windows...")
        processed_data = process_data_with_windows(data)
        
        # Prepare data
        test_data = processed_data.drop('label', axis=1)
        y_true = processed_data['label']
    
    # Make predictions
    y_pred = model.predict(test_data)
    y_prob = model.predict_proba(test_data)[:, 1]  # Probability of fall
    
    # Calculate metrics with zero_division parameter
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Calculate fall-specific metrics
    true_falls = (y_true == 1).sum()
    predicted_falls = (y_pred == 1).sum()
    correct_falls = sum((y_true == 1) & (y_pred == 1))
    
    fall_accuracy = (correct_falls / true_falls * 100) if true_falls > 0 else 0
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Fall Detection Accuracy: {fall_accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall (Sensitivity): {recall:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    
    # Calculate specificity
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = (tn / (tn + fp)) * 100
    print(f"Specificity: {specificity:.2f}%")
    
    # Print detailed fall detection statistics
    print("\nDetailed Fall Detection Statistics:")
    print(f"Total actual falls: {true_falls}")
    print(f"Total predicted falls: {predicted_falls}")
    print(f"Correctly predicted falls: {correct_falls}")
    print(f"Missed falls (False Negatives): {fn}")
    print(f"False alarms (False Positives): {fp}")
    print(f"Fall detection rate: {fall_accuracy:.2f}%")
    print(f"False alarm rate: {(fp / (fp + tn) * 100):.2f}%")
    
    # Print prediction distribution
    print("\nPrediction Distribution:")
    print("Falls (1):")
    print(f"  True falls: {true_falls}")
    print(f"  Predicted falls: {predicted_falls}")
    print(f"  Correctly predicted: {correct_falls} ({fall_accuracy:.2f}%)")
    print("No Falls (0):")
    print(f"  True no-falls: {(y_true == 0).sum()}")
    print(f"  Predicted no-falls: {(y_pred == 0).sum()}")
    print(f"  Correctly predicted: {tn} ({(tn / (y_true == 0).sum() * 100):.2f}%)")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    labels = ['No Fall', 'Fall']
    plt.xticks([0.5, 1.5], labels)
    plt.yticks([0.5, 1.5], labels)
    plt.show()
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def main():
    while True:
        print("\nRandom Forest Model Trainer and Evaluator")
        print("1. Train new model")
        print("2. Evaluate existing model")
        print("3. Train and evaluate immediately")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            train_model()
        elif choice == '2':
            evaluate_model()
        elif choice == '3':
            model, X_test, y_test = train_model()
            if model is not None:
                evaluate_model(model, X_test, y_test)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()