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

def load_data(title="Select CSV file"):
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("CSV files", "*.csv")]
    )
    
    if not file_path:
        print("No data file selected. Exiting...")
        return None
        
    return pd.read_csv(file_path)

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
    """Extract features from a time window"""
    acc_mag, gyro_mag = calculate_magnitudes(window_data)
    
    features = {}
    
    # Statistical features for raw accelerometer data
    for axis in ['acceleration_x', 'acceleration_y', 'acceleration_z']:
        features[f'{axis}_mean'] = window_data[axis].mean()
        features[f'{axis}_std'] = window_data[axis].std()
        features[f'{axis}_max'] = window_data[axis].max()
        features[f'{axis}_min'] = window_data[axis].min()
    
    # Statistical features for raw gyroscope data
    for axis in ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']:
        features[f'{axis}_mean'] = window_data[axis].mean()
        features[f'{axis}_std'] = window_data[axis].std()
        features[f'{axis}_max'] = window_data[axis].max()
        features[f'{axis}_min'] = window_data[axis].min()
    
    # Statistical features for magnitudes
    features['acc_mag_mean'] = acc_mag.mean()
    features['acc_mag_std'] = acc_mag.std()
    features['acc_mag_max'] = acc_mag.max()
    features['acc_mag_min'] = acc_mag.min()
    
    features['gyro_mag_mean'] = gyro_mag.mean()
    features['gyro_mag_std'] = gyro_mag.std()
    features['gyro_mag_max'] = gyro_mag.max()
    features['gyro_mag_min'] = gyro_mag.min()
    
    # Most common label in the window
    features['label'] = window_data['label'].mode().iloc[0]
    
    return features

def process_data_with_windows(data, window_size=30):
    """Process data using sliding windows"""
    processed_data = []
    
    # Ensure data is sorted by timestamp
    data = data.sort_values('timestamp')
    
    # Create windows with 50% overlap
    step_size = window_size // 2
    
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i:i + window_size]
        features = extract_features(window)
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
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
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
    model_path = os.path.join(current_dir, "model.joblib")
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
    
    # Calculate metrics with zero_division parameter
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    
    # Print prediction distribution
    print("\nPrediction Distribution:")
    unique_labels = np.unique(y_true)
    for label in unique_labels:
        pred_count = (y_pred == label).sum()
        true_count = (y_true == label).sum()
        print(f"Label {label}:")
        print(f"  True count: {true_count}")
        print(f"  Predicted count: {pred_count}")
        print(f"  Accuracy: {(pred_count/true_count)*100:.2f}%")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
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