import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tkinter as tk
from tkinter.filedialog import askopenfilename
import joblib

#########################
# Feature Extraction Functions
#########################

def calculate_magnitudes(window_data):
    """Calculate magnitudes for accelerometer and gyroscope data."""
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
    """Extract features from a time window.
    
    Computes statistical features for each sensor axis and for the overall magnitude.
    If a 'label' column exists, the mode of the labels in the window is computed.
    """
    acc_mag, gyro_mag = calculate_magnitudes(window_data)
    
    features = {}
    
    # Statistical features for raw accelerometer data
    for axis in ['acceleration_x', 'acceleration_y', 'acceleration_z']:
        features[f'{axis}_mean'] = window_data[axis].mean()
        features[f'{axis}_std']  = window_data[axis].std()
        features[f'{axis}_max']  = window_data[axis].max()
        features[f'{axis}_min']  = window_data[axis].min()
    
    # Statistical features for raw gyroscope data
    for axis in ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']:
        features[f'{axis}_mean'] = window_data[axis].mean()
        features[f'{axis}_std']  = window_data[axis].std()
        features[f'{axis}_max']  = window_data[axis].max()
        features[f'{axis}_min']  = window_data[axis].min()
    
    # Statistical features for magnitudes
    features['acc_mag_mean'] = acc_mag.mean()
    features['acc_mag_std']  = acc_mag.std()
    features['acc_mag_max']  = acc_mag.max()
    features['acc_mag_min']  = acc_mag.min()
    
    features['gyro_mag_mean'] = gyro_mag.mean()
    features['gyro_mag_std']  = gyro_mag.std()
    features['gyro_mag_max']  = gyro_mag.max()
    features['gyro_mag_min']  = gyro_mag.min()
    
    # Only add a label if available (used during training/evaluation)
    if 'label' in window_data.columns:
        features['label'] = window_data['label'].mode().iloc[0]
    
    return features

def process_data_with_windows(data, window_size=135):
    """Process data using sliding windows with 50% overlap.
    
    The window’s center index is stored for plotting purposes.
    If a 'label' exists, it is converted to binary (fall=1 if label==5, else 0).
    """
    processed_data = []
    centers = []
    
    # Ensure data is sorted by timestamp
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    # Use 50% overlap: step size = window_size // 2
    step_size = window_size // 4
    
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i:i + window_size]
        features = extract_features(window)
        # Save the center index of this window for plotting later
        center = i + window_size // 2
        features['center'] = center
        
        # If label is present, convert to binary (as in training)
        if 'label' in features:
            features['label'] = 1 if features['label'] == 5 else 0
        processed_data.append(features)
    
    return pd.DataFrame(processed_data)

#########################
# Main Prediction & Visualization
#########################

def main():
    # Initialize Tkinter and hide the main window.
    root = tk.Tk()
    root.withdraw()
    
    # Ask the user to select the saved Random Forest .joblib model file.
    model_path = askopenfilename(
        title="Select Random Forest .joblib model file",
        filetypes=[("Joblib Files", "*.joblib")]
    )
    if not model_path:
        print("No model file selected. Exiting.")
        return
    
    model = joblib.load(model_path)
    print("Model loaded from:", model_path)
    
    # Ask the user to select the CSV file with sensor data.
    csv_path = askopenfilename(
        title="Select CSV file for prediction",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not csv_path:
        print("No CSV file selected. Exiting.")
        return
    
    # Load CSV data (expects at least: timestamp, acceleration_*, gyroscope_* columns)
    df = pd.read_csv(csv_path, parse_dates=['timestamp'], infer_datetime_format=True)
    
    # Process the data using sliding windows
    processed_df = process_data_with_windows(df, window_size=125)
    if processed_df.empty:
        print("No windows were extracted. Please check your data and window size.")
        return
    
    # Extract window centers (for plotting) and feature data.
    window_centers = processed_df['center']
    
    # Remove non-feature columns ('center' and 'label' if present)
    X = processed_df.drop(columns=['center', 'label'], errors='ignore')
    
    # Make predictions using the loaded model.
    predicted_labels = model.predict(X)
    print("Predicted labels for each window:")
    print(predicted_labels)
    
    #########################
    # Visualization
    #########################
    
    # Define colors for predictions: here, 0 (no-fall) is blue, 1 (fall) is red.
    color_mapping = {0: 'blue', 1: 'red'}
    pred_colors = [color_mapping.get(label, 'black') for label in predicted_labels]
    
    # Create x-axis for the full sensor data (using DataFrame index)
    x_full = range(len(df))
    
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Accelerometer data.
    plt.subplot(3, 1, 1)
    plt.plot(x_full, df['acceleration_x'], label='Acc X')
    plt.plot(x_full, df['acceleration_y'], label='Acc Y')
    plt.plot(x_full, df['acceleration_z'], label='Acc Z')
    plt.title('Accelerometer Data')
    plt.xlabel('Data Index')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Gyroscope data.
    plt.subplot(3, 1, 2)
    plt.plot(x_full, df['gyroscope_x'], label='Gyro X')
    plt.plot(x_full, df['gyroscope_y'], label='Gyro Y')
    plt.plot(x_full, df['gyroscope_z'], label='Gyro Z')
    plt.title('Gyroscope Data')
    plt.xlabel('Data Index')
    plt.ylabel('Angular Velocity')
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Predicted Fall Labels (window-based).
    plt.subplot(3, 1, 3)
    plt.scatter(window_centers, predicted_labels, c=pred_colors, s=50, marker='o')
    plt.title('Predicted Fall Labels (Window Centers)')
    plt.xlabel('Data Index (Window Center)')
    plt.ylabel('Predicted Label (0 = No Fall, 1 = Fall)')
    
    # Build a custom legend.
    legend_handles = []
    for label, color in color_mapping.items():
        patch = mpatches.Patch(color=color, label=f'Label {label}')
        legend_handles.append(patch)
    plt.legend(handles=legend_handles)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()