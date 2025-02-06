import pandas as pd
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

def load_csv_data(csv_path, sensor_cols):
    # Read CSV file, parsing timestamps if present, and drop rows with missing sensor data.
    df = pd.read_csv(csv_path, parse_dates=['timestamp'], infer_datetime_format=True)
    df = df.dropna(subset=sensor_cols)
    return df

def preprocess_data(df, sensor_cols):
    # Standardize sensor data using StandardScaler.
    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    return df

def create_sequences(df, sensor_cols, window_size):
    # Create overlapping sequences from sensor data.
    data = df[sensor_cols].values
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

def main():
    # Define sensor columns and window size for sequences.
    sensor_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                   'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    window_size = 20

    # Initialize Tkinter and hide the root window.
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select the pre-trained Keras .h5 model file.
    model_path = askopenfilename(
        title="Select Keras .h5 model file",
        filetypes=[("H5 Files", "*.h5")]
    )
    if not model_path:
        print("No model file selected. Exiting.")
        return

    model = load_model(model_path)
    print("Model loaded from:", model_path)

    # Ask the user to select the CSV file containing sensor data.
    csv_path = askopenfilename(
        title="Select CSV file for prediction",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not csv_path:
        print("No CSV file selected. Exiting.")
        return

    # Load and preprocess the sensor data.
    df = load_csv_data(csv_path, sensor_cols)
    df = preprocess_data(df, sensor_cols)
    
    # Create sequences for prediction.
    sequences = create_sequences(df, sensor_cols, window_size)
    
    # Generate predictions on all sequences.
    predictions = model.predict(sequences)
    predicted_labels = predictions.argmax(axis=1)
    print("Predicted labels:")
    print(predicted_labels)
    
    # Create an x-axis for the predictions by taking the sequence center index.
    x_seq = [i + window_size // 2 for i in range(len(sequences))]
    
    # Define a color mapping for predicted labels.
    color_mapping = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple'}
    pred_colors = [color_mapping.get(label, 'black') for label in predicted_labels]
    
    # Create an x-axis for the full CSV data (using the DataFrame index).
    x_full = range(len(df))
    
    # Create a figure with three subplots: accelerometer data, gyroscope data, and predicted labels.
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Plot Accelerometer Data.
    plt.subplot(3, 1, 1)
    plt.plot(x_full, df['acceleration_x'], label='Acc X')
    plt.plot(x_full, df['acceleration_y'], label='Acc Y')
    plt.plot(x_full, df['acceleration_z'], label='Acc Z')
    plt.title('Accelerations')
    plt.xlabel('Index')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Plot Gyroscope Data.
    plt.subplot(3, 1, 2)
    plt.plot(x_full, df['gyroscope_x'], label='Gyro X')
    plt.plot(x_full, df['gyroscope_y'], label='Gyro Y')
    plt.plot(x_full, df['gyroscope_z'], label='Gyro Z')
    plt.title('Gyroscope')
    plt.xlabel('Index')
    plt.ylabel('Angular Velocity')
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Plot Predicted Labels.
    plt.subplot(3, 1, 3)
    # Scatter plot where each point is colored based on the predicted label.
    plt.scatter(x_seq, predicted_labels, c=pred_colors, s=50, marker='o')
    plt.title('Predicted Labels')
    plt.xlabel('Index (Sequence Center)')
    plt.ylabel('Label')
    
    # Create a custom legend for the predicted labels.
    legend_handles = []
    for label, color in color_mapping.items():
        if label in predicted_labels:
            patch = mpatches.Patch(color=color, label=f'Label {label}')
            legend_handles.append(patch)
    plt.legend(handles=legend_handles)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
