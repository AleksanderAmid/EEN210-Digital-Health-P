import pandas as pd
import numpy as np
import tkinter as tk
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tkinter.filedialog

def load_csv_data(csv_path, sensor_cols):
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    # Drop rows with missing sensor data
    df = df.dropna(subset=sensor_cols)
    return df

def preprocess_data(df, sensor_cols):
    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    return df

def create_sequences(df, sensor_cols, window_size):
    data = df[sensor_cols].values
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i+window_size])
    sequences = np.array(sequences)
    return sequences

def main():
    sensor_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                   'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    window_size = 20

    root = tk.Tk()
    root.withdraw()

    # Ask the user to select the pre-trained .h5 model file.
    model_path = tkinter.filedialog.askopenfilename(
        title="Select Keras .h5 model file",
        filetypes=[("H5 Files", "*.h5")]
    )
    if not model_path:
        print("No model file selected. Exiting.")
        return

    model = load_model(model_path)
    print("Model loaded from:", model_path)

    # Ask the user to select a CSV file containing new sensor data.
    csv_path = tkinter.filedialog.askopenfilename(
        title="Select CSV file for prediction",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not csv_path:
        print("No CSV file selected. Exiting.")
        return

    df = load_csv_data(csv_path, sensor_cols)
    df = preprocess_data(df, sensor_cols)
    sequences = create_sequences(df, sensor_cols, window_size)

    # Generate predictions on all sequences.
    predictions = model.predict(sequences)
    predicted_labels = predictions.argmax(axis=1)
    
    print("Predicted labels:")
    print(predicted_labels)

    # Ask the user to select the CSV file containing the real labels.
    label_csv_path = tkinter.filedialog.askopenfilename(
        title="Select CSV file containing real labels",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not label_csv_path:
        print("No real label CSV file selected. Exiting.")
        return

    # Assume the CSV has a column named 'label'
    label_df = pd.read_csv(label_csv_path)
    # Drop rows with missing values in the 'label' column to avoid NaNs
    label_df = label_df.dropna(subset=['label'])
    real_labels = label_df["label"].values

    # In case the number of labels and predictions differ, take the common length.
    common_length = min(len(predicted_labels), len(real_labels))
    predicted_labels = predicted_labels[:common_length]
    real_labels = real_labels[:common_length]

    accuracy = accuracy_score(real_labels, predicted_labels)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

if __name__ == "__main__":
    main()
