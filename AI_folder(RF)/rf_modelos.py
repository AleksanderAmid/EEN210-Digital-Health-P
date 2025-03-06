import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the raw data from CSV; update the file path as needed.
raw_data = pd.read_csv("LabeledData/Old label/Merged Data/Merged_Data_11-38-59.csv")
raw_data['label'] = raw_data['label'].fillna(0)

def extract_features_sliding(data, window_size=135, step=67):
    """
    Extract features over multiple sliding windows.
    Each window will generate a row of features.
    The features here must match those used during training.
    """
    windows = []
    n_samples = len(data)
    for start in range(0, n_samples - window_size + 1, step):
        window = data.iloc[start:start+window_size].copy()
        
        # Compute acceleration and gyroscope magnitudes.
        window['acc_mag'] = np.sqrt(window['acceleration_x']**2 +
                                    window['acceleration_y']**2 +
                                    window['acceleration_z']**2)
        window['gyro_mag'] = np.sqrt(window['gyroscope_x']**2 +
                                     window['gyroscope_y']**2 +
                                     window['gyroscope_z']**2)
        
        features = {
            # Raw axis features (if used during training)
            'acceleration_x_mean': window['acceleration_x'].mean(),
            'acceleration_x_std':  window['acceleration_x'].std(),
            'acceleration_x_max':  window['acceleration_x'].max(),
            'acceleration_x_min':  window['acceleration_x'].min(),
            'acceleration_y_mean': window['acceleration_y'].mean(),
            'acceleration_y_std':  window['acceleration_y'].std(),
            'acceleration_y_max':  window['acceleration_y'].max(),
            'acceleration_y_min':  window['acceleration_y'].min(),
            'acceleration_z_mean': window['acceleration_z'].mean(),
            'acceleration_z_std':  window['acceleration_z'].std(),
            'acceleration_z_max':  window['acceleration_z'].max(),
            'acceleration_z_min':  window['acceleration_z'].min(),
            
            'gyroscope_x_mean': window['gyroscope_x'].mean(),
            'gyroscope_x_std':  window['gyroscope_x'].std(),
            'gyroscope_x_max':  window['gyroscope_x'].max(),
            'gyroscope_x_min':  window['gyroscope_x'].min(),
            'gyroscope_y_mean': window['gyroscope_y'].mean(),
            'gyroscope_y_std':  window['gyroscope_y'].std(),
            'gyroscope_y_max':  window['gyroscope_y'].max(),
            'gyroscope_y_min':  window['gyroscope_y'].min(),
            'gyroscope_z_mean': window['gyroscope_z'].mean(),
            'gyroscope_z_std':  window['gyroscope_z'].std(),
            'gyroscope_z_max':  window['gyroscope_z'].max(),
            'gyroscope_z_min':  window['gyroscope_z'].min(),
            
            # Magnitude features (expected by the model)
            'acc_mag_mean': window['acc_mag'].mean(),
            'acc_mag_std': window['acc_mag'].std(),
            'acc_mag_max': window['acc_mag'].max(),
            'acc_mag_min': window['acc_mag'].min(),
            'gyro_mag_mean': window['gyro_mag'].mean(),
            'gyro_mag_std': window['gyro_mag'].std(),
            'gyro_mag_max': window['gyro_mag'].max(),
            'gyro_mag_min': window['gyro_mag'].min(),
        }
        windows.append(features)
    return pd.DataFrame(windows)

# Extract features over sliding windows.
processed_features = extract_features_sliding(raw_data, window_size=135, step=67)

if 'label' in raw_data.columns:
    # Aggregate labels from the window (using the window center for demonstration)
    labels = []
    for start in range(0, len(raw_data) - 135 + 1, 67):
        window = raw_data.iloc[start:start+135].reset_index(drop=True)
        center_label = window['label'].iloc[len(window)//2]
        # Convert to binary: if center_label equals 5, then it's a fall (1), else no-fall (0)
        labels.append(1 if center_label == 5 else 0)
    y = pd.Series(labels)
else:
    raise ValueError("Label column not found in CSV.")

X = processed_features

# Load the pre-trained model (adjust path/filename as needed)
model = load("AI_folder(RF)/fall_detection_model.joblib")

# Now split the dataset:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Assuming your classes are 0 and 1; adjust as needed.
class_names = ['No Fall', 'Fall']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()