# train_model.py
import pandas as pd
import tkinter as tk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
import json  # Used for saving the training history


class ModelTrainer:
    def __init__(self, csv_path, window_size=20):
        self.csv_path = csv_path
        self.window_size = window_size
        self.sensor_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                            'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    
    def load_and_preprocess_data(self):
        df = pd.read_csv(self.csv_path, parse_dates=['timestamp'])
        # Handle missing labels and convert label to numeric if necessary.
        df = df.dropna(subset=['label'])
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label'])
        
        scaler = StandardScaler()
        df[self.sensor_cols] = scaler.fit_transform(df[self.sensor_cols])
        return df
    
    def create_sequences(self, df):
        X, y = [], []
        data = df[self.sensor_cols].values
        labels = df['label'].values
        for i in range(len(data) - self.window_size + 1):
            X.append(data[i:i+self.window_size])
            center_label = labels[i + self.window_size // 2]
            y.append(center_label)
        X = np.array(X)
        y = np.array(y)
        return X, y

    def build_model(self, num_classes):
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.window_size, len(self.sensor_cols)), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def train(self, epochs=20, batch_size=32):
        df = self.load_and_preprocess_data()
        X, y = self.create_sequences(df)
        # Determine number of classes based on maximum label value.
        num_classes = int(np.max(y)) + 1  
        y_cat = to_categorical(y, num_classes=num_classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
        
        model = self.build_model(num_classes)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        model.save('fall_detection_model.h5')
        
        # Save the training history to a JSON file.
        with open("training_history.json", "w") as f:
            json.dump(history.history, f)
        
        return model, history
    
# run the script
if __name__ == "__main__":
    import tkinter.filedialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = tkinter.filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if file_path:
        trainer = ModelTrainer(file_path, window_size=20)
        model, history = trainer.train(epochs=40, batch_size=32)
        print("Model training complete.")
        print("Training history saved to 'training_history.json'.")
    else:
        print("No file selected. Exiting.")
