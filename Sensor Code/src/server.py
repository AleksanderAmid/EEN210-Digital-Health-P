import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from joblib import load

# ---------------------------
# FastAPI and CORS Setup
# ---------------------------
app = FastAPI()
# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("Sensor Code\src\index.html", "r") as f:
    html = f.read()

# ---------------------------
# Data Buffer and CSV Saving
# ---------------------------
class DataProcessor:
    def __init__(self):
        self.data_buffer = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = f"fall_data_{timestamp}.csv"

    def add_data(self, data):
        self.data_buffer.append(data)

    def save_to_csv(self):
        df = pd.DataFrame.from_dict(self.data_buffer)
        self.data_buffer = []
        # Append to CSV (create header if file doesn't exist)
        df.to_csv(
            self.file_path,
            index=False,
            mode="a",
            header=not os.path.exists(self.file_path),
        )
        # Uncomment for debugging:
        # print(f"Data saved to {self.file_path}")

data_processor = DataProcessor()

# ---------------------------
# Helper Functions for Feature Extraction
# ---------------------------
def calculate_magnitudes(window_data):
    """Calculate the magnitude of accelerometer and gyroscope data."""
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

def extract_features_from_window(window_data):
    """
    Extract statistical features from a window of sensor data.
    Expects window_data to have the following columns:
    'acceleration_x', 'acceleration_y', 'acceleration_z',
    'gyroscope_x', 'gyroscope_y', 'gyroscope_z'
    """
    acc_mag, gyro_mag = calculate_magnitudes(window_data)
    features = {}
    
    # Features for accelerometer data
    for axis in ['acceleration_x', 'acceleration_y', 'acceleration_z']:
        features[f'{axis}_mean'] = window_data[axis].mean()
        features[f'{axis}_std'] = window_data[axis].std()
        features[f'{axis}_max'] = window_data[axis].max()
        features[f'{axis}_min'] = window_data[axis].min()
    
    # Features for gyroscope data
    for axis in ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']:
        features[f'{axis}_mean'] = window_data[axis].mean()
        features[f'{axis}_std'] = window_data[axis].std()
        features[f'{axis}_max'] = window_data[axis].max()
        features[f'{axis}_min'] = window_data[axis].min()
    
    # Features for magnitudes
    features['acc_mag_mean'] = acc_mag.mean()
    features['acc_mag_std'] = acc_mag.std()
    features['acc_mag_max'] = acc_mag.max()
    features['acc_mag_min'] = acc_mag.min()
    
    features['gyro_mag_mean'] = gyro_mag.mean()
    features['gyro_mag_std'] = gyro_mag.std()
    features['gyro_mag_max'] = gyro_mag.max()
    features['gyro_mag_min'] = gyro_mag.min()
    
    return features

# ---------------------------
# Model Loading and Prediction Functions
# ---------------------------
def load_model():
    """
    Load and return the trained Random Forest model.
    The model file is assumed to be saved as 'RF_MODEL.joblib'
    in the same directory as this server.py.
    """
    model_path = os.path.join(os.path.dirname(__file__), "Sensor Code\src\Model_135_25_72.joblib)
    if not os.path.exists(model_path):
        print("Model file not found at:", model_path)
        return None
    model = load(model_path)
    print("Model loaded successfully from", model_path)
    return model

def predict_label(model=None, data=None):
    """
    Use the loaded model to predict the fall label (0 for no-fall, 1 for fall)
    based on the most recent window of sensor data.
    
    This function extracts a sliding window (default size 135 samples)
    from the global data_processor buffer, computes features, and returns the prediction.
    If not enough data has been accumulated, it returns 0.
    """
    WINDOW_SIZE = 135  # Must match the window size used during training
    if model is None:
        return 0

    if len(data_processor.data_buffer) < WINDOW_SIZE:
        # Not enough data to form a window; return default label (no fall)
        return 0

    # Get the latest WINDOW_SIZE samples from the data buffer
    window_data = data_processor.data_buffer[-WINDOW_SIZE:]
    df_window = pd.DataFrame(window_data)
    
    # Define the required sensor columns
    required_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                        'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    
    # Check if all required columns are present
    if not set(required_columns).issubset(df_window.columns):
        print("Warning: Not all required columns are present for prediction.")
        return 0

    # Extract features using only the required columns
    features = extract_features_from_window(df_window[required_columns])
    X = pd.DataFrame([features])
    
    # Predict using the loaded model
    predicted = model.predict(X)
    return int(predicted[0])

# ---------------------------
# WebSocket Manager
# ---------------------------
class WebSocketManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print("WebSocket connected")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print("WebSocket disconnected")

    async def broadcast_message(self, message: str):
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(connection)

websocket_manager = WebSocketManager()

# Load the trained model once at startup
model = load_model()

# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            json_data = json.loads(data)

            # Add current timestamp to the data
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add the received data to our data buffer
            data_processor.add_data(json_data)
            
            # Save the data buffer to CSV every 100 samples (adjustable)
            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()
            
            # Use the accumulated data to predict the label.
            # Note: The prediction is based on the most recent window of data.
            label = predict_label(model, None)
            json_data["label"] = label

            # Log the data with prediction to the terminal
            print(json_data)

            # Broadcast the updated data to all connected clients
            await websocket_manager.broadcast_message(json.dumps(json_data))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

# ---------------------------
# Main: Start the Server
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
