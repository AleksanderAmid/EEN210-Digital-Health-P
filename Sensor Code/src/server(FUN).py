import os
import json
from datetime import datetime

import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()
# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("./src/index.html", "r") as f:
    html = f.read()


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
        # Append the new row to the existing DataFrame
        df.to_csv(
            self.file_path,
            index=False,
            mode="a",
            header=not os.path.exists(self.file_path),
        )
        # print(f"DataFrame saved to {self.file_path}")


data_processor = DataProcessor()


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

def extract_features(window_data):
    """Extract statistical features from a window of sensor data."""
    acc_mag, gyro_mag = calculate_magnitudes(window_data)
    features = {}
    for axis in ['acceleration_x', 'acceleration_y', 'acceleration_z']:
        features[f'{axis}_mean'] = window_data[axis].mean()
        features[f'{axis}_std'] = window_data[axis].std()
        features[f'{axis}_max'] = window_data[axis].max()
        features[f'{axis}_min'] = window_data[axis].min()
    for axis in ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']:
        features[f'{axis}_mean'] = window_data[axis].mean()
        features[f'{axis}_std'] = window_data[axis].std()
        features[f'{axis}_max'] = window_data[axis].max()
        features[f'{axis}_min'] = window_data[axis].min()
    features['acc_mag_mean'] = acc_mag.mean()
    features['acc_mag_std'] = acc_mag.std()
    features['acc_mag_max'] = acc_mag.max()
    features['acc_mag_min'] = acc_mag.min()
    features['gyro_mag_mean'] = gyro_mag.mean()
    features['gyro_mag_std'] = gyro_mag.std()
    features['gyro_mag_max'] = gyro_mag.max()
    features['gyro_mag_min'] = gyro_mag.min()
    return features


def load_model():
    """Load and return the saved Random Forest model."""
    model_path = "src\\Model_135_25_72.joblib"  # Use double backslashes
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded from:", model_path)
        return model
    else:
        print("Model file not found.")
        return None


def predict_label(model, data_buffer, window_size=135):
    """
    Form a window using the last window_size samples,
    extract features and predict using the loaded model.
    Returns 1 if a fall is predicted; otherwise 0.
    """
    if len(data_buffer) < window_size:
        print("Debug: Not enough samples for prediction")
        return 0

    # Create a DataFrame from the most recent window_size samples
    window_df = pd.DataFrame(data_buffer[-window_size:])
    if 'timestamp' in window_df.columns:
        window_df = window_df.sort_values('timestamp').reset_index(drop=True)

    # Extract features using the defined function
    features = extract_features(window_df)
    X = pd.DataFrame([features])
    try:
        # Predict label using the pre-loaded model
        label = model.predict(X)[0]
        print(f"Debug: Predicted label from features: {label}")
        return label
    except Exception as e:
        print("Debug: Error during prediction:", str(e))
        return 0


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
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                # Handle disconnect if needed
                self.disconnect(connection)


websocket_manager = WebSocketManager()
model = load_model()


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
            # Add a timestamp
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_processor.add_data(json_data)
            
            # Once we have enough samples, do prediction
            label = 0
            if len(data_processor.data_buffer) >= 135:
                label = predict_label(model, data_processor.data_buffer, window_size=135)
            json_data["label"] = int(label)
            
            # Print entire JSON data including label
            print(json_data)
            
            # Create a new dictionary with only timestamp and label
            minimal_data = {"timestamp": json_data["timestamp"], "label": json_data["label"]}
            
            # Broadcast minimal data to clients
            await websocket_manager.broadcast_message(json.dumps(minimal_data))
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)