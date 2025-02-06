import os
import json
from datetime import datetime

import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

# For model loading and prediction
from tensorflow.keras.models import load_model as keras_load_model
import numpy as np
from scipy.signal import savgol_filter

# Conversion and filter constants (as used in your labeling script)
ACC_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0
G_MS2 = 9.81
WINDOW_LENGTH_FILTER = 11  # window length for the Savitzky-Golay filter
POLY_ORDER = 2

# Window size for prediction (should match what you used in training)
WINDOW_SIZE = 20
# Sensor columns in the expected order
SENSOR_COLS = ['acceleration_x', 'acceleration_y', 'acceleration_z',
               'gyroscope_x', 'gyroscope_y', 'gyroscope_z']

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("./src/index.html", "r") as f:
    html = f.read()

# Global sensor data buffer to accumulate raw samples
sensor_data_buffer = []

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
        df.to_csv(
            self.file_path,
            index=False,
            mode="a",
            header=not os.path.exists(self.file_path),
        )

data_processor = DataProcessor()

# ------------------ Model Loading and Prediction ------------------

def load_model():
    """Loads the saved Keras model."""
    model_path = "fall_detection_model.h5"
    try:
        model = keras_load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_window(window: np.ndarray) -> np.ndarray:
    """
    Applies the same offset correction and filtering to a window of sensor data.
    Expected shape of window: (WINDOW_SIZE, 6)
    """
    # Copy and ensure float type for calculations
    processed = window.copy().astype(float)
    
    # Convert accelerometer values from raw units to m/s^2
    processed[:, :3] = (processed[:, :3] / ACC_LSB_PER_G) * G_MS2
    # Convert gyroscope values from raw units to °/s
    processed[:, 3:] = processed[:, 3:] / GYRO_LSB_PER_DPS
    
    # Apply Savitzky-Golay filter on each channel
    for col in range(processed.shape[1]):
        processed[:, col] = savgol_filter(processed[:, col], WINDOW_LENGTH_FILTER, POLY_ORDER)
    return processed

def predict_label(model=None, new_data=None):
    """
    Processes incoming sensor data sample-by-sample.
    Applies conversion and filtering to the sliding window before prediction.
    
    new_data: dictionary containing one sensor sample with keys matching SENSOR_COLS.
    """
    global sensor_data_buffer
    if model is None or new_data is None:
        return 0

    try:
        # Extract sensor values from the incoming JSON data.
        sample = [float(new_data[col]) for col in SENSOR_COLS]
    except KeyError:
        print("Incoming data does not have the required sensor fields.")
        return 0

    # Append the new sample to the global buffer
    sensor_data_buffer.append(sample)

    # Only predict once we have accumulated at least WINDOW_SIZE samples
    if len(sensor_data_buffer) >= WINDOW_SIZE:
        # Use the most recent WINDOW_SIZE samples
        window = np.array(sensor_data_buffer[-WINDOW_SIZE:])
        # Apply offset correction and filtering
        processed_window = preprocess_window(window)
        # Reshape to match model input: (1, WINDOW_SIZE, number_of_features)
        input_data = processed_window.reshape((1, WINDOW_SIZE, len(SENSOR_COLS)))
        prediction = model.predict(input_data)
        # Extract predicted label from softmax output
        label = int(np.argmax(prediction, axis=1)[0])
        return label
    else:
        return None

# ------------------------------------------------------------------

class WebSocketManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print("WebSocket connected")

    def disconnect(self, websocket):
        self.active_connections.remove(websocket)
        print("WebSocket disconnected")

    async def broadcast_message(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(connection)

websocket_manager = WebSocketManager()
model = load_model()  # Load the model when the server starts

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            json_data = json.loads(data)
            # Add a timestamp for logging
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_processor.add_data(json_data)
            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()

            # Process the new sensor sample and predict when enough data has accumulated.
            label = predict_label(model, json_data)
            if label is not None:
                json_data["label"] = label
            else:
                json_data["label"] = "Waiting for enough data..."

            print(json_data)
            await websocket_manager.broadcast_message(json.dumps(json_data))
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
