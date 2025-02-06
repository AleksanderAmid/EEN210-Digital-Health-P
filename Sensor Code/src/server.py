import os
import json
from datetime import datetime

import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

# Import for model loading and prediction
from tensorflow.keras.models import load_model as keras_load_model
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

# Global sensor buffer for accumulating samples (for a sliding window)
sensor_data_buffer = []
WINDOW_SIZE = 20  # should match your training window size
SENSOR_COLS = ['acceleration_x', 'acceleration_y', 'acceleration_z',
               'gyroscope_x', 'gyroscope_y', 'gyroscope_z']

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
    """
    Loads the saved Keras model.
    """
    model_path = "fall_detection_model.h5"
    try:
        model = keras_load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_label(model=None, new_data=None):
    """
    Collects incoming sensor data into a sliding window and predicts the label once enough samples have been received.
    
    new_data: dictionary containing a single sensor sample.
    """
    global sensor_data_buffer
    if model is None or new_data is None:
        return 0

    # Extract sensor values from the incoming JSON data.
    # It is assumed that new_data contains keys matching SENSOR_COLS.
    try:
        sensor_values = [float(new_data[col]) for col in SENSOR_COLS]
    except KeyError:
        print("Incoming data does not have the required sensor fields.")
        return 0

    # Append the new sample to the global sensor data buffer
    sensor_data_buffer.append(sensor_values)

    # If we have accumulated enough samples, prepare input for prediction
    if len(sensor_data_buffer) >= WINDOW_SIZE:
        # Use the most recent WINDOW_SIZE samples
        window = sensor_data_buffer[-WINDOW_SIZE:]
        # Convert to numpy array and reshape to (1, WINDOW_SIZE, number_of_features)
        window = np.array(window).reshape((1, WINDOW_SIZE, len(SENSOR_COLS)))
        # Run prediction
        prediction = model.predict(window)
        # Get the predicted label (assumes softmax activation and categorical output)
        label = int(np.argmax(prediction, axis=1)[0])
        return label
    else:
        # Not enough data accumulated yet; optionally return a default label or a flag.
        return None

# ------------------------------------------------------------------

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
                self.disconnect(connection)

websocket_manager = WebSocketManager()
model = load_model()  # Load the model when the server starts

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
            # Add timestamp to the received data
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_processor.add_data(json_data)
            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()

            # Run prediction using the incoming sensor sample.
            # Note: predict_label uses a global buffer to accumulate WINDOW_SIZE samples.
            label = predict_label(model, json_data)
            # Only update the JSON if a prediction was made.
            if label is not None:
                json_data["label"] = label
            else:
                json_data["label"] = "Insufficient data for prediction"

            print(json_data)
            await websocket_manager.broadcast_message(json.dumps(json_data))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
