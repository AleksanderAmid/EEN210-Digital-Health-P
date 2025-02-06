import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model as keras_load_model

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
        self.file_path = f"laying_up_data_{timestamp}.csv"

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


def load_model():
    """
    Load the saved Keras model from the .h5 file.
    """
    try:
        model = keras_load_model('fall_detection_model.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    return model


def predict_label(model, _data):
    """
    Predict the label using the provided model.
    
    This function waits until at least 20 sensor samples have been accumulated in the data buffer.
    It then extracts the sensor values from the last 20 entries, forms a sequence, and runs the prediction.
    """
    # Define the sensor keys as used during training.
    sensor_keys = [
        'acceleration_x', 'acceleration_y', 'acceleration_z',
        'gyroscope_x', 'gyroscope_y', 'gyroscope_z'
    ]
    window_size = 20  # Must match the training window_size

    # Check if there are enough samples to form a sequence.
    if len(data_processor.data_buffer) < window_size:
        # Not enough data yet to form a complete sequence.
        return None

    # Extract the last `window_size` sensor samples from the data buffer.
    window = data_processor.data_buffer[-window_size:]
    sequence = []
    for entry in window:
        try:
            sensor_values = [float(entry[key]) for key in sensor_keys]
            sequence.append(sensor_values)
        except KeyError:
            # In case some keys are missing, skip prediction.
            print("Missing sensor keys in data entry; skipping prediction.")
            return None

    # Convert to NumPy array and add the batch dimension.
    sequence = np.array(sequence)  # shape: (20, 6)
    sequence = np.expand_dims(sequence, axis=0)  # shape: (1, 20, 6)

    # Run prediction
    predictions = model.predict(sequence)
    label = int(np.argmax(predictions, axis=1)[0])
    return label


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

            # Add time stamp to the received data.
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Add incoming data to our buffer.
            data_processor.add_data(json_data)
            # Optionally save to CSV every 100 samples.
            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()

            # Run prediction using the current data buffer.
            label = predict_label(model, json_data)
            # If not enough data, you may choose to keep the label empty or set a default.
            json_data["label"] = label if label is not None else "pending"

            # Print the last data to the terminal.
            print(json_data)

            # Broadcast the data (with label) to connected clients.
            await websocket_manager.broadcast_message(json.dumps(json_data))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
