import os
import json
from datetime import datetime

import pandas as pd
import uvicorn
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware

# For model loading and prediction
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

# Load the HTML page (for demonstration)
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
        df.to_csv(
            self.file_path,
            index=False,
            mode="a",
            header=not os.path.exists(self.file_path),
        )
        # print(f"DataFrame saved to {self.file_path}")


data_processor = DataProcessor()


# Global variable to store live sensor data for prediction.
# The model was trained on sequences (windows) of size 20 and 6 features.
live_data_buffer = []
WINDOW_SIZE = 20  # Must match the training window size

def load_model():
    """
    Loads the trained Keras model from file.
    Make sure the path is correct.
    """
    model = keras_load_model("fall_detection_model.h5")
    return model


def predict_label(model, data):
    """
    Receives a NumPy array 'data' of shape (1, window_size, 6)
    and returns the predicted label.
    """
    # Get the prediction probabilities.
    prediction = model.predict(data)
    # Choose the label with the highest probability.
    label = int(np.argmax(prediction, axis=1)[0])
    return label


class WebSocketManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print("WebSocket connected")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        print("WebSocket disconnected")

    async def broadcast_message(self, message: str):
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(connection)


websocket_manager = WebSocketManager()
# Load the model once at startup.
model = load_model()
print("Model loaded for live prediction.")


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # For debugging/logging
            print("Received data:", data)
            
            # Parse the incoming JSON data.
            json_data = json.loads(data)

            # Extract sensor data.
            # Ensure that your keys match those sent by your embedded device.
            # Here, we assume the JSON includes the following keys:
            # "acceleration_x", "acceleration_y", "acceleration_z",
            # "gyroscope_x", "gyroscope_y", "gyroscope_z"
            sensor_values = [
                json_data.get("acceleration_x", 0),
                json_data.get("acceleration_y", 0),
                json_data.get("acceleration_z", 0),
                json_data.get("gyroscope_x", 0),
                json_data.get("gyroscope_y", 0),
                json_data.get("gyroscope_z", 0)
            ]
            
            # Append the sensor values to the live buffer.
            live_data_buffer.append(sensor_values)
            
            # If the buffer is larger than the desired window size, keep only the most recent WINDOW_SIZE samples.
            if len(live_data_buffer) > WINDOW_SIZE:
                live_data_buffer.pop(0)
            
            # If we have enough data, perform prediction.
            if len(live_data_buffer) == WINDOW_SIZE:
                # Convert buffer to numpy array with shape (1, WINDOW_SIZE, 6)
                window_array = np.array(live_data_buffer, dtype=np.float32)
                window_array = window_array.reshape((1, WINDOW_SIZE, 6))
                # Get predicted label from the model.
                label = predict_label(model, window_array)
            else:
                # If not enough data, set a default label (e.g., 0 for "no event" or -1 for "insufficient data")
                label = -1

            # Add a timestamp to the data.
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            json_data["label"] = label

            # Optionally, add the data to the CSV buffer and save periodically.
            data_processor.add_data(json_data)
            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()

            # Print the data for debugging.
            print("Broadcasting data:", json_data)
            
            # Broadcast the updated data to all connected clients.
            await websocket_manager.broadcast_message(json.dumps(json_data))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
