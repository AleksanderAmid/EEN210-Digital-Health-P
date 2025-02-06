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
from tensorflow.keras.models import load_model as keras_load_model  # Import Keras load_model

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


data_processor = DataProcessor()


def load_model():
    """
    Load the Keras model from the .h5 file.
    """
    model = keras_load_model("fall_detection_model.h5")
    return model


def predict_label(model=None, data=None):
    """
    Process the input data and predict the label using the loaded model.
    This example assumes that:
      - The model expects a 2D input (batch_size x features).
      - The model outputs a single probability (for binary classification).
    Adjust the preprocessing and threshold as needed.
    """
    if model is not None and data is not None:
        # Convert list of raw data into a NumPy array and reshape it for prediction
        data_array = np.array(data).reshape(1, -1)
        prediction = model.predict(data_array)
        # For a binary classification model returning a probability,
        # we can use 0.5 as the threshold.
        label = int(prediction[0][0] > 0.5)
        return label
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
            print("Received data from websocket")

            # Parse JSON data and prepare it for processing
            json_data = json.loads(data)
            # Extract raw values (ensure they are in the expected order)
            raw_data = list(json_data.values())

            # Add a timestamp to the received data
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            data_processor.add_data(json_data)
            # Save to CSV every 100 samples (adjust if needed)
            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()

            # Use the model to predict the label for the incoming data
            label = predict_label(model, raw_data)
            json_data["label"] = label

            # Log the data with prediction to the terminal
            print(json_data)

            # Broadcast the data with prediction to all connected clients
            await websocket_manager.broadcast_message(json.dumps(json_data))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
