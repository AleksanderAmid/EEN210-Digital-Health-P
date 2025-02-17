import os
import json
from datetime import datetime

import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
import numpy as np
from joblib import load

# Import the necessary functions from your model code.
# Ensure that 'model_code.py' is in the same directory or in your PYTHONPATH.
from RF_MODEL import extract_features

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


#############################
# Data Logging and Buffering
#############################
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


#############################
# Model Loading
#############################
def load_model():
    model_path = "./fall_detection_model.joblib"  # Update if necessary
    model = load(model_path)
    print(f"Model loaded from {model_path}")
    return model

model = load_model()

# Global buffer to accumulate live sensor data for a sliding window.
PREDICTION_WINDOW_SIZE = 135  # Same window size used during training
prediction_buffer = []


#############################
# Live Prediction Function
#############################
def predict_label_live(model):
    """
    If enough sensor data has been accumulated, extract features
    from the most recent window and use the model to predict the label.
    """
    if len(prediction_buffer) >= PREDICTION_WINDOW_SIZE:
        # Use the most recent window of data
        window_data = pd.DataFrame(prediction_buffer[-PREDICTION_WINDOW_SIZE:])
        features = extract_features(window_data)
        # Convert features to the format expected by the model (2D array)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        label = model.predict(feature_vector)[0]
        return label
    return None


#############################
# WebSocket Communication
#############################
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


#############################
# API Endpoints
#############################
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
            # Add a timestamp for logging purposes
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Log the data to CSV periodically
            data_processor.add_data(json_data)
            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()
            
            # Append the new reading to the prediction buffer
            prediction_buffer.append(json_data)
            
            # Make a prediction if a full window is available
            label = predict_label_live(model)
            if label is None:
                json_data["label"] = "insufficient data"
            else:
                json_data["label"] = label

            print(json_data)
            await websocket_manager.broadcast_message(json.dumps(json_data))
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
