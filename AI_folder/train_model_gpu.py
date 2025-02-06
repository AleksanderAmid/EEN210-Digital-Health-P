'''# train_model.py
import pandas as pd
import tkinter as tk
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
import json  # Used for saving the training history

# Force TensorFlow to use the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ GPU is being used:", physical_devices[0])
    with tf.device('/GPU:0'):
        print("🚀 Running training on GPU.")
else:
    print("❌ No GPU detected, running on CPU.")

class ModelTrainer:
    def __init__(self, csv_path, window_size=20):
        self.csv_path = csv_path
        self.window_size = window_size
        self.sensor_cols = ['Acc_1', 'Acc_2', 'Acc_3',
                            'Gyr_1', 'Gyr_2', 'Gyr_3']
    
    def load_and_preprocess_data(self):
        df = pd.read_csv(self.csv_path)
        # Handle missing ActivityIDs and convert ActivityID to numeric if necessary.
        df = df.dropna(subset=['ActivityID'])
        df['ActivityID'] = pd.to_numeric(df['ActivityID'], errors='coerce')
        df = df.dropna(subset=['ActivityID'])
        
        scaler = StandardScaler()
        df[self.sensor_cols] = scaler.fit_transform(df[self.sensor_cols])
        return df
    

    def create_sequences(self, df):
        X, y = [], []
        data = df[self.sensor_cols].values
        ActivityIDs = df['ActivityID'].values
        for i in range(len(data) - self.window_size + 1):
            X.append(data[i:i+self.window_size])
            center_ActivityID = ActivityIDs[i + self.window_size // 2]
            y.append(center_ActivityID)
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
        # Determine number of classes based on maximum ActivityID value.
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
        model, history = trainer.train(epochs=40, batch_size=2048*4)
        print("Model training complete.")
        print("Training history saved to 'training_history.json'.")
    else:
        print("No file selected. Exiting.")
'''

# train_model_gpu.py
import json
import tkinter as tk
import tkinter.filedialog
import tensorflow as tf
import numpy as np
import os

# Force TensorFlow to use the GPU and allow memory growth.
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ GPU is being used:", physical_devices[0])
    with tf.device('/GPU:0'):
        print("🚀 Running training on GPU.")
else:
    print("❌ No GPU detected, running on CPU.")

class ModelTrainer:
    def __init__(self, csv_path, window_size=20):
        self.csv_path = csv_path
        self.window_size = window_size
        self.sensor_cols = ['Acc_1', 'Acc_2', 'Acc_3', 'Gyr_1', 'Gyr_2', 'Gyr_3']

    def create_dataset(self):
        """
        Create a tf.data.Dataset that streams the CSV file.
        Assumes the CSV has a header with columns: ActivityID, Acc_1, ..., Gyr_3.
        Each element is a tuple (features, label) where:
          - features: tensor of shape (6,)
          - label: scalar (ActivityID)
        """
        column_names = ['ActivityID'] + self.sensor_cols
        column_defaults = [0] + [0.0] * len(self.sensor_cols)

        dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=self.csv_path,
            batch_size=1,          # one row per element
            column_names=column_names,
            column_defaults=column_defaults,
            header=True,
            shuffle=False,
            num_epochs=1
        )
        # Convert each dictionary element into a tuple (features, label)
        # and explicitly set the feature shape to (6,)
        dataset = dataset.map(lambda x: (
            tf.ensure_shape(tf.stack([tf.squeeze(x[col]) for col in self.sensor_cols], axis=-1), (len(self.sensor_cols),)),
            tf.cast(tf.squeeze(x['ActivityID']), tf.int32)
        ))
        return dataset

    def create_sequence_dataset(self):
        """
        Converts the per-timestep dataset into sequences (windows) of length `window_size`.
        Each element is a tuple:
          - features: tensor of shape (window_size, 6)
          - label: the ActivityID from the center time step (scalar)
        """
        ds = self.create_dataset()
        # Create sliding windows over the dataset.
        ds = ds.window(self.window_size, shift=1, drop_remainder=True)
        
        # Helper function that takes two nested datasets (one for features, one for labels)
        # and batches them separately, then zips them together.
        def batch_window(features_window, labels_window):
            batched_features = features_window.batch(self.window_size)
            batched_labels = labels_window.batch(self.window_size)
            return tf.data.Dataset.zip((batched_features, batched_labels))
        
        # For each window (which is a tuple of nested datasets), apply our helper.
        ds = ds.flat_map(lambda features_window, labels_window: batch_window(features_window, labels_window))
        
        # Now, each element is a tuple:
        #   features: tensor of shape (window_size, 6) (but static shape might be unknown)
        #   labels: tensor of shape (window_size,)
        # Set their static shapes explicitly.
        ds = ds.map(lambda features, labels: (
            tf.ensure_shape(features, (self.window_size, len(self.sensor_cols))),
            tf.ensure_shape(labels, (self.window_size,))
        ))
        
        # Use the center label (index window_size // 2) as the label for the sequence.
        ds = ds.map(lambda features, labels: (features, labels[self.window_size // 2]))
        
        # Finally, ensure that the features have shape (window_size, 6) and the label is a scalar.
        ds = ds.map(lambda features, label: (
            tf.ensure_shape(features, (self.window_size, len(self.sensor_cols))),
            tf.ensure_shape(label, ())
        ))
        return ds



    def get_dataset_count(self, dataset):
        """
        Count the number of elements in a dataset.
        """
        count = 0
        for _ in dataset:
            count += 1
        return count

    def build_model(self, num_classes, normalization_layer):
        # Lower the learning rate and add gradient clipping.
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.window_size, len(self.sensor_cols))),
            normalization_layer,  # The normalization layer should ensure the inputs are scaled.
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def train(self, epochs=20, batch_size=32, steps_per_epoch=100, validation_steps=20):
        # Build training pipeline:
        train_ds = (
            self.create_sequence_dataset()
            .shuffle(buffer_size=10000)
            .batch(batch_size)
            .repeat()  # Repeat indefinitely for training.
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        # Build validation pipeline (separate, repeating pipeline):
        val_ds = (
            self.create_sequence_dataset()
            .shuffle(buffer_size=10000)
            .batch(batch_size)
            .repeat()  # Repeat indefinitely for validation.
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        # Determine the number of classes by iterating over a few elements.
        # Here, we simply take 1000 elements (without unbatching) because each element's label is already a scalar.
        max_label = 0
        for _, label in self.create_sequence_dataset().take(1000):
            max_label = max(max_label, int(label.numpy()))
        num_classes = max_label + 1
        print("Number of classes:", num_classes)

        # Adapt the normalization layer on a sample of individual feature vectors.
        # We extract the features, unbatch them (only the features part), and then stack them.
        norm_layer = tf.keras.layers.Normalization(axis=-1)
        sample_features = []
        # Create a dataset of only features.
        sample_ds = self.create_sequence_dataset().map(lambda f, l: f)
        # Unbatch the feature tensors: each sequence of shape (window_size, 6) becomes window_size individual rows of shape (6,).
        sample_ds = sample_ds.unbatch()
        for features in sample_ds.take(1000):
            sample_features.append(features)
        sample_features = tf.stack(sample_features)  # shape: (num_samples, 6)
        norm_layer.adapt(sample_features)
        print("Normalization layer adapted on sample data.")

        # Build and compile the model.
        model = self.build_model(num_classes, norm_layer)

        # Start training. Because both datasets repeat indefinitely, we need to specify steps.
        history = model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps
        )

        # Save the model and training history.
        model.save('fall_detection_model.h5')
        with open("training_history.json", "w") as f:
            import json
            json.dump(history.history, f)
        return model, history




# --- Run the Training Script via a File Selection Dialog ---
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = tk.filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if file_path:
        trainer = ModelTrainer(file_path, window_size=20)
        # Adjust epochs and batch size as desired.
        model, history = trainer.train(epochs=40, batch_size=32)
        print("Model training complete.")
        print("Training history saved to 'training_history.json'.")
    else:
        print("No file selected. Exiting.")



