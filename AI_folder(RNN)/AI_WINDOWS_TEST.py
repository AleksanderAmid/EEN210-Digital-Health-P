import pandas as pd
import tkinter as tk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import json
import os

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
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
        
        # Spara (om du vill) modellen och historiken
        model.save('fall_detection_model.h5')
        with open("training_history.json", "w") as f:
            json.dump(history.history, f)
        
        return model, history

def compare_window_sizes(csv_path, window_sizes=[10,20,30,40,50], epochs=5, batch_size=32):
    """
    Tränar modellen med ett antal olika window_sizes och samma (låg) antal epoker
    för att snabbt se vilket fönster som verkar fungera bäst. Returnerar en
    dictionary med (window_size -> history).
    """
    results = {}
    
    for w in window_sizes:
        print(f"\nTränar med window_size = {w}")
        trainer = ModelTrainer(csv_path, window_size=w)
        _, history = trainer.train(epochs=epochs, batch_size=batch_size)
        # Spara historia i en dictionary
        results[w] = history.history  # => t.ex. {'loss': [...], 'accuracy': [...], 'val_loss': [...], 'val_accuracy': [...]}
        
        # Om du vill behålla modellerna (tillfälligt) för varje window_size kan du spara dem 
        # under olika filnamn. Ex.:
        # model.save(f"model_w{w}.h5")

    return results

def plot_comparison(results):
    """
    Tar in en dictionary {window_size: history} och plottar val_accuracy för att
    jämföra hur olika window_sizes presterar över 5 epoch.
    """
    plt.figure(figsize=(10, 5))
    
    for w, hist in results.items():
        val_acc = hist['val_accuracy']
        plt.plot(range(1, len(val_acc)+1), val_acc, label=f'window={w}')
    
    plt.title("Jämförelse av window_sizes - val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Valideringsnoggrannhet")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # En enkel main som:
    # 1) ber dig välja en CSV-fil
    # 2) kör compare_window_sizes
    # 3) plottar resultat
    
    import tkinter.filedialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = tkinter.filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if file_path:
        # Bestäm vilka window sizes du vill testa och hur många epoch du vill köra (ex. 5).
        sizes_to_try = [50, 60, 70, 80, 90, 100]
        results_dict = compare_window_sizes(file_path, window_sizes=sizes_to_try, epochs=10, batch_size=1024)
        plot_comparison(results_dict)
    else:
        print("No file selected. Exiting.")
