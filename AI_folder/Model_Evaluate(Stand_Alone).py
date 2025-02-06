import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def load_and_preprocess_data(csv_path, window_size=20):
    """
    Laddar in CSV-filen, hanterar saknade etiketter, konverterar label till numerisk
    och standardiserar sensor-kolumner.
    """
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.dropna(subset=['label'])
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    
    sensor_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                   'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    return df, sensor_cols

def create_sequences(df, sensor_cols, window_size=20):
    """
    Skapar sekvenser (X) och etiketter (y) med ett glidande fönster.
    Vid varje steg används mittenvärdet i fönstret som etikett.
    """
    X, y = [], []
    data = df[sensor_cols].values
    labels = df['label'].values
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i+window_size])
        center_label = labels[i + window_size // 2]
        y.append(center_label)
    X = np.array(X)
    y = np.array(y)
    return X, y

def plot_training_history(history):
    """
    Plottar träningskurvor för loss och accuracy över epoch.
    """
    epochs = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    # Plot för förlust (loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label='Träningsförlust')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], label='Valideringsförlust')
    plt.title('Epoch-kurva: Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Förlust')
    plt.legend()
    
    # Plot för noggrannhet (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], label='Träningsnoggrannhet')
    if 'val_accuracy' in history:
        plt.plot(epochs, history['val_accuracy'], label='Valideringsnoggrannhet')
    plt.title('Epoch-kurva: Noggrannhet')
    plt.xlabel('Epoch')
    plt.ylabel('Noggrannhet')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """
    Plottar en konfusionsmatris med hjälp av seaborn.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Förutsagda värden')
    plt.ylabel('Sanna värden')
    plt.title('Konfusionsmatris')
    plt.show()

def check_overfitting(history):
    """
    Enkel heuristik för att kolla överanpassning genom att jämföra slutlig tränings- 
    och valideringsnoggrannhet.
    """
    if 'accuracy' in history and 'val_accuracy' in history:
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        print(f"Slutlig träningsnoggrannhet: {final_train_acc:.4f}")
        print(f"Slutlig valideringsnoggrannhet: {final_val_acc:.4f}")
        if final_train_acc - final_val_acc > 0.1:
            print("Det verkar finnas överanpassning.")
        else:
            print("Ingen tydlig överanpassning upptäckt.")
    else:
        print("Ingen tillräcklig information om noggrannhet för att bedöma överanpassning.")

def main():
    # Initiera tkinter och dölja huvudfönstret
    root = tk.Tk()
    root.withdraw()
    
    # Välj träningsdata CSV-fil
    csv_path = filedialog.askopenfilename(
        title="Välj träningsdata CSV-fil",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not csv_path:
        print("Ingen CSV-fil vald. Avslutar.")
        return

    # Välj modellfil (.h5)
    model_path = filedialog.askopenfilename(
        title="Välj modellfil (.h5)",
        filetypes=[("H5 Files", "*.h5"), ("All Files", "*.*")]
    )
    if not model_path:
        print("Ingen modellfil vald. Avslutar.")
        return
    
    # Välj träningshistorik JSON-fil
    history_path = filedialog.askopenfilename(
        title="Välj träningshistorik JSON-fil",
        filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
    )
    if not history_path:
        print("Ingen träningshistorik vald. Avslutar.")
        return

    # Ladda träningshistoriken
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Ladda och förbehandla data
    window_size = 20  # Kan ändras vid behov
    df, sensor_cols = load_and_preprocess_data(csv_path, window_size)
    X, y = create_sequences(df, sensor_cols, window_size)
    
    # Bestäm antal klasser (antaget att etiketterna är 0-indexerade)
    num_classes = int(np.max(y)) + 1
    # Omvandla y till one-hot encoding
    y_cat = to_categorical(y, num_classes=num_classes)
    
    # Dela upp i tränings- och testdata (80% träning, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    
    # Ladda den sparade modellen
    model = load_model(model_path)
    
    # Utvärdera modellen på testdata
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Testförlust: {loss:.4f}")
    print(f"Testnoggrannhet: {accuracy:.4f}")
    
    # Generera prediktioner på testdata
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Utskrift av klassificeringsrapport (precision, recall, f1-score, support)
    report = classification_report(y_true, y_pred)
    print("Klassificeringsrapport:")
    print(report)
    
    # Skapa och visa konfusionsmatris
    cm = confusion_matrix(y_true, y_pred)
    class_names = [str(i) for i in range(num_classes)]
    print("Konfusionsmatris:")
    print(cm)
    plot_confusion_matrix(cm, class_names)
    
    # Plot epoch-träningskurvor från träningshistoriken
    plot_training_history(history)
    
    # Enkel överanpassningsbedömning
    check_overfitting(history)

if __name__ == "__main__":
    main()
