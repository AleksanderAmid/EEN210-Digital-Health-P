# evaluate_h5.py

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model

# Import the ModelTrainer class from your training module.
from ModelTrainer import ModelTrainer


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        # Compute the row sums and avoid division by zero.
        row_sum = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm.astype('float'), row_sum, where=(row_sum != 0))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2. if cm.max() != 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = format(cm[i, j], '.2f') if normalize else format(cm[i, j], 'd')
        plt.text(j, i, value,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_training_curves(model):
    """
    Plots the training and validation accuracy and loss curves if available.
    Note: The training history is not saved automatically with model.save().
    """
    # Check if the model has a history attribute.
    if hasattr(model, 'history') and hasattr(model.history, 'history'):
        hist = model.history.history

        # Check for an accuracy key.
        if 'accuracy' in hist:
            acc_key = 'accuracy'
            val_acc_key = 'val_accuracy'
        elif 'acc' in hist:
            acc_key = 'acc'
            val_acc_key = 'val_acc'
        else:
            print("No training accuracy available in the loaded model history.")
            return

        epochs = range(1, len(hist[acc_key]) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, hist[acc_key], marker='o', label='Train Accuracy')
        plt.plot(epochs, hist[val_acc_key], marker='o', label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, hist['loss'], marker='o', label='Train Loss')
        plt.plot(epochs, hist['val_loss'], marker='o', label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    else:
        print("No training history available in the loaded model.")


def main():
    # Initialize Tkinter and hide the root window.
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select the trained model (.h5 file).
    model_path = filedialog.askopenfilename(
        title="Select the trained model (.h5 file)",
        filetypes=[("H5 files", "*.h5")]
    )
    if not model_path:
        print("No model file selected. Exiting.")
        return

    # Load the trained model.
    model = load_model(model_path)
    print(f"Loaded model from: {model_path}")

    # Ask the user to select the CSV file containing test sensor data.
    test_csv = filedialog.askopenfilename(
        title="Select the CSV test file",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not test_csv:
        print("No test CSV file selected. Exiting.")
        return

    # Create an instance of ModelTrainer to preprocess the test data.
    trainer = ModelTrainer(csv_path=test_csv, window_size=20)
    df_test = trainer.load_and_preprocess_data()
    X_test, y_true = trainer.create_sequences(df_test)

    # Run predictions on the test sequences.
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    # Compute unique classes from the test labels.
    unique_classes = sorted(np.unique(y_true))
    
    # Compute the confusion matrix.
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    # Plot the normalized confusion matrix.
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm, classes=[str(c) for c in unique_classes], normalize=True, title="Normalized Confusion Matrix")
    plt.show()

    # Additional evaluation metrics.
    accuracy = accuracy_score(y_true, y_pred)
    print("Overall Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=unique_classes, target_names=[str(c) for c in unique_classes]))

    # Plot training curves (accuracy and loss vs. epoch) if available.
    plot_training_curves(model)


if __name__ == '__main__':
    main()
