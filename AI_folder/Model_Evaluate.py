# evaluate_model.py
import matplotlib.pyplot as plt
from ModelTrainer import ModelTrainer
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Initialize trainer with the same CSV file.
trainer = ModelTrainer(csv_path=r'AI_folder\Merged_Data_15-46-39.csv')
model, history = trainer.train(epochs=20, batch_size=32)

# Load model if needed:
model = load_model('fall_detection_model.h5')

# Example: Plotting training/validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Example: Confusion Matrix
# You would need to generate predictions on X_test and compare with true labels.
# Suppose we have y_true and y_pred from our test set:
# y_pred = model.predict(X_test)
# y_pred_labels = np.argmax(y_pred, axis=1)
# y_true_labels = np.argmax(y_test, axis=1)
# cm = confusion_matrix(y_true_labels, y_pred_labels)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
