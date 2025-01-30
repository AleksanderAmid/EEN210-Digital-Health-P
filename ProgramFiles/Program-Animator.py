import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def load_images():
    """Ladda alla bilder i en ordbok."""
    images = {
        "walking": [
            cv2.imread("Images/Walking 1.png"), 
            cv2.imread("Images/Walking 2.png")
        ],
        "jumping": [
            cv2.imread("Images/Jump 1.png"), 
            cv2.imread("Images/Jump 2.png")
        ],
        "standing": cv2.imread("Images/Standing.png")
    }
    return images

def animate_movement(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["label"])

    images = load_images()
    for key, val in images.items():
        if isinstance(val, list):
            images[key] = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in val]
        else:
            images[key] = cv2.cvtColor(val, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    plt.tight_layout(pad=3.0)

    x = np.arange(len(df))
    ax1.plot(x, df['acceleration_x'], label='Acc X')
    ax1.plot(x, df['acceleration_y'], label='Acc Y')
    ax1.plot(x, df['acceleration_z'], label='Acc Z')
    ax1.set_title('Accelerationsdata')
    ax1.set_xlabel('Provdatalängd (index)')
    ax1.set_ylabel('Acceleration')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x, df['gyroscope_x'], label='Gyro X')
    ax2.plot(x, df['gyroscope_y'], label='Gyro Y')
    ax2.plot(x, df['gyroscope_z'], label='Gyro Z')
    ax2.set_title('Gyroskopdata')
    ax2.set_xlabel('Provdatalängd (index)')
    ax2.set_ylabel('Vinkelhastighet')
    ax2.legend()
    ax2.grid(True)

    vertical_line_acc = ax1.axvline(x=0, color='r', linestyle='--')
    vertical_line_gyro = ax2.axvline(x=0, color='r', linestyle='--')
    max_acc_x = df['acceleration_x'].max()
    dot_label = ax1.text(
        0, max_acc_x, '', 
        fontsize=12, color='red', weight='bold', 
        verticalalignment='bottom'
    )

    ax3.set_title('Rörelseanimation')
    img_plot = ax3.imshow(images["standing"])
    ax3.axis('off')

    frame_idx = 0
    for i in range(0, len(df), 10):
        label = int(df.iloc[i]["label"])
        if label == 1:
            img = images["walking"][frame_idx % 2]
        elif label == 4:
            img = images["jumping"][frame_idx % 2]
        else:
            img = images["standing"]
        img_plot.set_data(img)
        vertical_line_acc.set_xdata([i])
        vertical_line_gyro.set_xdata([i])
        dot_label.set_x(i)
        dot_label.set_text(f"Label: {label}")
        plt.pause(0.01)
        frame_idx += 1

    plt.show()

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    csv_path = askopenfilename(
        initialdir="C:/Users/sshan/Desktop/EEN210-Digital-Health-P/LabeledData",
        filetypes=[("CSV files", "*.csv")],
        title="Välj CSV-fil"
    )
    animate_movement(csv_path)
