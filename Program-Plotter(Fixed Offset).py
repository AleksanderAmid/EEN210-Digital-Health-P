import pandas as pd
import matplotlib.pyplot as plt
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def fix_offset(real_file):
    real_data = pd.read_csv(real_file)
    cols = ['acceleration_x','acceleration_y','acceleration_z',
            'gyroscope_x','gyroscope_y','gyroscope_z']
    offsets = real_data[cols].head(10).mean()
    real_data[cols] = real_data[cols] - offsets
    directory = os.path.dirname(real_file)
    filename = os.path.basename(real_file)
    new_filename = os.path.join(directory, f"offset_fixed_{filename}")
    real_data.to_csv(new_filename, index=False)
    return new_filename

# Skapa en Tkinter-root och göm den
root = Tk()
root.withdraw()

# Öppna filutforskaren och låt användaren välja en fil
file_path = askopenfilename(title="Välj en CSV-fil", filetypes=[("CSV files", "*.csv")])

# Kontrollera om en fil valdes
if file_path:
    # Fix offset and get new file path
    file_path = fix_offset(file_path)

    # Läs in data med pandas
    df = pd.read_csv(file_path)

    # Skapa en x-axel som bara är indexet (0 till antal rader - 1)
    x = range(len(df))

    # Skapa figur och två subplots
    plt.figure(figsize=(10, 6))

    # Subplot 1: Accelerationsdata
    plt.subplot(2, 1, 1)
    plt.plot(x, df['acceleration_x'], label='Acc X')
    plt.plot(x, df['acceleration_y'], label='Acc Y')
    plt.plot(x, df['acceleration_z'], label='Acc Z')
    plt.title('Accelerationsdata')
    plt.xlabel('Provdatalängd (index)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Gyroskopdata
    plt.subplot(2, 1, 2)
    plt.plot(x, df['gyroscope_x'], label='Gyro X')
    plt.plot(x, df['gyroscope_y'], label='Gyro Y')
    plt.plot(x, df['gyroscope_z'], label='Gyro Z')
    plt.title('Gyroskopdata')
    plt.xlabel('Provdatalängd (index)')
    plt.ylabel('Vinkelhastighet')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("Ingen fil valdes.")
