import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Initialize Tkinter and hide the root window
Tk().withdraw()

# Open file dialog and ask for the CSV file
file_path = askopenfilename(filetypes=[("CSV files", "*.csv")])

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