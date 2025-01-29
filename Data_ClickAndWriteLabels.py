import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# -------- Nya import för labeling-funktionaliteten -------- #
import os
import tkinter as tk
from tkinter import simpledialog
from matplotlib.widgets import SpanSelector, Button
# ---------------------------------------------------------- #

# 1. Läs in data
file_path = r'data2\fall_zipperUp_data_20250129_110818.csv'
df = pd.read_csv(file_path)

ACC_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0
G_MS2 = 9.81
SAMPLE_RATE = 66.5

# Skapa en tidsaxel i sekunder
time = np.arange(len(df)) / SAMPLE_RATE

# 2. Konvertera accelerationsdata till m/s^2
acc_x_ms2 = (df['acceleration_x'] / ACC_LSB_PER_G) * G_MS2
acc_y_ms2 = (df['acceleration_y'] / ACC_LSB_PER_G) * G_MS2
acc_z_ms2 = (df['acceleration_z'] / ACC_LSB_PER_G) * G_MS2

# 3. Konvertera gyroskopdata till grader/s
gyro_x_dps = df['gyroscope_x'] / GYRO_LSB_PER_DPS
gyro_y_dps = df['gyroscope_y'] / GYRO_LSB_PER_DPS
gyro_z_dps = df['gyroscope_z'] / GYRO_LSB_PER_DPS

# 4. Applicera Savitzky-Golay-filter
window_length = 11
poly_order = 2

acc_x_filt = savgol_filter(acc_x_ms2, window_length, poly_order)
acc_y_filt = savgol_filter(acc_y_ms2, window_length, poly_order)
acc_z_filt = savgol_filter(acc_z_ms2, window_length, poly_order)

gyro_x_filt = savgol_filter(gyro_x_dps, window_length, poly_order)
gyro_y_filt = savgol_filter(gyro_y_dps, window_length, poly_order)
gyro_z_filt = savgol_filter(gyro_z_dps, window_length, poly_order)

# 5. Integrera gyroskopdata -> vinkel (grader)
angle_x = np.cumsum(gyro_x_filt) / SAMPLE_RATE
angle_y = np.cumsum(gyro_y_filt) / SAMPLE_RATE
angle_z = np.cumsum(gyro_z_filt) / SAMPLE_RATE

# 6. Dubbelintegrera accelerationsdata -> avstånd (meter)
vel_x = np.cumsum(acc_x_filt) / SAMPLE_RATE
vel_y = np.cumsum(acc_y_filt) / SAMPLE_RATE
vel_z = np.cumsum(acc_z_filt) / SAMPLE_RATE

dist_x = np.cumsum(vel_x) / SAMPLE_RATE
dist_y = np.cumsum(vel_y) / SAMPLE_RATE
dist_z = np.cumsum(vel_z) / SAMPLE_RATE

# ---------------------------- #
# Sektion för interaktiv labeling
# ---------------------------- #

fig = plt.figure(figsize=(10, 10))

# Skapa subplots
ax1 = fig.add_subplot(4, 1, 1)
ax2 = fig.add_subplot(4, 1, 2)
ax3 = fig.add_subplot(4, 1, 3)
ax4 = fig.add_subplot(4, 1, 4)

# 1) Filtered acceleration
ax1.plot(time, acc_x_filt, label='Acc X (filt)')
ax1.plot(time, acc_y_filt, label='Acc Y (filt)')
ax1.plot(time, acc_z_filt, label='Acc Z (filt)')
ax1.set_title('Filtrerad Accelerationsdata (m/s^2)')
ax1.set_xlabel('Tid (s)')
ax1.set_ylabel('Acceleration')
ax1.legend()
ax1.grid(True)

# 2) Filtered gyroscope
ax2.plot(time, gyro_x_filt, label='Gyro X (filt)')
ax2.plot(time, gyro_y_filt, label='Gyro Y (filt)')
ax2.plot(time, gyro_z_filt, label='Gyro Z (filt)')
ax2.set_title('Filtrerad Gyroskopdata (°/s)')
ax2.set_xlabel('Tid (s)')
ax2.set_ylabel('Vinkelhastighet')
ax2.legend()
ax2.grid(True)

# 3) Integrated gyro angles
ax3.plot(time, angle_x, label='Gyro X (deg)')
ax3.plot(time, angle_y, label='Gyro Y (deg)')
ax3.plot(time, angle_z, label='Gyro Z (deg)')
ax3.set_title('Integrerad Gyroskopdata (grader)')
ax3.set_xlabel('Tid (s)')
ax3.set_ylabel('Vinkel (°)')
ax3.legend()
ax3.grid(True)

# 4) Double-integrated acceleration (distance)
ax4.plot(time, dist_x, label='Dist X (m)')
ax4.plot(time, dist_y, label='Dist Y (m)')
ax4.plot(time, dist_z, label='Dist Z (m)')
ax4.set_title('Dubbelintegrerad Accelerationsdata (meter)')
ax4.set_xlabel('Tid (s)')
ax4.set_ylabel('Avstånd (m)')
ax4.legend()
ax4.grid(True)

plt.tight_layout()

# Variabler för att hantera selection
current_min = None
current_max = None

def onselect(xmin, xmax):
    """
    Callback som anropas när man drar med musen över en region i ax1.
    x-värdena xmin och xmax motsvarar sekunder i vår tidsaxel.
    """
    global df, current_min, current_max

    # För att hantera fall då man drar från höger till vänster
    if xmax < xmin:
        xmin, xmax = xmax, xmin

    # Konvertera tid -> index
    start_idx = int(np.searchsorted(time, xmin))
    end_idx = int(np.searchsorted(time, xmax))

    # Säkerställ att vi är inom DataFrame:ens gränser
    start_idx = max(0, start_idx)
    end_idx = min(len(df) - 1, end_idx)

    # Skapa ett litet dialogfönster för att fråga om etikett
    root = tk.Tk()
    root.withdraw()  # Dölj själva Tk-fönstret
    user_label = simpledialog.askstring("Label Data", 
                                        f"Du har markerat index {start_idx} till {end_idx}.\nAnge etikett:")
    root.destroy()

    # Uppdatera DataFrame om man inte avbröt dialogen
    if user_label is not None and user_label.strip() != "":
        df.loc[start_idx:end_idx, 'label'] = user_label
        print(f"Rader {start_idx} till {end_idx} är nu märkta med '{user_label}'.")

# Koppla SpanSelector till första subplot (acceleration)
span = SpanSelector(
    ax1, onselect, 'horizontal',
    useblit=True,  # Snabbare rendering
    props=dict(alpha=0.5, facecolor='red')  # Byt ut 'rectprops' mot 'props'
)

# Funktion för att spara DataFrame till CSV
def save_data(event):
    """
    Callback för att spara den uppdaterade DataFrame:en 
    till en ny CSV i mappen 'LabeledData'.
    """
    if not os.path.exists('LabeledData'):
        os.makedirs('LabeledData')

    out_file = os.path.join('LabeledData', os.path.basename(f"DONE_LABEL_{file_path}"))
    df.to_csv(out_file, index=False)
    print(f"Data sparad till: {out_file}")

# Skapa en "Spara"-knapp
ax_button = plt.axes([0.82, 0.01, 0.12, 0.05])
button = Button(ax_button, 'Spara')
button.on_clicked(save_data)

plt.show()
