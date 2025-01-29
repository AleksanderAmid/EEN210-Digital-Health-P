import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# -------- Nya import för labeling-funktionaliteten -------- #
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
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

# ---------------------------- #
# Sektion för interaktiv labeling
# ---------------------------- #

# Skapa en figur med 2 underliggande plot-fält (subplots)
fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, figsize=(10, 8))

# 1) Filtrerad accelerationsdata
ax_acc.plot(time, acc_x_filt, label='Acc X (filt)')
ax_acc.plot(time, acc_y_filt, label='Acc Y (filt)')
ax_acc.plot(time, acc_z_filt, label='Acc Z (filt)')
ax_acc.set_title('Filtrerad Accelerationsdata (m/s^2)')
ax_acc.set_xlabel('Tid (s)')
ax_acc.set_ylabel('Acceleration')
ax_acc.legend()
ax_acc.grid(True)

# 2) Filtrerad gyroskopdata
ax_gyro.plot(time, gyro_x_filt, label='Gyro X (filt)')
ax_gyro.plot(time, gyro_y_filt, label='Gyro Y (filt)')
ax_gyro.plot(time, gyro_z_filt, label='Gyro Z (filt)')
ax_gyro.set_title('Filtrerad Gyroskopdata (°/s)')
ax_gyro.set_xlabel('Tid (s)')
ax_gyro.set_ylabel('Vinkelhastighet')
ax_gyro.legend()
ax_gyro.grid(True)

plt.tight_layout()


def make_onselect(plot_name):
    """
    Returnerar en funktion onselect(xmin, xmax) som kan kopplas till SpanSelector.
    plot_name är en sträng ('accelerometer' eller 'gyroscope') för utskrift/logg.
    """
    def _onselect(xmin, xmax):
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        
        # Tid -> index
        start_idx = int(np.searchsorted(time, xmin))
        end_idx = int(np.searchsorted(time, xmax))
        
        # Begränsa inom DataFrame
        start_idx = max(0, start_idx)
        end_idx = min(len(df) - 1, end_idx)
        
        # Dialogruta för etikett
        root = tk.Tk()
        root.withdraw()
        user_label = simpledialog.askstring(
            f"Label Data: {plot_name}",
            f"Markerat index {start_idx} till {end_idx} på {plot_name}.\nAnge etikett:"
        )
        root.destroy()
        
        # Uppdatera DataFrame
        if user_label is not None and user_label.strip() != "":
            df.loc[start_idx:end_idx, 'label'] = user_label
            print(f"[{plot_name}] Rader {start_idx} till {end_idx} är nu märkta med '{user_label}'.")
    return _onselect

# Koppla en SpanSelector till accelerationsgrafen
span_acc = SpanSelector(
    ax_acc,
    make_onselect("accelerometer"),
    'horizontal',
    useblit=True,
    props=dict(alpha=0.5, facecolor='red')
)

# Koppla en SpanSelector till gyrografen
span_gyro = SpanSelector(
    ax_gyro,
    make_onselect("gyroscope"),
    'horizontal',
    useblit=True,
    props=dict(alpha=0.5, facecolor='green')
)

def save_data(event):
    """
    Callback för att spara den uppdaterade DataFrame:en 
    till en ny CSV i mappen 'LabeledData'.
    """
    if not os.path.exists('LabeledData'):
        os.makedirs('LabeledData')
    
    out_file = os.path.join('LabeledData', f"DONE_LABEL_{os.path.basename(file_path)}")
    df.to_csv(out_file, index=False)
    print(f"Data sparad till: {out_file}")

    # Visa en "lyckat-sparande"-ruta
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Sparad Data", "Success Saved, Good job my friend")
    root.destroy()

# Skapa en "Spara"-knapp i marginalen av figuren
ax_button = plt.axes([0.82, 0.01, 0.12, 0.05])
button = Button(ax_button, 'Spara')
button.on_clicked(save_data)

plt.show()


