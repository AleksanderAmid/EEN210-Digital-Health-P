import pandas as pd
import numpy as np
import os
import shutil
import tkinter as tk
import sys
from tkinter import messagebox
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.widgets import Button

real_file = r'data2\still_data_20250129_102942.csv'
df_original = pd.read_csv(real_file)

def fix_offset_in_memory(df):
    df_copy = df.copy()
    cols = ['acceleration_x','acceleration_y','acceleration_z',
            'gyroscope_x','gyroscope_y','gyroscope_z']
    offsets = df_copy[cols].head(10).mean()
    df_copy[cols] = df_copy[cols] - offsets
    return df_copy

offset_df = fix_offset_in_memory(df_original)

ACC_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0
G_MS2 = 9.81
SAMPLE_RATE = 66.5
time = np.arange(len(offset_df)) / SAMPLE_RATE

acc_x_ms2 = (offset_df['acceleration_x'] / ACC_LSB_PER_G) * G_MS2
acc_y_ms2 = (offset_df['acceleration_y'] / ACC_LSB_PER_G) * G_MS2
acc_z_ms2 = (offset_df['acceleration_z'] / ACC_LSB_PER_G) * G_MS2

gyro_x_dps = offset_df['gyroscope_x'] / GYRO_LSB_PER_DPS
gyro_y_dps = offset_df['gyroscope_y'] / GYRO_LSB_PER_DPS
gyro_z_dps = offset_df['gyroscope_z'] / GYRO_LSB_PER_DPS

window_length = 11
poly_order = 2

acc_x_filt = savgol_filter(acc_x_ms2, window_length, poly_order)
acc_y_filt = savgol_filter(acc_y_ms2, window_length, poly_order)
acc_z_filt = savgol_filter(acc_z_ms2, window_length, poly_order)

gyro_x_filt = savgol_filter(gyro_x_dps, window_length, poly_order)
gyro_y_filt = savgol_filter(gyro_y_dps, window_length, poly_order)
gyro_z_filt = savgol_filter(gyro_z_dps, window_length, poly_order)

fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, figsize=(10, 8))

ax_acc.plot(time, acc_x_filt, label='Acc X (filt)')
ax_acc.plot(time, acc_y_filt, label='Acc Y (filt)')
ax_acc.plot(time, acc_z_filt, label='Acc Z (filt)')
ax_acc.set_title('Filtrerad Accelerationsdata (m/s^2) [Offset-korr i minnet]')
ax_acc.set_xlabel('Tid (s)')
ax_acc.set_ylabel('m/s^2')
ax_acc.legend()
ax_acc.grid(True)

ax_gyro.plot(time, gyro_x_filt, label='Gyro X (filt)')
ax_gyro.plot(time, gyro_y_filt, label='Gyro Y (filt)')
ax_gyro.plot(time, gyro_z_filt, label='Gyro Z (filt)')
ax_gyro.set_title('Filtrerad Gyroskopdata (°/s) [Offset-korr i minnet]')
ax_gyro.set_xlabel('Tid (s)')
ax_gyro.set_ylabel('°/s')
ax_gyro.legend()
ax_gyro.grid(True)

plt.tight_layout()

def move_offset_file(event):
    target_dir = 'readyToLabel2'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filename = os.path.basename(real_file)
    offset_filename = f"offset_fixed_samefile_{filename}"
    offset_path = os.path.join(target_dir, offset_filename)
    offset_df.to_csv(offset_path, index=False)

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Offset-fil skapad och flyttad",
                        f"Offset-fil:\n{offset_filename}\nhar flyttats till '{target_dir}'.")
    root.destroy()
    sys.exit(0)  # Avsluta programmet

def move_original_file(event):
    trash_dir = 'papperskorg'
    if not os.path.exists(trash_dir):
        os.makedirs(trash_dir)
    target_path = os.path.join(trash_dir, os.path.basename(real_file))
    shutil.move(real_file, target_path)

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Originalfil flyttad",
                        f"Originalfilen:\n{os.path.basename(real_file)}\n"
                        f"har flyttats till '{trash_dir}'.")
    root.destroy()
    sys.exit(0)  # Avsluta programmet

ax_button_offset = plt.axes([0.68, 0.01, 0.3, 0.05])
button_offset = Button(ax_button_offset, 'Ready to Label')
button_offset.on_clicked(move_offset_file)

ax_button_trash = plt.axes([0.35, 0.01, 0.3, 0.05])
button_trash = Button(ax_button_trash, 'Flytta till Papperskorg')
button_trash.on_clicked(move_original_file)

plt.show()
