import os
import sys
import shutil
import tkinter as tk
import tkinter.filedialog as fd
from tkinter import messagebox, simpledialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button

ACC_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0
G_MS2 = 9.81
SAMPLE_RATE = 66.5

def process_file(file_path: str):
    df = pd.read_csv(file_path)
    time = np.arange(len(df)) / SAMPLE_RATE

    # Omvandla accelerationsdata till m/s^2 (ej filtrerade)
    acc_x = (df['acceleration_x'] / ACC_LSB_PER_G) * G_MS2
    acc_y = (df['acceleration_y'] / ACC_LSB_PER_G) * G_MS2
    acc_z = (df['acceleration_z'] / ACC_LSB_PER_G) * G_MS2

    # Omvandla gyroskopdata till grader per sekund (ej filtrerade)
    gyro_x = df['gyroscope_x'] / GYRO_LSB_PER_DPS
    gyro_y = df['gyroscope_y'] / GYRO_LSB_PER_DPS
    gyro_z = df['gyroscope_z'] / GYRO_LSB_PER_DPS

    # Skapa figur och axlar för plotting
    fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title(file_path)

    # Plotta accelerationsdata
    ax_acc.plot(time, acc_x, label='Acc X')
    ax_acc.plot(time, acc_y, label='Acc Y')
    ax_acc.plot(time, acc_z, label='Acc Z')
    ax_acc.set_title('Accelerationsdata (m/s^2)')
    ax_acc.set_xlabel('Tid (s)')
    ax_acc.set_ylabel('Acceleration')
    ax_acc.legend()
    ax_acc.grid(True)

    # Plotta gyroskopdata
    ax_gyro.plot(time, gyro_x, label='Gyro X')
    ax_gyro.plot(time, gyro_y, label='Gyro Y')
    ax_gyro.plot(time, gyro_z, label='Gyro Z')
    ax_gyro.set_title('Gyroskopdata (°/s)')
    ax_gyro.set_xlabel('Tid (s)')
    ax_gyro.set_ylabel('Vinkelhastighet')
    ax_gyro.legend()
    ax_gyro.grid(True)

    plt.tight_layout()

    # Funktion för att hantera markering och etikettering
    def make_onselect(plot_name):
        def _onselect(xmin, xmax):
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            start_idx = int(np.searchsorted(time, xmin))
            end_idx   = int(np.searchsorted(time, xmax))
            start_idx = max(0, start_idx)
            end_idx   = min(len(df) - 1, end_idx)
            root = tk.Tk()
            root.withdraw()
            user_label = simpledialog.askstring(
                f"Label Data: {plot_name}",
                f"Markerat index {start_idx} till {end_idx}.\nAnge etikett:"
            )
            root.destroy()
            if user_label and user_label.strip():
                df.loc[start_idx:end_idx, 'label'] = user_label
                print(f"[{plot_name}] index {start_idx}..{end_idx} => '{user_label}'")
        return _onselect

    # Skapa span selectors för att välja intervall i de två diagrammen
    span_acc = SpanSelector(
        ax_acc, make_onselect("accelerometer"), 'horizontal',
        useblit=True, props=dict(alpha=0.5, facecolor='red')
    )
    span_gyro = SpanSelector(
        ax_gyro, make_onselect("gyroscope"), 'horizontal',
        useblit=True, props=dict(alpha=0.5, facecolor='green')
    )

    # Funktion för att spara data och flytta originalfilen
    def save_data(event):
        if not os.path.exists('LabeledData'):
            os.makedirs('LabeledData')
        out_file = os.path.join('LabeledData', f"DONE_LABEL_{os.path.basename(file_path)}")
        df.to_csv(out_file, index=False)
        print(f"Data sparad till: {out_file}")

        # Flytta originalfilen till mappen "Papperskorg"
        trash_dir = 'Papperskorg'
        if not os.path.exists(trash_dir):
            os.makedirs(trash_dir)
        target_path = os.path.join(trash_dir, os.path.basename(file_path))
        shutil.move(file_path, target_path)
        print(f"Originalfilen flyttad till: {target_path}")

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Sparad Data", "Success Saved, Good job my friend")
        root.destroy()
        plt.close(fig)

    # Lägg till en knapp i figuren för att spara data
    ax_button = plt.axes([0.82, 0.01, 0.12, 0.05])
    button = Button(ax_button, 'Spara')
    button.on_clicked(save_data)
    plt.show()

def main():
    root = tk.Tk()
    root.withdraw()
    choice = messagebox.askquestion("Välj källa", "Ja=Enskild fil, Nej=Mapp med filer")
    if choice == 'yes':
        path = fd.askopenfilename(
            title="Välj EN CSV-fil",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialdir=os.getcwd()
        )
        if path and os.path.isfile(path):
            process_file(path)
        else:
            messagebox.showwarning("Varning", "Ingen giltig fil vald.")
    else:
        folder_path = fd.askdirectory(title="Välj en mapp med CSV-filer", initialdir=os.getcwd())
        if folder_path and os.path.isdir(folder_path):
            all_files = [os.path.join(folder_path, f)
                         for f in os.listdir(folder_path)
                         if f.lower().endswith(".csv")]
            all_files.sort()
            if not all_files:
                messagebox.showwarning("Inga CSV-filer", "Ingen CSV-fil hittades i mappen.")
            else:
                for csv_file in all_files:
                    process_file(csv_file)
        else:
            messagebox.showwarning("Avbrutet", "Ingen giltig mapp vald.")
    sys.exit(0)

if __name__ == "__main__":
    main()
