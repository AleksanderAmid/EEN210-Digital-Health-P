import os
import sys
import shutil
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import messagebox
from scipy.signal import savgol_filter
from matplotlib.widgets import Button

# Parametrar för filtret och konstanter
ACC_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0
G_MS2 = 9.81
SAMPLE_RATE = 66.5
WINDOW_LENGTH = 11
POLY_ORDER = 2

def fix_offset_in_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gör offset-korrigering i minnet (tar medelvärde av de första 10 raderna).
    Skapar ingen fysisk fil.
    """
    df_copy = df.copy()
    cols = ['acceleration_x','acceleration_y','acceleration_z',
            'gyroscope_x','gyroscope_y','gyroscope_z']
    offsets = df_copy[cols].head(10).mean()
    df_copy[cols] = df_copy[cols] - offsets
    return df_copy

def process_file(file_path: str):
    """
    Läser in en fil, gör offset-korrigering i minnet,
    filtrerar, plottar och skapar två knappar:
      - "Ready to Label": Skapar offset-fil och flyttar till 'readyToLabel2'
      - "Flytta till Papperskorg": Flyttar originalfilen till 'papperskorg'
    När en knapp klickas stängs plot-fönstret och funktionen returnerar.
    """
    print(f"\n--- Bearbetar fil: {file_path} ---")

    # 1) Läs in data
    df_original = pd.read_csv(file_path)

    # 2) Offset i minnet
    offset_df = fix_offset_in_memory(df_original)

    # 3) Skapa tidsaxel
    time = np.arange(len(offset_df)) / SAMPLE_RATE

    # 4) Extrahera och filtrera accelerationsdata
    acc_x_ms2 = (offset_df['acceleration_x'] / ACC_LSB_PER_G) * G_MS2
    acc_y_ms2 = (offset_df['acceleration_y'] / ACC_LSB_PER_G) * G_MS2
    acc_z_ms2 = (offset_df['acceleration_z'] / ACC_LSB_PER_G) * G_MS2

    acc_x_filt = savgol_filter(acc_x_ms2, WINDOW_LENGTH, POLY_ORDER)
    acc_y_filt = savgol_filter(acc_y_ms2, WINDOW_LENGTH, POLY_ORDER)
    acc_z_filt = savgol_filter(acc_z_ms2, WINDOW_LENGTH, POLY_ORDER)

    # 5) Extrahera och filtrera gyroskopdata
    gyro_x_dps = offset_df['gyroscope_x'] / GYRO_LSB_PER_DPS
    gyro_y_dps = offset_df['gyroscope_y'] / GYRO_LSB_PER_DPS
    gyro_z_dps = offset_df['gyroscope_z'] / GYRO_LSB_PER_DPS

    gyro_x_filt = savgol_filter(gyro_x_dps, WINDOW_LENGTH, POLY_ORDER)
    gyro_y_filt = savgol_filter(gyro_y_dps, WINDOW_LENGTH, POLY_ORDER)
    gyro_z_filt = savgol_filter(gyro_z_dps, WINDOW_LENGTH, POLY_ORDER)

    # 6) Plotta
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

    # Inre funktioner för knapp-klick
    def on_ready_to_label(event):
        """
        Skapar en offset-fil av offset_df, flyttar den till 'readyToLabel2'.
        Stänger plott-fönstret och återgår till skriptet.
        """
        target_dir = 'readyToLabel2'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        filename = os.path.basename(file_path)
        offset_filename = f"offset_fixed_samefile_{filename}"
        offset_path = os.path.join(target_dir, offset_filename)

        offset_df.to_csv(offset_path, index=False)

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Offset-fil skapad och flyttad",
            f"Offset-fil:\n{offset_filename}\nhar flyttats till '{target_dir}'."
        )
        root.destroy()

        plt.close(fig)  # Stäng figuren för att fortsätta i koden

    def on_move_to_trash(event):
        """
        Flyttar originalfilen till 'papperskorg', stänger plott-fönstret.
        """
        trash_dir = 'papperskorg'
        if not os.path.exists(trash_dir):
            os.makedirs(trash_dir)

        target_path = os.path.join(trash_dir, os.path.basename(file_path))
        shutil.move(file_path, target_path)

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Originalfil flyttad",
            f"Originalfilen:\n{os.path.basename(file_path)}\n"
            f"har flyttats till '{trash_dir}'."
        )
        root.destroy()

        plt.close(fig)  # Stäng figuren för att fortsätta i koden

    # Knappar
    ax_button_offset = plt.axes([0.68, 0.01, 0.3, 0.05])
    button_offset = Button(ax_button_offset, 'Ready to Label')
    button_offset.on_clicked(on_ready_to_label)

    ax_button_trash = plt.axes([0.35, 0.01, 0.3, 0.05])
    button_trash = Button(ax_button_trash, 'Flytta till Papperskorg')
    button_trash.on_clicked(on_move_to_trash)

    plt.show()

def main():
    """
    Huvudfunktion som låter användaren välja antingen en fil eller en mapp
    (båda dialogerna startar i skriptets arbetskatalog).
    - Om fil: Bearbetar bara den.
    - Om ingen fil valts -> be om mapp.
    - Om mapp: Bearbetar alla .csv-filer i mappen i tur och ordning.
    """
    root = tk.Tk()
    root.withdraw()

    # Fil-dialog för att välja FIL (med initialdir i samma mapp som programmet körs)
    path = fd.askopenfilename(
        title="Välj EN fil (eller Avbryt för att välja en mapp istället)",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        initialdir=os.getcwd()
    )

    if not path:
        # Ingen fil valdes, fråga då om mapp istället
        folder_path = fd.askdirectory(
            title="Välj en mapp med CSV-filer",
            initialdir=os.getcwd()
        )
        if folder_path:
            if os.path.isdir(folder_path):
                # Hämta alla csv-filer i mappen
                all_files = [os.path.join(folder_path, f)
                             for f in os.listdir(folder_path)
                             if f.lower().endswith(".csv")]
                all_files.sort()

                if not all_files:
                    messagebox.showwarning(
                        "Inga CSV-filer",
                        "Ingen CSV-fil hittades i mappen."
                    )
                else:
                    for csv_file in all_files:
                        process_file(csv_file)
            else:
                messagebox.showerror(
                    "Fel",
                    "Mappen existerar inte eller är ogiltig."
                )
        else:
            # Användare avbröt helt
            messagebox.showwarning(
                "Avbrutet",
                "Varken fil eller mapp valdes."
            )
    else:
        # Användaren valde en fil
        if os.path.isfile(path):
            process_file(path)
        else:
            messagebox.showerror(
                "Fel",
                "Vald sökväg är ingen fil."
            )

    sys.exit(0)

if __name__ == "__main__":
    main()
