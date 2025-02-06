import os
import sys
import shutil
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import messagebox
from matplotlib.widgets import Button

# Constant for time axis calculation (if your CSV is sampled at this rate)
SAMPLE_RATE = 66.5

def process_file(file_path: str):
    """
    Reads a CSV file, plots the raw sensor data (accelerometer and gyroscope),
    and provides two buttons:
      - "Ready to Label": Moves the file to the 'readyToLabel2' folder.
      - "Flytta till Papperskorg": Moves the file to the 'papperskorg' folder.
    """
    print(f"\n--- Processing file: {file_path} ---")
    
    # 1) Read data
    df = pd.read_csv(file_path)
    
    # 2) Create a time axis based on the sample rate
    time = np.arange(len(df)) / SAMPLE_RATE
    
    # 3) Extract raw sensor data
    acc_x = df['acceleration_x']
    acc_y = df['acceleration_y']
    acc_z = df['acceleration_z']
    
    gyro_x = df['gyroscope_x']
    gyro_y = df['gyroscope_y']
    gyro_z = df['gyroscope_z']
    
    # 4) Plot the raw sensor data
    fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title(file_path)
    
    ax_acc.plot(time, acc_x, label='Acc X')
    ax_acc.plot(time, acc_y, label='Acc Y')
    ax_acc.plot(time, acc_z, label='Acc Z')
    ax_acc.set_title('Accelerometer Data (Raw)')
    ax_acc.set_xlabel('Time (s)')
    ax_acc.set_ylabel('Acceleration')
    ax_acc.legend()
    ax_acc.grid(True)
    
    ax_gyro.plot(time, gyro_x, label='Gyro X')
    ax_gyro.plot(time, gyro_y, label='Gyro Y')
    ax_gyro.plot(time, gyro_z, label='Gyro Z')
    ax_gyro.set_title('Gyroscope Data (Raw)')
    ax_gyro.set_xlabel('Time (s)')
    ax_gyro.set_ylabel('Angular Velocity')
    ax_gyro.legend()
    ax_gyro.grid(True)
    
    plt.tight_layout()
    
    # Callback for "Ready to Label": move file to 'readyToLabel2'
    def on_ready_to_label(event):
        target_dir = 'readyToLabel2'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        filename = os.path.basename(file_path)
        target_path = os.path.join(target_dir, filename)
        shutil.move(file_path, target_path)
        
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "File Moved",
            f"File:\n{filename}\nhas been moved to '{target_dir}'."
        )
        root.destroy()
        plt.close(fig)
    
    # Callback for "Flytta till Papperskorg": move file to 'papperskorg'
    def on_move_to_trash(event):
        trash_dir = 'papperskorg'
        if not os.path.exists(trash_dir):
            os.makedirs(trash_dir)
        target_path = os.path.join(trash_dir, os.path.basename(file_path))
        shutil.move(file_path, target_path)
        
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "File Moved",
            f"File:\n{os.path.basename(file_path)}\nhas been moved to '{trash_dir}'."
        )
        root.destroy()
        plt.close(fig)
    
    # Create buttons for user choice
    ax_button_ready = plt.axes([0.68, 0.01, 0.3, 0.05])
    button_ready = Button(ax_button_ready, 'Ready to Label')
    button_ready.on_clicked(on_ready_to_label)
    
    ax_button_trash = plt.axes([0.35, 0.01, 0.3, 0.05])
    button_trash = Button(ax_button_trash, 'Flytta till Papperskorg')
    button_trash.on_clicked(on_move_to_trash)
    
    plt.show()

def main():
    """
    Main function that asks the user if they want to process a single file or a folder of files.
    """
    root = tk.Tk()
    root.withdraw()
    
    choice = messagebox.askquestion(
        "Choose Source",
        "Yes = Single file, No = Folder with files"
    )
    
    if choice == 'yes':
        # User selected a single file
        path = fd.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialdir=os.getcwd()
        )
        if path and os.path.isfile(path):
            process_file(path)
        else:
            messagebox.showwarning("Warning", "No valid file selected.")
    else:
        # User selected a folder
        folder_path = fd.askdirectory(
            title="Select a folder with CSV files",
            initialdir=os.getcwd()
        )
        if folder_path and os.path.isdir(folder_path):
            all_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(".csv")
            ]
            all_files.sort()
            if not all_files:
                messagebox.showwarning("No CSV Files", "No CSV file found in the folder.")
            else:
                for csv_file in all_files:
                    process_file(csv_file)
        else:
            messagebox.showwarning("Cancelled", "No valid folder selected.")
    
    sys.exit(0)

if __name__ == "__main__":
    main()
