import os
import glob
import datetime
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def merge_csv_files():
    # Initialize Tkinter and hide the main window.
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select a directory containing CSV files.
    folder_path = filedialog.askdirectory(title="Select Folder with CSV Files")
    if not folder_path:
        print("No folder selected.")
        return

    # Find all CSV files in the selected folder.
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print("No CSV files found in the selected folder.")
        return

    # Read and concatenate all CSV files.
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dataframes:
        print("No CSV files were successfully read.")
        return

    merged_df = pd.concat(dataframes, ignore_index=True)

    # Create 'Merged Data' folder if it doesn't exist.
    output_folder = os.path.join(folder_path, "Merged Data")
    os.makedirs(output_folder, exist_ok=True)

    # Get current time formatted as H-M-S (Windows does not allow ':' in filenames).
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    output_filename = f"Merged_Data_{current_time}.csv"
    output_path = os.path.join(output_folder, output_filename)

    # Save the merged DataFrame to CSV.
    merged_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved to: {output_path}")

if __name__ == "__main__":
    merge_csv_files()