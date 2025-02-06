import pandas as pd
import tkinter as tk
from tkinter import filedialog
import sys

def process_csv(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # If the "label" column exists, set its value to 0 for all rows
    if 'label' in df.columns:
        df['label'] = 0
    else:
        print("WARNING: 'label' column not found in the CSV file.")
    
    # Save the modified DataFrame to the output CSV file
    df.to_csv(output_file, index=False)
    print(f"Processed CSV saved to {output_file}")

if __name__ == "__main__":
    # Initialize tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()
    
    # Ask user to select the input CSV file
    input_csv = filedialog.askopenfilename(
        title="Select Input CSV File",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not input_csv:
        print("No input file selected. Exiting.")
        sys.exit(1)
    
    # Ask user to select a path and name for the output CSV file
    output_csv = filedialog.asksaveasfilename(
        title="Save Processed CSV As",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not output_csv:
        print("No output file selected. Exiting.")
        sys.exit(1)
    
    process_csv(input_csv, output_csv)