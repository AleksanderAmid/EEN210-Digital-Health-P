import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog

def load_data():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select CSV file to analyze",
        filetypes=[("CSV files", "*.csv")]
    )
    
    if not file_path:
        print("No file selected. Exiting...")
        return None
    
    return pd.read_csv(file_path)

def identify_problematic_rows(data):
    if data is None:
        return
    
    print("\nAnalyzing data...")
    print(f"Total rows in dataset: {len(data)}")
    
    # Find rows with NaN values
    rows_with_nan = data[data.isna().any(axis=1)]
    print(f"\nFound {len(rows_with_nan)} rows with NaN values:")
    if len(rows_with_nan) > 0:
        print("\nRows with NaN values:")
        for idx, row in rows_with_nan.iterrows():
            print(f"\nRow {idx}:")
            # Only print the columns that have NaN values
            nan_columns = row[row.isna()].index
            for col in nan_columns:
                print(f"{col}: NaN")
    
    # Find rows with non-numeric values (excluding the timestamp column if it exists)
    non_numeric_rows = pd.DataFrame()
    for column in data.columns:
        # Skip timestamp column
        if pd.to_datetime(data[column], errors='coerce').notna().any():
            continue
            
        # Try to convert to numeric
        numeric_conversion = pd.to_numeric(data[column], errors='coerce')
        non_numeric_mask = numeric_conversion.isna() & ~data[column].isna()
        if non_numeric_mask.any():
            non_numeric_in_column = data[non_numeric_mask]
            if len(non_numeric_in_column) > 0:
                print(f"\nFound non-numeric values in column '{column}':")
                for idx, row in non_numeric_in_column.iterrows():
                    print(f"Row {idx}: {row[column]}")
                non_numeric_rows = pd.concat([non_numeric_rows, non_numeric_in_column])
    
    # Remove duplicates from non_numeric_rows
    non_numeric_rows = non_numeric_rows.drop_duplicates()
    print(f"\nTotal rows with non-numeric values: {len(non_numeric_rows)}")
    
    # Summary
    total_problematic = len(pd.concat([rows_with_nan, non_numeric_rows]).drop_duplicates())
    print(f"\nSummary:")
    print(f"Total rows in dataset: {len(data)}")
    print(f"Rows with NaN values: {len(rows_with_nan)}")
    print(f"Rows with non-numeric values: {len(non_numeric_rows)}")
    print(f"Total problematic rows: {total_problematic}")
    print(f"Percentage of problematic rows: {(total_problematic/len(data))*100:.2f}%")

if __name__ == "__main__":
    print("Please select your CSV file to analyze...")
    data = load_data()
    identify_problematic_rows(data)