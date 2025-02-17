import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import os

def load_data():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select CSV file for analysis",
        filetypes=[("CSV files", "*.csv")]
    )
    
    if not file_path:
        print("No file selected. Exiting...")
        return None
        
    return pd.read_csv(file_path)

def analyze_distribution():
    # Load the data
    print("Please select your CSV file...")
    data = load_data()
    if data is None:
        return
    
    # Check if 'label' column exists
    if 'label' not in data.columns:
        print("Error: No 'label' column found in the CSV file!")
        return
    
    # Calculate distribution
    label_dist = data['label'].value_counts().sort_index()
    total_samples = len(data)
    
    # Print numerical distribution
    print("\nLabel Distribution:")
    print("-" * 40)
    print(f"{'Label':<10} {'Count':<10} {'Percentage':<10}")
    print("-" * 40)
    for label, count in label_dist.items():
        percentage = (count / total_samples) * 100
        print(f"{label:<10} {count:<10} {percentage:.2f}%")
    
    print("\nTotal samples:", total_samples)
    
    # Create visualization directory if it doesn't exist
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Distribution_Results")
    os.makedirs(result_dir, exist_ok=True)
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_dist.index, y=label_dist.values)
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    
    # Add value labels on top of each bar
    for i, v in enumerate(label_dist.values):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'label_distribution_bar.png'))
    plt.close()
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(label_dist.values, labels=[f'Label {l} ({v})' for l, v in label_dist.items()],
            autopct='%1.1f%%', startangle=90)
    plt.title('Label Distribution (Pie Chart)')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'label_distribution_pie.png'))
    plt.close()
    
    print(f"\nVisualization plots saved in: {result_dir}")

if __name__ == "__main__":
    analyze_distribution()