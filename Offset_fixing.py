import os
import pandas as pd

# Define file paths
still_file = r'data2\still_data_20250129_102942.csv'
real_file = r'data2\fall_zipperUp_data_20250129_110818.csv'

def apply_offset(still_file, real_file):
    still_data = pd.read_csv(still_file)
    # Select only sensor columns
    sensor_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                      'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    offset_means = still_data[sensor_columns].mean()
    
    real_data = pd.read_csv(real_file)
    real_data[sensor_columns] = real_data[sensor_columns] - offset_means
    
    directory = os.path.dirname(real_file)
    filename = os.path.basename(real_file)
    new_filename = os.path.join(directory, f"offset_fixed_{filename}")
    real_data.to_csv(new_filename, index=False)

apply_offset(still_file, real_file)