import pandas as pd
import os

real_file = r'data3\gait_sit_squat_jump_fall_data_20250129_115216.csv'


def fix_offset(real_file):
    real_data = pd.read_csv(real_file)
    cols = ['acceleration_x','acceleration_y','acceleration_z',
            'gyroscope_x','gyroscope_y','gyroscope_z']
    offsets = real_data[cols].head(10).mean()
    real_data[cols] = real_data[cols] - offsets
    directory = os.path.dirname(real_file)
    filename = os.path.basename(real_file)
    new_filename = os.path.join(directory, f"offset_fixed_samefile{filename}")
    real_data.to_csv(new_filename, index=False)
    return new_filename
