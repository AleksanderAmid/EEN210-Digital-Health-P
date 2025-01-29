import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# File path to your CSV file
file_path = "fall_data_20250127_01.csv"

# Load the data
df = pd.read_csv(file_path)

# Convert all columns to numeric, forcing errors to NaN and dropping non-numeric columns
df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

# Constants for MPU6050 sensor calibration
ACC_LSB_PER_G = 16384.0  # MPU6050 ±2g
GYRO_LSB_PER_DPS = 131.0  # MPU6050 ±250°/s
G_MS2 = 9.81  # 1 g ≈ 9.81 m/s^2

# Remove initial offsets (subtract the first value in each column)
df_corrected = df - df.iloc[0]

# Convert acceleration values to m/s²
acc_x_ms2 = (df_corrected["acceleration_x"] / ACC_LSB_PER_G) * G_MS2
acc_y_ms2 = (df_corrected["acceleration_y"] / ACC_LSB_PER_G) * G_MS2
acc_z_ms2 = (df_corrected["acceleration_z"] / ACC_LSB_PER_G) * G_MS2

# Convert gyroscope values to degrees per second
gyro_x_dps = df_corrected["gyroscope_x"] / GYRO_LSB_PER_DPS
gyro_y_dps = df_corrected["gyroscope_y"] / GYRO_LSB_PER_DPS
gyro_z_dps = df_corrected["gyroscope_z"] / GYRO_LSB_PER_DPS

# Create an x-axis (index 0 to number of rows - 1)
x = range(len(df))

# Ensure window_length is valid for Savitzky-Golay filtering
window_length = min(20, len(df)) if len(df) > 2 else len(df) - 1  # Must be odd
if window_length % 2 == 0:
    window_length += 1  # Ensure it's odd
poly_order = min(3, window_length - 1)  # Ensure polynomial order is valid

# Apply Savitzky-Golay filtering
acc_x_filt = savgol_filter(acc_x_ms2, window_length, poly_order) if len(df) > window_length else acc_x_ms2
acc_y_filt = savgol_filter(acc_y_ms2, window_length, poly_order) if len(df) > window_length else acc_y_ms2
acc_z_filt = savgol_filter(acc_z_ms2, window_length, poly_order) if len(df) > window_length else acc_z_ms2

gyro_x_filt = savgol_filter(gyro_x_dps, window_length, poly_order) if len(df) > window_length else gyro_x_dps
gyro_y_filt = savgol_filter(gyro_y_dps, window_length, poly_order) if len(df) > window_length else gyro_y_dps
gyro_z_filt = savgol_filter(gyro_z_dps, window_length, poly_order) if len(df) > window_length else gyro_z_dps

# Plotting
plt.figure(figsize=(10, 6))

# Subplot 1: Filtered Acceleration Data (m/s²)
plt.subplot(2, 1, 1)
plt.plot(x, acc_x_filt, label="Acc X (m/s², filt)")
plt.plot(x, acc_y_filt, label="Acc Y (m/s², filt)")
plt.plot(x, acc_z_filt, label="Acc Z (m/s², filt)")
plt.title("Filtered Acceleration Data (m/s²) - Offset Removed & Savitzky-Golay")
plt.xlabel("Sample Index")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid(True)

# Subplot 2: Filtered Gyroscope Data (°/s)
plt.subplot(2, 1, 2)
plt.plot(x, gyro_x_filt, label="Gyro X (°/s, filt)")
plt.plot(x, gyro_y_filt, label="Gyro Y (°/s, filt)")
plt.plot(x, gyro_z_filt, label="Gyro Z (°/s, filt)")
plt.title("Filtered Gyroscope Data (°/s) - Offset Removed & Savitzky-Golay")
plt.xlabel("Sample Index")
plt.ylabel("Angular Velocity (°/s)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
