import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Filväg till din CSV-fil
file_path = 'stairs_down_data_20250127_141140.csv'

# 1. Läs in data
df = pd.read_csv(file_path)

# 2. Konstanter för omvandling (MPU6050-liknande, ±2g och ±250°/s)
ACC_LSB_PER_G = 16384.0    # MPU6050 ±2g
GYRO_LSB_PER_DPS = 131.0   # MPU6050 ±250°/s
G_MS2 = 9.81               # 1 g ≈ 9.81 m/s^2

# 3. Omvandla accelerationsvärden till m/s^2
acc_x_ms2 = (df['acceleration_x'] / ACC_LSB_PER_G) * G_MS2
acc_y_ms2 = (df['acceleration_y'] / ACC_LSB_PER_G) * G_MS2
acc_z_ms2 = (df['acceleration_z'] / ACC_LSB_PER_G) * G_MS2

# 4. Omvandla gyrovärden till grader/sekund
gyro_x_dps = df['gyroscope_x'] / GYRO_LSB_PER_DPS
gyro_y_dps = df['gyroscope_y'] / GYRO_LSB_PER_DPS
gyro_z_dps = df['gyroscope_z'] / GYRO_LSB_PER_DPS
#
# 5. Skapa en x-axel (index 0 till antal rader - 1)
x = range(len(df))

# 6. Parametrar för Savitzky-Golay (måste vara udda window_length)
window_length = 20
poly_order = 3

# 7. Filtrera accelerationsdata (i m/s^2)
acc_x_filt = savgol_filter(acc_x_ms2, window_length, poly_order)
acc_y_filt = savgol_filter(acc_y_ms2, window_length, poly_order)
acc_z_filt = savgol_filter(acc_z_ms2, window_length, poly_order)

# 8. Filtrera gyroskopdata (i °/s)
gyro_x_filt = savgol_filter(gyro_x_dps, window_length, poly_order)
gyro_y_filt = savgol_filter(gyro_y_dps, window_length, poly_order)
gyro_z_filt = savgol_filter(gyro_z_dps, window_length, poly_order)

# 9. Plotting
plt.figure(figsize=(10, 6))

# Subplot 1: Accelerationsdata (m/s^2, filtrerad)
plt.subplot(2, 1, 1)
plt.plot(x, acc_x_filt, label='Acc X (m/s^2, filt)')
plt.plot(x, acc_y_filt, label='Acc Y (m/s^2, filt)')
plt.plot(x, acc_z_filt, label='Acc Z (m/s^2, filt)')
plt.title('Filtrerad Accelerationsdata (m/s^2) - Savitzky-Golay')
plt.xlabel('Provdatalängd (index)')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
plt.grid(True)

# Subplot 2: Gyroskopdata (°/s, filtrerad)
plt.subplot(2, 1, 2)
plt.plot(x, gyro_x_filt, label='Gyro X (°/s, filt)')
plt.plot(x, gyro_y_filt, label='Gyro Y (°/s, filt)')
plt.plot(x, gyro_z_filt, label='Gyro Z (°/s, filt)')
plt.title('Filtrerad Gyroskopdata (°/s) - Savitzky-Golay')
plt.xlabel('Provdatalängd (index)')
plt.ylabel('Vinkelhastighet (°/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
