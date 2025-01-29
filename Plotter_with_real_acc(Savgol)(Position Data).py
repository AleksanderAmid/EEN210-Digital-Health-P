import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

file_path = 'data2\offset_fixed_fall_zipperUp_data_20250129_110818.csv'
df = pd.read_csv(file_path)

ACC_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0
G_MS2 = 9.81
SAMPLE_RATE = 66.5
time = [i / SAMPLE_RATE for i in range(len(df))]

acc_x_ms2 = (df['acceleration_x'] / ACC_LSB_PER_G) * G_MS2
acc_y_ms2 = (df['acceleration_y'] / ACC_LSB_PER_G) * G_MS2
acc_z_ms2 = (df['acceleration_z'] / ACC_LSB_PER_G) * G_MS2

gyro_x_dps = df['gyroscope_x'] / GYRO_LSB_PER_DPS
gyro_y_dps = df['gyroscope_y'] / GYRO_LSB_PER_DPS
gyro_z_dps = df['gyroscope_z'] / GYRO_LSB_PER_DPS

window_length = 11
poly_order = 2

acc_x_filt = savgol_filter(acc_x_ms2, window_length, poly_order)
acc_y_filt = savgol_filter(acc_y_ms2, window_length, poly_order)
acc_z_filt = savgol_filter(acc_z_ms2, window_length, poly_order)

gyro_x_filt = savgol_filter(gyro_x_dps, window_length, poly_order)
gyro_y_filt = savgol_filter(gyro_y_dps, window_length, poly_order)
gyro_z_filt = savgol_filter(gyro_z_dps, window_length, poly_order)

# Integrate gyroscope data once (degrees/second -> degrees)
angle_x = np.cumsum(gyro_x_filt) / SAMPLE_RATE
angle_y = np.cumsum(gyro_y_filt) / SAMPLE_RATE
angle_z = np.cumsum(gyro_z_filt) / SAMPLE_RATE

# Double-integrate acceleration data (m/s^2 -> m)
vel_x = np.cumsum(acc_x_filt) / SAMPLE_RATE
vel_y = np.cumsum(acc_y_filt) / SAMPLE_RATE
vel_z = np.cumsum(acc_z_filt) / SAMPLE_RATE
dist_x = np.cumsum(vel_x) / SAMPLE_RATE
dist_y = np.cumsum(vel_y) / SAMPLE_RATE
dist_z = np.cumsum(vel_z) / SAMPLE_RATE

plt.figure(figsize=(10, 10))

# 1) Filtered acceleration
plt.subplot(4, 1, 1)
plt.plot(time, acc_x_filt, label='Acc X (filt)')
plt.plot(time, acc_y_filt, label='Acc Y (filt)')
plt.plot(time, acc_z_filt, label='Acc Z (filt)')
plt.title('Filtrerad Accelerationsdata (m/s^2)')
plt.xlabel('Tid (s)')
plt.ylabel('Acceleration')
plt.legend()
plt.grid(True)

# 2) Filtered gyroscope
plt.subplot(4, 1, 2)
plt.plot(time, gyro_x_filt, label='Gyro X (filt)')
plt.plot(time, gyro_y_filt, label='Gyro Y (filt)')
plt.plot(time, gyro_z_filt, label='Gyro Z (filt)')
plt.title('Filtrerad Gyroskopdata (°/s)')
plt.xlabel('Tid (s)')
plt.ylabel('Vinkelhastighet')
plt.legend()
plt.grid(True)

# 3) Integrated gyro angles
plt.subplot(4, 1, 3)
plt.plot(time, angle_x, label='Gyro X (deg)')
plt.plot(time, angle_y, label='Gyro Y (deg)')
plt.plot(time, angle_z, label='Gyro Z (deg)')
plt.title('Integrerad Gyroskopdata (grader)')
plt.xlabel('Tid (s)')
plt.ylabel('Vinkel (°)')
plt.legend()
plt.grid(True)

# 4) Double-integrated acceleration (distance)
plt.subplot(4, 1, 4)
plt.plot(time, dist_x, label='Dist X (m)')
plt.plot(time, dist_y, label='Dist Y (m)')
plt.plot(time, dist_z, label='Dist Z (m)')
plt.title('Dubbelintegrerad Accelerationsdata (meter)')
plt.xlabel('Tid (s)')
plt.ylabel('Avstånd (m)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
