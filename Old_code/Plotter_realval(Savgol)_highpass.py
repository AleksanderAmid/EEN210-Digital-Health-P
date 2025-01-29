import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt

file_path = 'data3\gait_sit_squat_jump_fall_data_20250129_114859.csv'
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

# Highpass filter
def highpass_filter(data, cutoff=0.1, fs=SAMPLE_RATE, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

acc_x_hp = highpass_filter(acc_x_filt)
acc_y_hp = highpass_filter(acc_y_filt)
acc_z_hp = highpass_filter(acc_z_filt)

gyro_x_hp = highpass_filter(gyro_x_filt)
gyro_y_hp = highpass_filter(gyro_y_filt)
gyro_z_hp = highpass_filter(gyro_z_filt)

# Integrate gyroscope data once (degrees/second -> degrees)
angle_x = np.cumsum(gyro_x_hp) / SAMPLE_RATE
angle_y = np.cumsum(gyro_y_hp) / SAMPLE_RATE
angle_z = np.cumsum(gyro_z_hp) / SAMPLE_RATE

# Double-integrate acceleration data (m/s^2 -> m)
vel_x = np.cumsum(acc_x_hp) / SAMPLE_RATE
vel_y = np.cumsum(acc_y_hp) / SAMPLE_RATE
vel_z = np.cumsum(acc_z_hp) / SAMPLE_RATE
dist_x = np.cumsum(vel_x) / SAMPLE_RATE
dist_y = np.cumsum(vel_y) / SAMPLE_RATE
dist_z = np.cumsum(vel_z) / SAMPLE_RATE

plt.figure(figsize=(10, 10))

# 1) Highpass-filtered acceleration
plt.subplot(4, 1, 1)
plt.plot(time, acc_x_hp, label='Acc X (HP)')
plt.plot(time, acc_y_hp, label='Acc Y (HP)')
plt.plot(time, acc_z_hp, label='Acc Z (HP)')
plt.title('Highpass-filtered Accelerationsdata (m/s²)')
plt.xlabel('Tid (s)')
plt.ylabel('Acceleration')
plt.legend()
plt.grid(True)

# 2) Highpass-filtered gyroscope
plt.subplot(4, 1, 2)
plt.plot(time, gyro_x_hp, label='Gyro X (HP)')
plt.plot(time, gyro_y_hp, label='Gyro Y (HP)')
plt.plot(time, gyro_z_hp, label='Gyro Z (HP)')
plt.title('Highpass-filtered Gyroskopdata (°/s)')
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
