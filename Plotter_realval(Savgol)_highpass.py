import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

file_path = 'stairs_down_data_20250127_141140.csv'
df = pd.read_csv(file_path)

ACC_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0
G_MS2 = 9.81

acc_x_ms2 = (df['acceleration_x'] / ACC_LSB_PER_G) * G_MS2
acc_y_ms2 = (df['acceleration_y'] / ACC_LSB_PER_G) * G_MS2
acc_z_ms2 = (df['acceleration_z'] / ACC_LSB_PER_G) * G_MS2

gyro_x_dps = df['gyroscope_x'] / GYRO_LSB_PER_DPS
gyro_y_dps = df['gyroscope_y'] / GYRO_LSB_PER_DPS
gyro_z_dps = df['gyroscope_z'] / GYRO_LSB_PER_DPS

fs = 66.5
cutoff_freq = 0.0000000000000000000000001

acc_x_hp = butter_highpass_filter(acc_x_ms2, cutoff_freq, fs)
acc_y_hp = butter_highpass_filter(acc_y_ms2, cutoff_freq, fs)
acc_z_hp = butter_highpass_filter(acc_z_ms2, cutoff_freq, fs)

gyro_x_hp = butter_highpass_filter(gyro_x_dps, cutoff_freq, fs)
gyro_y_hp = butter_highpass_filter(gyro_y_dps, cutoff_freq, fs)
gyro_z_hp = butter_highpass_filter(gyro_z_dps, cutoff_freq, fs)

x = range(len(df))

plt.figure(figsize=(10, 12))

plt.subplot(6, 1, 1)
plt.plot(x, acc_x_hp)
plt.title('Acc X HP')
plt.xlabel('Index')
plt.ylabel('m/s^2')
plt.grid(True)

plt.subplot(6, 1, 2)
plt.plot(x, acc_y_hp)
plt.title('Acc Y HP')
plt.xlabel('Index')
plt.ylabel('m/s^2')
plt.grid(True)

plt.subplot(6, 1, 3)
plt.plot(x, acc_z_hp)
plt.title('Acc Z HP')
plt.xlabel('Index')
plt.ylabel('m/s^2')
plt.grid(True)

plt.subplot(6, 1, 4)
plt.plot(x, gyro_x_hp)
plt.title('Gyro X HP')
plt.xlabel('Index')
plt.ylabel('°/s')
plt.grid(True)

plt.subplot(6, 1, 5)
plt.plot(x, gyro_y_hp)
plt.title('Gyro Y HP')
plt.xlabel('Index')
plt.ylabel('°/s')
plt.grid(True)

plt.subplot(6, 1, 6)
plt.plot(x, gyro_z_hp)
plt.title('Gyro Z HP')
plt.xlabel('Index')
plt.ylabel('°/s')
plt.grid(True)

plt.tight_layout()
plt.show()
