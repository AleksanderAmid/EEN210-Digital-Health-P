import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
#
# Filväg till din CSV-fil
file_path = 'fall_data_20250127_01.csv'

# Läs in data
df = pd.read_csv(file_path)

# Skapa en x-axel (index 0 till antal rader - 1)
x = range(len(df))

# Välj parametrar för Savitzky-Golay
# Notera att window_length måste vara ett udda tal och <= längden på data
window_length = 20
poly_order = 3

# Filtrera accelerationsdata
acc_x_filt = savgol_filter(df['acceleration_x'], window_length, poly_order)
acc_y_filt = savgol_filter(df['acceleration_y'], window_length, poly_order)
acc_z_filt = savgol_filter(df['acceleration_z'], window_length, poly_order)

# Filtrera gyroskopdata
gyro_x_filt = savgol_filter(df['gyroscope_x'], window_length, poly_order)
gyro_y_filt = savgol_filter(df['gyroscope_y'], window_length, poly_order)
gyro_z_filt = savgol_filter(df['gyroscope_z'], window_length, poly_order)

plt.figure(figsize=(10, 6))

# Subplot 1: Accelerationsdata (filtrerad)
plt.subplot(2, 1, 1)
plt.plot(x, acc_x_filt, label='Acc X (Filt)')
plt.plot(x, acc_y_filt, label='Acc Y (Filt)')
plt.plot(x, acc_z_filt, label='Acc Z (Filt)')
plt.title('Filtrerad Accelerationsdata - Savitzky-Golay')
plt.xlabel('Provdatalängd (index)')
plt.ylabel('Acceleration')
plt.legend()
plt.grid(True)

# Subplot 2: Gyroskopdata (filtrerad)
plt.subplot(2, 1, 2)
plt.plot(x, gyro_x_filt, label='Gyro X (Filt)')
plt.plot(x, gyro_y_filt, label='Gyro Y (Filt)')
plt.plot(x, gyro_z_filt, label='Gyro Z (Filt)')
plt.title('Filtrerad Gyroskopdata - Savitzky-Golay')
plt.xlabel('Provdatalängd (index)')
plt.ylabel('Vinkelhastighet')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
