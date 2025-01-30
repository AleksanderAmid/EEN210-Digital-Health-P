import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import mplcursors

def compute_fft(signal, fs):
    """
    Beräknar enkel-sidig FFT av en realvärdesignal.
    
    Parametrar:
        signal (array-like): Signalen som ska transformeras.
        fs (float): Samplingsfrekvens i Hz.
        
    Returnerar:
        freqs (1D-array): Frekvensaxel (bara positiva frekvenser).
        spectrum (1D-array): FFT-amplitud för respektive frekvens.
    """
    N = len(signal)
    # Beräkna FFT
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1/fs)  # d=1/fs => samplingsperiod
    
    # Välj endast den positiva delen av frekvensspektrumet
    idx_pos = freqs >= 0
    freqs = freqs[idx_pos]
    
    # Amplitudspektrum (skalar med 2/N för den enkel-sidiga varianten,
    # men observera att DC-komponenten inte dubblas — i många analyser 
    # är det dock inget problem om man vill snabbplotta.)
    X_mag = np.abs(X[idx_pos]) * (2.0 / N)
    
    return freqs, X_mag

# =================================================================
# Exempel på hur man kan använda funktionen
# =================================================================

# 1. Läs in data via filutforskaren
root = Tk()
root.withdraw()  # Dölj huvudfönstret
file_path = askopenfilename(title="Välj en CSV-fil", filetypes=[("CSV files", "*.csv")])

# Kontrollera om en fil valdes
if not file_path:
    print("Ingen fil valdes.")
    exit()

df = pd.read_csv(file_path)

# 2. Ange eller uppskatta din samplingsfrekvens (Hz).
fs = 50.0

# 3. Hämta signalerna
acc_x = df['acceleration_x'].values
acc_y = df['acceleration_y'].values
acc_z = df['acceleration_z'].values

gyro_x = df['gyroscope_x'].values
gyro_y = df['gyroscope_y'].values
gyro_z = df['gyroscope_z'].values

# 4. Beräkna FFT för respektive signal (acc och gyro)
freq_acc_x, spec_acc_x = compute_fft(acc_x, fs)
freq_acc_y, spec_acc_y = compute_fft(acc_y, fs)
freq_acc_z, spec_acc_z = compute_fft(acc_z, fs)

freq_gyro_x, spec_gyro_x = compute_fft(gyro_x, fs)
freq_gyro_y, spec_gyro_y = compute_fft(gyro_y, fs)
freq_gyro_z, spec_gyro_z = compute_fft(gyro_z, fs)

# 5. Plotta accelerations- och gyrosignaler i frekvensdomän
plt.figure(figsize=(10, 6))

# -- Subplot för accelerations-Fourier --
plt.subplot(2, 1, 1)
plt.plot(freq_acc_x, spec_acc_x, label='Acc X')
plt.plot(freq_acc_y, spec_acc_y, label='Acc Y')
plt.plot(freq_acc_z, spec_acc_z, label='Acc Z')
plt.title("Fouriertransform - Acceleration")
plt.xlabel("Frekvens (Hz)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

# -- Subplot för gyroskop-Fourier --
plt.subplot(2, 1, 2)
plt.plot(freq_gyro_x, spec_gyro_x, label='Gyro X')
plt.plot(freq_gyro_y, spec_gyro_y, label='Gyro Y')
plt.plot(freq_gyro_z, spec_gyro_z, label='Gyro Z')
plt.title("Fouriertransform - Gyroskop")
plt.xlabel("Frekvens (Hz)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Add interactive cursor
mplcursors.cursor(hover=True)

plt.show()