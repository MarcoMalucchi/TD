import numpy as np
import matplotlib.pyplot as plt
import struct
import os

# --- 1. SET YOUR PATHS HERE ---
filename = "DATA000_f0_300"  # Change this to your desired file base name
raw_bin_path = f"/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/{filename}.bin"
cleaned_csv_path = f"/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/cleaned_results/{filename}_CLEANED.csv"

def load_raw_bin(filename):
    """Loads the original dirty data from the .bin file"""
    packet_fmt = "<HhhhI"
    packet_size = struct.calcsize(packet_fmt)
    sync_word = 0xAAAA
    results = []
    with open(filename, "rb") as f:
        content = f.read()
    i = 0
    while i <= len(content) - packet_size:
        current_header, = struct.unpack_from("<H", content, i)
        if current_header == sync_word:
            _, x, y, z, timestamp = struct.unpack_from(packet_fmt, content, i)
            # Basic conversion to signed int (matching your original logic)
            x = struct.unpack(">h", struct.pack("<h", x))[0]
            y = struct.unpack(">h", struct.pack("<h", y))[0]
            z = struct.unpack(">h", struct.pack("<h", z))[0]
            results.append([timestamp / 1e6, x, y, z])
            i += packet_size
        else:
            i += 1
    data = np.array(results)
    data = data[data[:, 0].argsort()]
    return data[:, 0], [data[:, 1], data[:, 2], data[:, 3]]

def load_cleaned_csv(filename):
    """Loads the cleaned data from the .csv file"""
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    return data[:, 0], [data[:, 1], data[:, 2], data[:, 3]]

def get_fft(t, signal, fs=200):
    n = len(t)
    sig_detrended = signal - np.mean(signal) # Remove DC offset
    fft_values = np.fft.rfft(sig_detrended)
    fft_freqs = np.fft.rfftfreq(n, d=1/fs)
    amplitude = np.abs(fft_values) * (2.0 / n)
    return fft_freqs, amplitude

# --- Execution ---
try:
    # 2. Load the data
    t_raw, s_raw = load_raw_bin(raw_bin_path)
    t_clean, s_clean = load_cleaned_csv(cleaned_csv_path)

    # 3. Setup Plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes_labels = ['X-Axis', 'Y-Axis', 'Z-Axis']
    fs = 200 # Sampling frequency

    for i in range(3):
        # Calculate FFTs
        f_dirty, a_dirty = get_fft(t_raw, s_raw[i], fs=fs)
        f_clean, a_clean = get_fft(t_clean, s_clean[i], fs=fs)
        
        # Plotting
        axs[i].plot(f_dirty, a_dirty, color='gray', alpha=0.4, label='Original (Dirty)')
        axs[i].plot(f_clean, a_clean, color='blue', linewidth=1, label='Cleaned')
        
        axs[i].set_title(f"FFT Comparison: {axes_labels[i]}")
        axs[i].set_ylabel("Amplitude")
        axs[i].set_xlim(0, 60) # Structural focus
        axs[i].legend(loc='upper right')
        axs[i].grid(True, linestyle='--', alpha=0.5)

    plt.xlabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error: {e}. Check if your paths are correct!")