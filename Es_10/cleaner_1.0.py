import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def process_and_clean_data(filename, sigma_threshold=1.5):
    # 1. Load Data
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
            _, x_raw, y_raw, z_raw, timestamp = struct.unpack_from(packet_fmt, content, i)
            # Endianness swap for Big-Endian MPU-6050 registers
            x = struct.unpack(">h", struct.pack("<h", x_raw))[0]
            y = struct.unpack(">h", struct.pack("<h", y_raw))[0]
            z = struct.unpack(">h", struct.pack("<h", z_raw))[0]
            results.append([timestamp / 1e6, x, y, z])
            i += packet_size 
        else:
            i += 1 

    data = np.array(results)
    t_raw = data[:, 0]
    raw_signals = [data[:, 1], data[:, 2], data[:, 3]] # X, Y, Z

    # 2. Statistical Masking (Removes the Vertical Bars/Spikes)
    # We create a mask for points that are NOT outliers
    clean_masks = []
    for sig in raw_signals:
        median = np.median(sig)
        std = np.std(sig)
        clean_masks.append(np.abs(sig - median) < (sigma_threshold * std))
    
    # Combined mask: point is good only if all 3 axes are within threshold
    final_mask = clean_masks[0] & clean_masks[1] & clean_masks[2]
    
    t_masked = t_raw[final_mask]
    
    # 3. Regularization (Linear Interpolation for FFT/Integration)
    # This creates a perfectly even 200Hz clock
    t_regular = np.linspace(t_masked.min(), t_masked.max(), len(t_masked))
    clean_signals = []
    
    for i in range(3):
        sig_masked = raw_signals[i][final_mask]
        f_interp = interp1d(t_masked, sig_masked, kind='linear')
        clean_signals.append(f_interp(t_regular))

    return t_raw, raw_signals, t_regular, clean_signals

# --- Execution ---
file_path = "/home/marco/Desktop/Uni_anno3/TD/Es_10/TestBusI2C/DATA000_f1_780.bin"
t_raw, s_raw, t_clean, s_clean = process_and_clean_data(file_path)

# --- Comparison Plotting ---
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
labels = ['X-Axis', 'Y-Axis', 'Z-Axis']
colors = ['red', 'green', 'blue']

for i in range(3):
    # Plot original data with spikes in light color
    axs[i].scatter(t_raw, s_raw[i], marker='.', s=0.5, color='black', alpha=0.3, label='Raw (with Spikes)')
    # Plot cleaned/interpolated data in bold color
    axs[i].scatter(t_clean, s_clean[i], marker='.', s=0.5, color=colors[i], label='Cleaned & Regularized', linewidth=1)

    axs[i].set_ylabel(f"{labels[i]}\nRaw Value")
    axs[i].legend(loc='upper right', fontsize='small')
    axs[i].grid(True, which='both', linestyle='--', alpha=0.5)

plt.xlabel("Time (seconds)")
plt.suptitle("MPU-6050 Data Cleaning: Impact Removal for FFT and Integration")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()