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

    try:
        with open(filename, "rb") as f:
            content = f.read()
    except FileNotFoundError:
        print("File not found!")
        return None

    # --- BINARY PARSING ---
    i = 0
    while i <= len(content) - packet_size:
        current_header, = struct.unpack_from("<H", content, i)
        if current_header == sync_word:
            _, x_raw, y_raw, z_raw, timestamp = struct.unpack_from(packet_fmt, content, i)
            # Big-Endian swap for MPU-6050 registers
            x = struct.unpack(">h", struct.pack("<h", x_raw))[0]
            y = struct.unpack(">h", struct.pack("<h", y_raw))[0]
            z = struct.unpack(">h", struct.pack("<h", z_raw))[0]
            results.append([timestamp / 1e6, x, y, z])
            i += packet_size 
        else:
            i += 1 

    if not results:
        print("No valid packets found!")
        return None

    data = np.array(results)
    
    # --- FIX: Sort and Filter Time (Removes the 'Squeeze') ---
    data = data[data[:, 0].argsort()]
    t_full = data[:, 0]
    
    # Identify and remove large initialization gaps
    diffs = np.diff(t_full)
    if len(diffs) > 0 and np.max(diffs) > 1.0:
        gap_idx = np.where(diffs > 1.0)[0][-1]
        data = data[gap_idx + 1:]
    
    # Trim 5 samples from start/end to remove transients
    data = data[5:-5] 
    
    t_raw = data[:, 0]
    sigs_raw = [data[:, 1], data[:, 2], data[:, 3]]

    # --- 2. Statistical Masking (Cleaning Spikes) ---
    clean_masks = []
    for sig in sigs_raw:
        median = np.median(sig)
        std = np.std(sig)
        clean_masks.append(np.abs(sig - median) < (sigma_threshold * std))
    
    final_mask = clean_masks[0] & clean_masks[1] & clean_masks[2]
    
    t_masked = t_raw[final_mask]
    
    # --- 3. Regularization (Interpolation) ---
    # Create a perfectly even clock based on the cleaned data range
    t_regular = np.linspace(t_masked.min(), t_masked.max(), len(t_masked))
    clean_signals = []
    
    for i in range(3):
        sig_masked = sigs_raw[i][final_mask]
        f_interp = interp1d(t_masked, sig_masked, kind='linear', fill_value="extrapolate")
        clean_signals.append(f_interp(t_regular))

    return t_raw, sigs_raw, t_regular, clean_signals

# --- Execution ---
file_path = "/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/DATA001_f0_300.bin"
output = process_and_clean_data(file_path)

if output:
    t_raw, s_raw, t_clean, s_clean = output

    # --- Comparison Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    labels = ['X-Axis', 'Y-Axis', 'Z-Axis']
    colors = ['red', 'green', 'blue']

    for i in range(3):
        # Scatter for raw to see spikes clearly
        axs[i].scatter(t_raw, s_raw[i], s=1, color='black', alpha=0.2, label='Raw')
        # Line for cleaned data
        axs[i].plot(t_clean, s_clean[i], color=colors[i], label='Cleaned', linewidth=1)

        axs[i].set_ylabel(f"{labels[i]}\nRaw Value")
        axs[i].legend(loc='upper right', fontsize='small')
        axs[i].grid(True, alpha=0.3)

    plt.xlabel("Time (seconds)")
    plt.suptitle("MPU-6050 Cleaned Data: No More Squeeze")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print(f"Time Range: Min={t_raw.min():.2f}s, Max={t_raw.max():.2f}s, Duration={t_raw.max()-t_raw.min():.2f}s")
    plt.show()