import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from itertools import groupby

def process_clean_advanced(filename, sigma_threshold=2.5):
    # 1. Standard Loading
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
            # Big-Endian swap for MPU-6050
            x = struct.unpack(">h", struct.pack("<h", x_raw))[0]
            y = struct.unpack(">h", struct.pack("<h", y_raw))[0]
            z = struct.unpack(">h", struct.pack("<h", z_raw))[0]
            results.append([timestamp / 1e6, x, y, z])
            i += packet_size 
        else:
            i += 1 

    if not results:
        print("Error: No valid packets found.")
        return None

    data = np.array(results)
    
    # 2. SANITIZATION: Ensure strictly increasing timestamps
    # First, sort by timestamp just in case they arrived out of order
    data = data[data[:, 0].argsort()]
    
    # Second, remove duplicate timestamps (keeps only the first occurrence)
    _, unique_indices = np.unique(data[:, 0], return_index=True)
    data = data[unique_indices]
    
    # Optional: Skip first/last elements if they are anomalous 
    # (Handling that "big increasing start/end" you mentioned)
    data = data[1:-1] 

    t_raw = data[:, 0]
    x_raw, y_raw, z_raw = data[:, 1], data[:, 2], data[:, 3]

    # 3. Masking with Sigma Threshold
    def get_mask(sig):
        return np.abs(sig - np.median(sig)) < (sigma_threshold * np.std(sig))

    mask = get_mask(x_raw) & get_mask(y_raw) & get_mask(z_raw)
    
    # Safety check: Ensure we have enough points for a spline
    if np.sum(mask) < 4:
        print("Error: Not enough good points found for interpolation.")
        return None

    # 4. Gap Filling using Cubic Spline
    # We use a perfectly regular 200Hz grid for the result
    t_fixed = np.linspace(t_raw.min(), t_raw.max(), len(t_raw))
    clean_signals = []
    
    # We must ensure t_raw[mask] is strictly increasing (already handled by sorting/unique)
    t_for_spline = t_raw[mask]
    
    for sig in [x_raw, y_raw, z_raw]:
        sig_for_spline = sig[mask]
        cs = CubicSpline(t_for_spline, sig_for_spline)
        clean_signals.append(cs(t_fixed))

    return t_raw, [x_raw, y_raw, z_raw], t_fixed, np.array(clean_signals), mask

def find_longest_segment(mask):
    idx = np.arange(len(mask))
    segments = [list(g) for k, g in groupby(idx, lambda x: mask[x]) if k]
    if not segments: return 0, 0
    longest = max(segments, key=len)
    return longest[0], longest[-1]

# --- Execution ---
# Replace with your actual path
file_path = "/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/DATA000_f0_300.bin"
processed = process_clean_advanced(file_path)

if processed:
    t_raw, s_raw, t_clean, s_clean, mask = processed
    start, end = find_longest_segment(mask)

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    labels = ['X-Axis', 'Y-Axis', 'Z-Axis']
    colors = ['#d62728', '#2ca02c', '#1f77b4']

    for i in range(3):
        # Discarded spikes in light black/gray
        axs[i].scatter(t_raw[~mask], s_raw[i][~mask], color='black', alpha=0.15, s=2, label='Discarded Spikes')
        # Cleaned Cubic Spline
        axs[i].plot(t_clean, s_clean[i], color=colors[i], linewidth=1.2, label='Cubic Spline (Cleaned)')
        # Optimal FFT Zone
        axs[i].axvspan(t_clean[start], t_clean[end], color='yellow', alpha=0.1, label='Best FFT Segment')

        axs[i].set_ylabel(f"{labels[i]}\nRaw Value")
        axs[i].grid(True, linestyle=':', alpha=0.6)
        axs[i].legend(loc='upper right', fontsize='x-small')

    plt.xlabel("Time (seconds)")
    plt.suptitle("Advanced Cleaning: Cubic Spline & Timestamp Sanitization")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    print(f"Longest continuous segment: {t_clean[end] - t_clean[start]:.2f} seconds.")