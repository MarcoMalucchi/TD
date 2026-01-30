import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.ndimage import median_filter, binary_dilation

def process_clean_two_stage(filename, global_sigma=5.0, local_sigma=1.5, window_size=51, interp_type='linear'):
    # --- 1. LOADING & CHRONOLOGICAL SORTING ---
    packet_fmt = "<HhhhI" 
    packet_size = struct.calcsize(packet_fmt)
    sync_word = 0xAAAA
    results = []

    try:
        with open(filename, "rb") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

    i = 0
    while i <= len(content) - packet_size:
        current_header, = struct.unpack_from("<H", content, i)
        if current_header == sync_word:
            _, x_raw, y_raw, z_raw, timestamp = struct.unpack_from(packet_fmt, content, i)
            x = struct.unpack(">h", struct.pack("<h", x_raw))[0]
            y = struct.unpack(">h", struct.pack("<h", y_raw))[0]
            z = struct.unpack(">h", struct.pack("<h", z_raw))[0]
            results.append([timestamp / 1e6, x, y, z])
            i += packet_size 
        else:
            i += 1 

    data = np.array(results)
    data = data[data[:, 0].argsort()] 
    
    t_full = data[:, 0]
    diffs = np.diff(t_full)
    if len(diffs) > 0 and np.max(diffs) > 1.0:
        gap_idx = np.where(diffs > 1.0)[0][-1]
        data = data[gap_idx + 1:]
    
    data = data[5:-5] 
    t_raw = data[:, 0]
    sigs_raw = [data[:, 1], data[:, 2], data[:, 3]]

# --- STAGE 1: INTELLIGENT SPIKE DETECTION ---
    global_mask = np.ones(len(t_raw), dtype=bool)
    
    # 1. Identify "Digital Rails" (Values that are physically impossible for your sensor range)
    # Most MPU-6050 settings max out well before 32000
    for sig in sigs_raw:
        global_mask &= (np.abs(sig) < 32000)
        global_mask &= (np.abs(sig) > 0) # I2C glitches often drop to exactly 0

    # 2. Simultaneous Spike Detection (The "Cable Jiggle" logic)
    # If a spike happens on X, Y, and Z at the exact same time, it's the cable.
    cross_axis_spike = np.zeros(len(t_raw), dtype=bool)
    for sig in sigs_raw:
        # We look for points that are huge outliers compared to the WHOLE file
        # (Using a very high sigma so we don't catch the 'Hit')
        is_extreme = np.abs(sig - np.median(sig)) > (7.0 * np.std(sig))
        cross_axis_spike |= is_extreme
    
    # Dilation: Only kill a very small window (e.g., 5 samples) around these digital spikes
    # This preserves the high-frequency "Hit" right next to them
    bad_zones = binary_dilation(cross_axis_spike, structure=np.ones(5))
    global_mask &= ~bad_zones

    # --- STAGE 2: LOCAL PASS (Temporal MAD) ---
    final_mask = global_mask.copy()
    for sig in sigs_raw:
        local_med = median_filter(sig, size=window_size)
        abs_diff = np.abs(sig - local_med)
        local_mad = median_filter(abs_diff, size=window_size)
        
        # Robust local Z-score
        local_mask = (0.6745 * abs_diff / (local_mad + 1e-4)) < local_sigma
        final_mask &= local_mask

    # --- 3. RECONSTRUCTION ---
    t_masked = t_raw[final_mask]
    
    if len(t_masked) < 2:
        print(f"Warning: {filename} is too corrupted to recover.")
        return None

    t_regular = np.linspace(t_masked.min(), t_masked.max(), len(t_masked))
    clean_signals = []
    
    for i in range(3):
        good_samples = sigs_raw[i][final_mask]

        if interp_type == 'linear':
            f_interp = interp1d(t_masked, good_samples, kind='linear', fill_value="extrapolate")
            clean_signals.append(f_interp(t_regular))
        else:
            cs = CubicSpline(t_masked, good_samples)
            clean_signals.append(cs(t_regular))

    return t_raw, sigs_raw, t_regular, clean_signals, final_mask

# --- Execution ---
# Using 'linear' and a slightly tighter 'local_sigma' for the worst files
file_path = "/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/DATA010_f0_300.bin"
output = process_clean_two_stage(file_path, interp_type='linear', local_sigma=1.2)

if output:
    t_raw, s_raw, t_clean, s_clean, mask = output

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = ['#d62728', '#2ca02c', '#1f77b4']

    for i in range(3):
        # The background shows what was removed by the Cluster Killer and MAD
        axs[i].scatter(t_raw[~mask], s_raw[i][~mask], color='black', alpha=0.1, s=2, label='Cluster/Spike Removed')
        axs[i].plot(t_clean, s_clean[i], color=colors[i], label='Final Linear Bridge', linewidth=1.2)
        
        axs[i].set_ylabel("Accel")
        axs[i].legend(loc='upper right', fontsize='x-small')
        axs[i].grid(True, alpha=0.3)

    plt.xlabel("Time (seconds)")
    plt.suptitle("Final Correction: Simultaneous Axis Spike Detection & Linear Bridging")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()