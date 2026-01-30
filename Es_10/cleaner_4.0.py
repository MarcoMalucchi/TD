import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.ndimage import median_filter

def process_clean_two_stage(filename, global_sigma=5.0, local_sigma=2.0, window_size=51, interp_type='linear'):
    # --- 1. LOADING & CHRONOLOGICAL SORTING ---
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

    # --- STAGE 1: GLOBAL PASS (Hard Outliers) ---
    global_mask = np.ones(len(t_raw), dtype=bool)
    for sig in sigs_raw:
        hard_limit = np.abs(sig) < 32700
        wide_stat = np.abs(sig - np.median(sig)) < (global_sigma * np.std(sig))
        global_mask &= (hard_limit & wide_stat)

    # --- STAGE 2: LOCAL PASS (Temporal MAD) ---
    final_mask = global_mask.copy()
    for sig in sigs_raw:
        local_med = median_filter(sig, size=window_size)
        abs_diff = np.abs(sig - local_med)
        local_mad = median_filter(abs_diff, size=window_size)
        local_mask = (0.6745 * abs_diff / (local_mad + 1e-4)) < local_sigma
        final_mask &= local_mask

    # --- 3. RECONSTRUCTION SECTION (Now correctly indented) ---
    t_masked = t_raw[final_mask]
    
    if len(t_masked) < 2:
        print(f"Warning: {filename} has no usable data after cleaning.")
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
# Set interp_type='linear' for the bad files, 'cubic' for the good ones
file_path = "/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/DATA010_f0_300.bin"
t_raw, s_raw, t_clean, s_clean, mask = process_clean_two_stage(file_path, interp_type='linear', local_sigma=1.5)

# --- Plotting ---
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
colors = ['#d62728', '#2ca02c', '#1f77b4']

for i in range(3):
    axs[i].scatter(t_raw[~mask], s_raw[i][~mask], color='black', alpha=0.1, s=2, label='Removed Spikes')
    axs[i].plot(t_clean, s_clean[i], color=colors[i], label='Cleaned Signal', linewidth=1)
    axs[i].set_ylabel("Accel")
    axs[i].legend(loc='upper right', fontsize='x-small')
    axs[i].grid(True, alpha=0.3)

plt.xlabel("Time (seconds)")
plt.suptitle("Two-Stage Data Cleaning (Linear Interp for Hardware Glitches)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

save = False  # Set to True to save cleaned data
if save:

    import os

    # --- DEFINE YOUR SAVING PATH HERE ---
    # Example: "/home/marco/Desktop/Cleaned_Data_Tests/"
    save_folder = "/home/marco/Desktop/Uni_anno3/TD/Es_10/cleaned_results/"

    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Extract just the filename (e.g., "DATA010_f0_300.bin") from the full path
    base_name = os.path.basename(file_path)
    save_name = base_name.replace(".bin", "_CLEANED.csv")
    full_save_path = os.path.join(save_folder, save_name)

    # Save the data
    output_data = np.column_stack((t_clean, s_clean[0], s_clean[1], s_clean[2]))
    np.savetxt(full_save_path, output_data, 
            delimiter=",", header="time,acc_x,acc_y,acc_z", comments='')

    print(f"\n✅ ORIGINAL DATA IS SAFE.")
    print(f"✅ CLEANED VERSION SAVED TO: {full_save_path}")