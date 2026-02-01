import struct
import numpy as np
import matplotlib.pyplot as plt

def read_synchronized_log(filename):
    packet_fmt = "<HhhhI" 
    packet_size = struct.calcsize(packet_fmt)
    sync_word = 0xAAAA
    results = []

    try:
        with open(filename, "rb") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return np.array([])

    # --- SYNC-LOCK SCANNER ---
    i = 0
    start_found = False
    while i <= len(content) - (packet_size * 2):
        header, = struct.unpack_from("<H", content, i)
        if header == sync_word:
            next_header, = struct.unpack_from("<H", content, i + packet_size)
            if next_header == sync_word:
                start_found = True
                break
        i += 1 
    
    if not start_found:
        return np.array([])

    # --- DATA EXTRACTION ---
    while i <= len(content) - packet_size:
        header, = struct.unpack_from("<H", content, i)
        if header == sync_word:
            _, x, y, z, timestamp = struct.unpack_from(packet_fmt, content, i)
            results.append([timestamp / 1e6, x, y, z])
            i += packet_size 
        else:
            i += 1 

    return np.array(results)

# --- Execution ---
file_path = "/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/test/S04DATA0000_Hz00_000.bin" 
data = read_synchronized_log(file_path)

if len(data) > 1:
    # 1. Sort chronologically
    data = data[data[:, 0].argsort()]
    
    # 2. DISREGARD THE GHOST POINT & NORMALIZE TIME
    # Slice from index 1 to ignore the power-on spike
    t = data[1:, 0]
    x, y, z = data[1:, 1]/16384.0, data[1:, 2]/16384.0, data[1:, 3]/16384.0
    
    # Shift time to start at 0.0 for the actual burst
    t = t - t[0]

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, sig, lbl, col in zip(axs, [x, y, z], ['X', 'Y', 'Z'], ['crimson', 'seagreen', 'royalblue']):
        ax.plot(t, sig, color=col, label=f"{lbl}-Axis")
        ax.set_ylabel("Raw LSB")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.xlabel("Time (seconds)")
    plt.suptitle("Final Cleaned Acquisition: Burst Start at T=0")
    plt.tight_layout()
    plt.show()