import struct
import numpy as np
import matplotlib.pyplot as plt

def read_synchronized_log(filename):
    # We unpack the whole 12-byte packet as Little-Endian first
    # H = Header (2 bytes)
    # h, h, h = Accel X, Y, Z (2 bytes each)
    # I = Timestamp (4 bytes)
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

    i = 0
    while i <= len(content) - packet_size:
        # Check current 2 bytes for 0xAAAA
        current_header, = struct.unpack_from("<H", content, i)
        
        if current_header == sync_word:
            # Unpack the packet
            # Note: x, y, z will be "swapped" because they are Big-Endian in the file
            _, x_raw, y_raw, z_raw, timestamp = struct.unpack_from(packet_fmt, content, i)
            
            # MANUALLY SWAP BYTES for the Big-Endian Acceleration data
            # This converts the Little-Endian read into the correct Big-Endian value
            x = struct.unpack(">h", struct.pack("<h", x_raw))[0]
            y = struct.unpack(">h", struct.pack("<h", y_raw))[0]
            z = struct.unpack(">h", struct.pack("<h", z_raw))[0]
            
            results.append([timestamp / 1e6, x, y, z])
            i += packet_size 
        else:
            i += 1 

    return np.array(results)

# --- Execution and Plotting ---
file_path = "/home/marco/Desktop/Uni_anno3/TD/Es_10/TestBusI2C/DATA000_f1_780.bin"
data = read_synchronized_log(file_path)

if len(data) > 0:
    t, x, y, z = data[:-1, 0], data[:-1, 1], data[:-1, 2], data[:-1, 3]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = ['crimson', 'seagreen', 'royalblue']
    labels = ['X-Axis', 'Y-Axis', 'Z-Axis']
    
    for i, (ax, signal, color, label) in enumerate(zip(axs, [x, y, z], colors, labels)):
        ax.scatter(t, signal, marker='.', s=0.5, color=color, label=label, linewidth=0.8)
        ax.set_ylabel("Raw Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axs[2].set_xlabel("Time (seconds)")
    plt.suptitle("MPU-6050 Synchronized Data Acquisition (Fixed Endianness)")
    plt.tight_layout()
    plt.show()
else:
    print("No valid packets found with Sync Word 0xAAAA. Check if your Arduino is actually writing the header.")