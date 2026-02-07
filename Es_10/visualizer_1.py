import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os  # Added to handle file system operations

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

    while i <= len(content) - packet_size:
        header, = struct.unpack_from("<H", content, i)
        if header == sync_word:
            _, x, y, z, timestamp = struct.unpack_from(packet_fmt, content, i)
            results.append([timestamp / 1e6, x, y, z])
            i += packet_size 
        else:
            i += 1 
    return np.array(results)

def get_fft(t, sig, fs=200):
    n = len(t)
    sig_detrended = sig - np.mean(sig)
    fft_values = np.fft.rfft(sig_detrended)
    fft_freqs = np.fft.rfftfreq(n, d=1/fs)
    amplitude = np.abs(fft_values) * (2.0 / n)
    df = fs/n
    PSD = (amplitude**2)/(df*2)
    return fft_freqs, amplitude, PSD

def get_fft_welch_savgol(data, fs=200):
    sig_detrended = data - np.mean(data)
    f, Psd = signal.welch(sig_detrended, fs=fs, nperseg=min(len(sig_detrended), 4096))
    from scipy.signal import savgol_filter
    wPsd_smooth = savgol_filter(Psd, window_length=11, polyorder=2)
    return f, wPsd_smooth

# --- AUTOMATED FILE SELECTION ---
path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/spazzata_0_88/'

# Check if the FFT folder exists, if not, create it
if not os.path.exists(path + "FFT/"):
    os.makedirs(path + "FFT/")

# This line finds all files ending in .bin that start with 'S0'
names = [f for f in os.listdir(path) if f.endswith('.bin') and f.startswith('S0')]
names.sort() # Ensure we process them in order (S03DATA0000, then 0001, etc.)

print(f"Found {len(names)} acquisition files. Starting processing...")

for current_name in names:
    file_path = os.path.join(path, current_name)
    data = read_synchronized_log(file_path)

    if len(data) > 1:
        
        t = data[:-1, 0]
        # Divide by sensitivity to get 'g' units
        x, y, z = data[:-1, 1]/16384.0, data[:-1, 2]/16384.0, data[:-1, 3]/16384.0
        t = t - data[-1, 0]

        fx, XF, PSDX = get_fft(t, x)
        fy, YF, PSDY = get_fft(t, y)
        fz, ZF, PSDZ = get_fft(t, z)

        w_sfx, w_sPSDX = get_fft_welch_savgol(x)
        w_sfy, w_sPSDY = get_fft_welch_savgol(y)
        w_sfz, w_sPSDZ = get_fft_welch_savgol(z)

        # --- SAVING ---
        save = False  
        if save:
            header = "Freq_x, PSD_x, Freq_y, PSD_y, Freq_z, PSD_z"
            output_data = np.column_stack((fx[1:], PSDX[1:], fy[1:], PSDY[1:], fz[1:], PSDZ[1:]))
            output_filename = path + "FFT/" + current_name.replace(".bin", "") + "_FFT.txt"
            np.savetxt(output_filename, output_data, delimiter=",", header=header, comments='')
            print(f"✅ Processed: {current_name}")

        # --- Plotting ---
        # (Keeping your original plotting loop exactly as is)
        for i in range(3):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            if i == 0:
                sig, freq, PSD, msfreq, msPSD, color, axis_label = x, fx, PSDX, w_sfx, w_sPSDX, 'blue', 'X-Axis'
            elif i == 1:
                sig, freq, PSD, msfreq, msPSD, color, axis_label = y, fy, PSDY, w_sfy, w_sPSDY, 'green', 'Y-Axis'
            else:
                sig, freq, PSD, msfreq, msPSD, color, axis_label = z, fz, PSDZ, w_sfz, w_sPSDZ, 'red', 'Z-Axis'
            
            ax1.plot(t, sig, color=color, label=f'{axis_label} Time Domain')
            ax2.plot(freq[1:], PSD[1:], color=color, linewidth=2, label=f'{axis_label} Manual FFT')
            ax2.plot(msfreq[1:], msPSD[1:], color="gray", alpha=0.3, label=f'{axis_label} Welch FFT')
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Acceleration [g]")
            ax2.set_yscale('log')
            ax2.legend()
            fig.suptitle(f"File: {current_name} - {axis_label}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()