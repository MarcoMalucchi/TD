import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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

def get_fft(t, signal, fs=200):
    n = len(t)
    sig_detrended = signal - np.mean(signal) # Remove DC offset
    fft_values = np.fft.rfft(sig_detrended)
    fft_freqs = np.fft.rfftfreq(n, d=1/fs)
    amplitude = np.abs(fft_values) * (2.0 / n)
    df = fs/n
    PSD = (amplitude**2)/(df*2)
    return fft_freqs, amplitude, PSD

def get_fft_welch_savgol(data, fs=200):
    sig_detrended = data - np.mean(data) # Remove DC offset
    f, Psd = signal.welch(sig_detrended, fs=fs, nperseg=min(len(sig_detrended), 4096))

    # If it's still too noisy for a fit, you can smooth the PSD itself:
    from scipy.signal import savgol_filter
    wPsd_smooth = savgol_filter(Psd, window_length=11, polyorder=2)
    return f, wPsd_smooth

# --- Execution ---

path='/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/parte_bassa_ruotato/'
files = 'Elencofile.txt'

file_names = open(path+files, 'r')

names = np.array([])

while True:
    Line = file_names.readline()
    if Line == '': break
    names = np.append(names, Line.strip('\n'))

file_names.close()

for j in range(names.size):

    file_path = path + names[j]
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

        fx, XF, PSDX = get_fft(t, x)
        fy, YF, PSDY = get_fft(t, y)
        fz, ZF, PSDZ = get_fft(t, z)

        w_sfx, w_sPSDX = get_fft_welch_savgol(x)
        w_sfy, w_sPSDY = get_fft_welch_savgol(y)
        w_sfz, w_sPSDZ = get_fft_welch_savgol(z)

        # --- SAVING ---
        save = False  # Set to True to enable saving
        if save:
            header = "Freq_x, PSD_x, Freq_y, PSD_y, Freq_z, PSD_z"
            output_data = np.column_stack((fx[1:], PSDX[1:], fy[1:], PSDY[1:], fz[1:], PSDZ[1:]))
            output_filenames = path + "FFT/" + names[j].replace(".bin", "") + "_FFT.txt"
            np.savetxt(output_filenames, output_data, delimiter=",", header=header, comments='')
            print(f"FFT of: {names[j]} saved to: {output_filenames}")

        # --- Plotting ---
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
            ax1.set_title(f"{axis_label} - Time Domain")
            ax2.set_xlabel("Frequency [Hz]")
            ax2.set_ylabel(r"PSD [$\frac{g^2}{Hz}$]")
            ax2.set_yscale('log')
            ax2.legend()
            ax2.set_title(f"{axis_label} - Frequency Domain (FFT)")
            
            fig.suptitle(f"Acceleration Data and FFT for {axis_label} - File: {names[j]}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()