import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os  # Added to handle file system operations
import re

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

    #print(i)

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
    #print(i)
    return np.array(results)

# def get_fft(t, sig, fs=200):
#     n = len(t)
#     sig_detrended = sig - np.mean(sig)
#     fft_values = np.fft.rfft(sig_detrended)
#     fft_freqs = np.fft.rfftfreq(n, d=1/fs)
#     amplitude = np.abs(fft_values) * (2.0 / n)
#     df = fs/n
#     PSD = (amplitude**2)/(df*2)
#     return fft_freqs, amplitude, PSD

# def get_fft_welch_savgol(data, fs=200):
#     sig_detrended = data - np.mean(data)
#     f, Psd = signal.welch(sig_detrended, fs=fs, nperseg=min(len(sig_detrended), 4096))
#     from scipy.signal import savgol_filter
#     wPsd_smooth = savgol_filter(Psd, window_length=11, polyorder=2)
#     return f, wPsd_smooth

def extract_file_value(filename):
    # Regex breakdown:
    # (\d{2})  -> Group 1: Match exactly two digits
    # _        -> Match the literal underscore
    # (\d{3})  -> Group 2: Match exactly three digits
    # (?=\.)   -> Lookahead: Ensure this is followed by a dot (extension)
    match = re.search(r"(\d{2})_(\d{3})(?=\.)", filename)
    
    if match:
        # Join the two groups with a dot
        float_str = f"{match.group(1)}.{match.group(2)}"
        return float(float_str)
    
    return None

def sin_fit_func(x, a, b, c, d):
    return a*np.sin(b*x + c) + d

# --- AUTOMATED FILE SELECTION ---
path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/spazzata_completa/'

# Check if the FFT folder exists, if not, create it
if not os.path.exists(path + "FFT/"):
    os.makedirs(path + "FFT/")

# This line finds all files ending in .bin that start with 'S0'
names = [f for f in os.listdir(path) if f.endswith('.bin') and f.startswith('S0')]
names.sort() # Ensure we process them in order (S03DATA0000, then 0001, etc.)

print(f"Found {len(names)} acquisition files. Starting processing...")

f = []
Ax = []
Ay = []
i = 0

for current_name in names:
    file_path = os.path.join(path, current_name)
    data = read_synchronized_log(file_path)
    # print(data[0,0])
    # print(data[1,0])
    # print(data[-1,0])
    w = extract_file_value(current_name)*(2*np.pi)
    f.append(w/(2*np.pi))
    #print(w/(2*np.pi))

    if len(data) > 1:
        
        t = data[:-1,0]
        # Divide by sensitivity to get 'g' units
        x, y, z = data[:-1,1]/16384.0, data[:-1,2]/16384.0, data[:-1,3]/16384.0
        t = t - data[-1,0]
        Ax.append(max(x)-min(x))
        Ay.append(max(y)-min(y))
        t = np.where(t<0, t+2**32/1e6, t)

        # fx, XF, PSDX = get_fft(t, x)
        # fy, YF, PSDY = get_fft(t, y)
        # fz, ZF, PSDZ = get_fft(t, z)

        # w_sfx, w_sPSDX = get_fft_welch_savgol(x)
        # w_sfy, w_sPSDY = get_fft_welch_savgol(y)
        # w_sfz, w_sPSDZ = get_fft_welch_savgol(z)

        # --- STIMA PARAMETRI DI BEST FIT ---

        # Calcolo stima del delay temporale tra forzante e accelerazioni
        # l'idea è di trovare il primo punto in cui il segnale attraversa il livello 0


        popt1 = [Ax[i], f[i], 1, np.mean(x)]
        popt2 = [Ay[i], f[i], 1, np.mean(y)]

        # --- SAVING ---
        # save = False  
        # if save:
        #     header = "Freq_x, PSD_x, Freq_y, PSD_y, Freq_z, PSD_z"
        #     output_data = np.column_stack((fx[1:], PSDX[1:], fy[1:], PSDY[1:], fz[1:], PSDZ[1:]))
        #     output_filename = path + "FFT/" + current_name.replace(".bin", "") + "_FFT.txt"
        #     np.savetxt(output_filename, output_data, delimiter=",", header=header, comments='')
        #     print(f"✅ Processed: {current_name}")

        i += 1

        # --- Plotting ---
        # (Keeping your original plotting loop exactly as is)
        # for i in range(2):
        #     fig, ax1 = plt.subplots(figsize=(12, 10))
        #     if i == 0:
        #         sig, color, axis_label = x, 'blue', 'X-Axis'
        #     elif i == 1:
        #         sig, color, axis_label = y, 'green', 'Y-Axis'
            
        #     ax1.scatter(t, sig, color=color, s=0.5, label=f'{axis_label} accelaration')
        #     ax1.plot(t, 0.025*np.sin(w*t) + np.mean(sig), color='red', label=f'Forzante')
        #     ax1.set_xlabel("Time [s]")
        #     ax1.set_ylabel("Acceleration [g]")
        #     ax1.legend()
        #     fig.suptitle(f"File: {current_name} - {axis_label}")
        #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])

fig1, ax = plt.subplots()
ax.scatter(f, Ax, c='green', s=10, label='X')
ax.scatter(f, Ay, c='red', s=1, label='Y')
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude [g]")
ax.set_title("Sweep")
ax.legend(loc='upper right')

plt.show()
            
