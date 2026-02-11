import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os  # Added to handle file system operations
import re
from scipy.optimize import curve_fit

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

plt.close('all')

path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/spazzata_completa_x/'

# Check if the FFT folder exists, if not, create it
if not os.path.exists(path + "FFT/"):
    os.makedirs(path + "FFT/")

# This line finds all files ending in .bin that start with 'S0'
names = [f for f in os.listdir(path) if f.endswith('.bin') and f.startswith('S0')]
names.sort() # Ensure we process them in order (S03DATA0000, then 0001, etc.)

print(f"Found {len(names)} acquisition files. Starting processing...")

f = []
Ax = []
AmplitudeX = []
s_AmX = []
AmplitudeY = []
s_AmY = []
dephaseX = []
s_dpX = []
dephaseY = []
s_dpY = []
Ay = []
forcing_amplitude = []
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
        
        # 1. Get raw timestamps
        t_raw = data[:-1, 0]
        t_sync = data[-1, 0]

        # 2. Handle the 32-bit wrapping/two's complement on the raw values
        # (Do this to both the array and the reference point)
        # t_raw = np.where(t_raw < 0, t_raw + (2**32)/1e6, t_raw)
        # t_sync = np.where(t_sync < 0, t_sync + (2**32)/1e6, t_sync)

        # 3. Now shift. Values before the sync pulse will stay negative.
        # Values at the sync pulse will be 0.
        t = t_raw - t_sync
        t = np.where(t < 0, t + (2**32)/1e6, t)

        # Divide by sensitivity to get 'g' units
        x, y, z = data[:-1,1]/16384.0, data[:-1,2]/16384.0, data[:-1,3]/16384.0
        Ax.append((max(x)-min(x))/2)
        Ay.append((max(y)-min(y))/2)

        # --- STIMA PARAMETRI DI BEST FIT ---

        # Calcolo stima del delay temporale tra forzante e accelerazioni
        # l'idea è di trovare il primo punto in cui il segnale attraversa il livello 0

        rising_indicesX = np.where(np.diff(np.sign(x-np.mean(x)))>0)[0]
        rising_indicesY = np.where(np.diff(np.sign(y-np.mean(y)))>0)[0]

        if rising_indicesX.size > 0:
            first_rising_indexX = rising_indicesX[0]

            delayX = t[first_rising_indexX]

            p0X = [Ax[i], w, -w*delayX, np.mean(x)]
            
        else:
            print(f"WARNING: No rising edge found for {current_name} in X axis")

            p0X = [Ax[i], f[i], 0, np.mean(x)]
        
        if rising_indicesY.size > 0:
            first_rising_indexY = rising_indicesY[0]

            delayY = t[first_rising_indexY]

            p0Y = [Ay[i], w, -w*delayY, np.mean(y)]
            
        else:
            print(f"WARNING: No rising edge found for {current_name} in Y axis")

            p0Y = [Ay[i], f[i], 0, np.mean(y)]

        ppX, pcovX = curve_fit(sin_fit_func, t, x, p0=p0X, sigma=np.ones(len(x))*1.46e-3, absolute_sigma=True)
        ppY, pcovY = curve_fit(sin_fit_func, t, y, p0=p0Y, sigma=np.ones(len(y))*1.41e-3, absolute_sigma=True)

        # Aggiustamento parametri se ampiezzamisurata negativa
        if ppX[0] < 0: 
            ppX[0] *= -1
            ppX[2] += np.pi    
        if ppY[0] < 0:
            ppY[0] *= -1
            ppY[2] += np.pi

        # Aggiornamento dei dati
        AmplitudeX.append(ppX[0])
        AmplitudeY.append(ppY[0])
        s_AmX.append(np.sqrt(np.diag(pcovX))[0])
        s_AmY.append(np.sqrt(np.diag(pcovY))[0])
        
        #Adjusting the dephasing values so that they are between -pi and pi

        ppX[2] = (ppX[2] + np.pi) % (2*np.pi) - np.pi
        ppY[2] = (ppY[2] + np.pi) % (2*np.pi) - np.pi

        # fig3, ax = plt.subplots(figsize=(12, 10))
        # ax.plot(t, sin_fit_func(t, ppX[0], ppX[1], ppX[2], ppX[3]), c='blue', label='fit model')
        # ax.scatter(t, x, c='green', s=1, label='X')
        # # ax.plot(t, sin_fit_func(t, ppY[0], ppY[1], ppY[2], ppY[3]), c='black', label='Y')
        # # ax.scatter(t, y, c='red', s=1, label='Y')
        # ax.plot(t, ppX[0]*np.sin(ppX[1]*t) + ppX[3], c='red', label='driving force')
        # ax.set_xlabel("Time [s]")
        # ax.set_ylabel("Acceleration [g]")
        # ax.set_title("Fit sinusoidale di prova")
        # ax.legend(loc='upper right')
        # plt.show()

        dephaseX.append(ppX[2])
        dephaseY.append(ppY[2])
        s_dpX.append(np.sqrt(np.diag(pcovX))[2])
        s_dpY.append(np.sqrt(np.diag(pcovY))[2])

        forcing_amplitude.append(37.39e-3*(w)**2) # 37.39e-3 distanza dall'asse di rotazione del motore del baricentro della massa fuori asse
        # la forzante è data dall'accelerazione centrifuga agente sulla massa nell'SR solidale al disco di supporto

        # --- SAVING ---
        # save = False  
        # if save:
        #     header = "Freq_x, PSD_x, Freq_y, PSD_y, Freq_z, PSD_z"
        #     output_data = np.column_stack((fx[1:], PSDX[1:], fy[1:], PSDY[1:], fz[1:], PSDZ[1:]))
        #     output_filename = path + "FFT/" + current_name.replace(".bin", "") + "_FFT.txt"
        #     np.savetxt(output_filename, output_data, delimiter=",", header=header, comments='')
        #     print(f"✅ Processed: {current_name}")

        i += 1

fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(f, Ax, c='green', s=1, label='X')
ax.scatter(f, Ay, c='red', s=1, label='Y')
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude [g]")
ax.set_title("Approsimative gain plot")
ax.legend(loc='upper right')

# Convert lists to numpy arrays
dephaseX = np.array(dephaseX)
dephaseY = np.array(dephaseY)

fig1, ax1 = plt.subplots(figsize=(12, 10))
ax1.errorbar(f, dephaseX, yerr=np.array(s_dpX), fmt='o', markersize=2, c='green', ecolor='lightgreen', label='X')
#ax1.errorbar(f, dephaseY, yerr=np.array(s_dpY), fmt='o', markersize=2, c='red', ecolor='lightcoral', label='Y')
ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("Dephase [rad]")
ax1.set_title("Dephasing")
ax1.legend(loc='upper right')

fig2, ax2 = plt.subplots(figsize=(12, 10))
ax2.errorbar(f, np.array(AmplitudeX)/np.array(forcing_amplitude), yerr=np.array(s_AmX)/np.array(forcing_amplitude), fmt='o', markersize=2, c='green', ecolor='lightgreen', label='X')
#ax2.errorbar(f, np.array(AmplitudeY)/np.array(forcing_amplitude), yerr=np.array(s_AmX)/np.array(forcing_amplitude), fmt='o', markersize=2, c='red', ecolor='lightcoral', label='Y')
ax2.set_yscale('symlog', linthresh=1e-4)
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Amplitude [a.u.]")
ax2.set_title("Amplitude")
ax2.legend(loc='upper right')

plt.show()

# ================= SAVE SWEEP FRF DATA =================

save = True

if save:
    output_folder = os.path.join(path, "FRF")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    f_arr = np.array(f)
    Hx = np.array(AmplitudeX) / np.array(forcing_amplitude)
    Hy = np.array(AmplitudeY) / np.array(forcing_amplitude)

    sHx = np.array(s_AmX) / np.array(forcing_amplitude)
    sHy = np.array(s_AmY) / np.array(forcing_amplitude)

    header = "Frequency[Hz], Hx, sHx, Hy, sHy"

    np.savetxt(
        os.path.join(output_folder, "SWEEP_FRF.csv"),
        np.column_stack((f_arr, Hx, sHx, Hy, sHy)),
        delimiter=",",
        header=header,
        comments=""
    )

    print("✅ FRF sweep data saved.")




