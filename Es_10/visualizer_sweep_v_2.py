import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import re
from scipy.optimize import curve_fit

# ================= STILE GLOBALE =================
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2
})

# ================= FUNZIONI =================

def read_synchronized_log(filename):
    packet_fmt = "<HhhhI"
    packet_size = struct.calcsize(packet_fmt)
    sync_word = 0xAAAA
    results = []

    try:
        with open(filename, "rb") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File {filename} non trovato.")
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


def extract_file_value(filename):
    match = re.search(r"(\d{2})_(\d{3})(?=\.)", filename)
    if match:
        float_str = f"{match.group(1)}.{match.group(2)}"
        return float(float_str)
    return None


def sin_fit_func(x, a, b, c, d):
    return a * np.sin(b * x + c) + d


# ================= PATH =================

plt.close('all')

path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/spazzata_completa_x/'

parent_folder = os.path.basename(path.rstrip("/"))

if not os.path.exists(path + "FFT/"):
    os.makedirs(path + "FFT/")

names = [f for f in os.listdir(path) if f.endswith('.bin') and f.startswith('S0')]
names.sort()

print(f"Trovati {len(names)} file di acquisizione.")

# ================= INIZIALIZZAZIONE =================

f = []
AmplitudeX = []
AmplitudeY = []
s_AmX = []
s_AmY = []
dephaseX = []
dephaseY = []
s_dpX = []
s_dpY = []
forcing_amplitude = []

i = 0

# ================= LOOP FILE =================

for current_name in names:

    file_path = os.path.join(path, current_name)
    data = read_synchronized_log(file_path)

    w = extract_file_value(current_name) * (2 * np.pi)
    f.append(w / (2 * np.pi))

    if len(data) > 1:

        t_raw = data[:-1, 0]
        t_sync = data[-1, 0]
        t = t_raw - t_sync
        t = np.where(t < 0, t + (2**32)/1e6, t)

        x = data[:-1, 1] / 16384.0
        y = data[:-1, 2] / 16384.0

        Ax = (max(x) - min(x)) / 2
        Ay = (max(y) - min(y)) / 2

        rising_indicesX = np.where(np.diff(np.sign(x - np.mean(x))) > 0)[0]
        rising_indicesY = np.where(np.diff(np.sign(y - np.mean(y))) > 0)[0]

        p0X = [Ax, w, 0, np.mean(x)]
        p0Y = [Ay, w, 0, np.mean(y)]

        if rising_indicesX.size > 0:
            delayX = t[rising_indicesX[0]]
            p0X = [Ax, w, -w * delayX, np.mean(x)]

        if rising_indicesY.size > 0:
            delayY = t[rising_indicesY[0]]
            p0Y = [Ay, w, -w * delayY, np.mean(y)]

        ppX, pcovX = curve_fit(sin_fit_func, t, x, p0=p0X,
                               sigma=np.ones(len(x))*1.46e-3,
                               absolute_sigma=True)

        ppY, pcovY = curve_fit(sin_fit_func, t, y, p0=p0Y,
                               sigma=np.ones(len(y))*1.41e-3,
                               absolute_sigma=True)

        if ppX[0] < 0:
            ppX[0] *= -1
            ppX[2] += np.pi

        if ppY[0] < 0:
            ppY[0] *= -1
            ppY[2] += np.pi

        AmplitudeX.append(ppX[0])
        AmplitudeY.append(ppY[0])
        s_AmX.append(np.sqrt(np.diag(pcovX))[0])
        s_AmY.append(np.sqrt(np.diag(pcovY))[0])
        dephaseX.append(ppX[2])
        dephaseY.append(ppY[2])
        s_dpX.append(np.sqrt(np.diag(pcovX))[2])
        s_dpY.append(np.sqrt(np.diag(pcovY))[2])

        forcing_amplitude.append(37.39e-3 * (w)**2)

        i += 1

# ================= PREPARAZIONE DATI =================

f = np.array(f)
AmplitudeX = np.array(AmplitudeX)
AmplitudeY = np.array(AmplitudeY)
s_AmX = np.array(s_AmX)
s_AmY = np.array(s_AmY)
forcing_amplitude = np.array(forcing_amplitude)

dephaseX = np.array(dephaseX)
dephaseY = np.array(dephaseY)
s_dpX = np.array(s_dpX)
s_dpY = np.array(s_dpY)

sort_idx = np.argsort(f)
f_sorted = f[sort_idx]

dpX_norm = (dephaseX[sort_idx] + np.pi) % (2*np.pi) - np.pi
dpY_norm = (dephaseY[sort_idx] + np.pi) % (2*np.pi) - np.pi

dephaseX_unwrapped = np.unwrap(dpX_norm)
dephaseY_unwrapped = np.unwrap(dpY_norm)

s_dpX_sorted = s_dpX[sort_idx]
s_dpY_sorted = s_dpY[sort_idx]

# ================= PLOT FASE =================

fig1, ax1 = plt.subplots(figsize=(12, 8))

if "spazzata_completa_y" in parent_folder:
    ax1.errorbar(f_sorted, dephaseY_unwrapped,
                 yerr=s_dpY_sorted,
                 fmt='o', markersize=4,
                 color='black', elinewidth=1.2,
                 alpha=0.8,
                 label='Sfasamento asse Y')
    fase_plot = dephaseY_unwrapped
    asse_label = "Y"

else:
    ax1.errorbar(f_sorted, dephaseX_unwrapped,
                 yerr=s_dpX_sorted,
                 fmt='o', markersize=4,
                 color='black', elinewidth=1.2,
                 alpha=0.8,
                 label='Sfasamento asse X')
    fase_plot = dephaseX_unwrapped
    asse_label = "X"

ax1.set_xlabel("Frequenza [Hz]")
ax1.set_ylabel("Sfasamento [rad]")
ax1.set_title(f"{parent_folder} — Sfasamento asse {asse_label}",
              fontweight='bold')

ax1.grid(True, which='both', linewidth=0.8, alpha=0.3)
ax1.legend()
plt.tight_layout()
plt.show()

# ================= PLOT MODULO FRF =================

fig2, ax2 = plt.subplots(figsize=(12, 8))

Hx = AmplitudeX / forcing_amplitude
Hy = AmplitudeY / forcing_amplitude

sHx = s_AmX / forcing_amplitude
sHy = s_AmY / forcing_amplitude

if "spazzata_completa_y" in parent_folder:
    ax2.errorbar(f, Hy, yerr=sHy,
                 fmt='o', markersize=4,
                 elinewidth=1.2, color='black',
                 alpha=0.8,
                 label='Guadagno FRF asse Y')
    asse_label = "Y"

else:
    ax2.errorbar(f, Hx, yerr=sHx,
                 fmt='o', markersize=4,
                 elinewidth=1.2, color='black',
                 alpha=0.8,
                 label='Guadagno FRF asse X')
    asse_label = "X"

ax2.set_yscale('symlog', linthresh=1e-4)
ax2.set_xlabel("Frequenza [Hz]")
ax2.set_ylabel("Guadagno FRF [a.u.]")
ax2.set_title(f"{parent_folder} — Guadagno FRF asse {asse_label}",
              fontweight='bold')

ax2.grid(True, which='both', linewidth=0.8, alpha=0.3)
ax2.legend()
plt.tight_layout()
plt.show()

# ================= SALVATAGGIO OPZIONALE =================

save_fig = False

if save_fig:
    output_folder = os.path.join(path, "FRF")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fig1.savefig(os.path.join(output_folder,
                 f"{parent_folder}_fase.png"), dpi=300)
    fig2.savefig(os.path.join(output_folder,
                 f"{parent_folder}_modulo.png"), dpi=300)

    print("Figure salvate correttamente.")
