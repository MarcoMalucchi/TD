import sys
from pathlib import Path
import time
import math

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tdwf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.signal import periodogram

from utils.labplot import save_lab_figure, float_to_str


plt.close('all')


# ==========================================================================
#  FFT/PSD robusta (Hann + detrend + density scaling)
#  Restituisce: freq [Hz], ASD [V/sqrt(Hz)]
# ==========================================================================
def get_asd(signal, fs):
    sig = signal - np.mean(signal)                 # detrend (remove DC)
    f, psd = periodogram(
        sig,
        fs=fs,
        window="hann",                             # finestra Hann
        scaling="density",                         # PSD in V^2/Hz
        detrend=False,
        return_onesided=True
    )
    asd = np.sqrt(psd)                             # ASD in V/sqrt(Hz)
    return f, asd


# ==========================================================================
#  Parametri dello script (sweep in frequenza)
# ==========================================================================
nf = 500
f0 = 1e3
f1 = 1e4
fv = np.logspace(np.log10(f0), np.log10(f1), nf)

A_DRIVE = 0.6          # ampiezza forzante (fissa in questo sweep)
FS = 1e6               # sampling rate
NPT = 16384            # punti acquisiti (fisso per avere griglia FFT costante)
SETTLE_S = 0.03        # attesa dopo cambio frequenza (puoi portarla a 0.05–0.2 se serve)


# ==========================================================================
#  Configurazione AD2
# ==========================================================================
ad2 = tdwf.AD2()
ad2.vdd = 5
ad2.vss = -5
ad2.power(True)

wavegen = tdwf.WaveGen(ad2.hdwf)
wavegen.w1.ampl = A_DRIVE
wavegen.w1.func = tdwf.funcSine
wavegen.w1.start()

scope = tdwf.Scope(ad2.hdwf)
scope.fs = FS
scope.npt = NPT
scope.ch1.rng = 5
scope.ch2.rng = 50
scope.ch1.avg = True
scope.ch2.avg = True


# ==========================================================================
#  Ciclo misura
#  data[ii, :] = ASD(f) per la frequenza di forzante fv[ii]
# ==========================================================================
scope.sample()
freq_fft, asd0 = get_asd(scope.ch1.vals, scope.fs)

# Limita l'interesse spettrale (es. fino a 20 kHz): evita di “sporcare” il plot
FMAX_PLOT = 2.0e4
mask = (freq_fft >= 10.0) & (freq_fft <= FMAX_PLOT)    # evita anche 0 Hz (log)
freq_fft = freq_fft[mask]
nfft = len(freq_fft)

data = np.zeros((len(fv), nfft), dtype=float)

for ii, ff in enumerate(tqdm(fv, desc="Sweep in frequenza")):
    wavegen.w1.freq = float(ff)
    time.sleep(SETTLE_S)        # lascia assestare un minimo il transitorio
    scope.sample()

    f, asd = get_asd(scope.ch1.vals, scope.fs)
    data[ii, :] = asd[mask]


# ==========================================================================
#  Visualizzazione “tipo giallo”: dB + jet + percentili
# ==========================================================================
plt.rcParams.update({
    "font.size": 18,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

eps = 1e-20
data_db = 20.0 * np.log10(data + eps)   # dB re 1 V/sqrt(Hz)

# Range colore robusto: evita saturazione dovuta a pochi picchi
vmin = np.percentile(data_db, 5)
vmax = np.percentile(data_db, 99.5)

fig, ax = plt.subplots(figsize=(11, 7))

# Assi corretti: x = driving frequency, y = spectral frequency
pcm = ax.pcolormesh(
    fv, freq_fft, data_db.T,
    shading="auto",
    cmap="jet",
    vmin=vmin, vmax=vmax
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Driving frequency [Hz]", fontweight="bold")
ax.set_ylabel("Spectral frequency [Hz]", fontweight="bold")

ax.set_xlim(f0, f1)
ax.set_ylim(freq_fft[0], freq_fft[-1])

# Tick/spine più visibili
ax.tick_params(axis="both", which="both", direction="in", length=8, width=2)
for sp in ax.spines.values():
    sp.set_linewidth(2)
for lab in ax.get_xticklabels() + ax.get_yticklabels():
    lab.set_fontweight("bold")

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label("ASD [dB re 1 V/√Hz]", fontweight="bold")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

ax.set_title(f"Color plot della FFT di Vc (A = {A_DRIVE} V)", fontweight="bold")

plt.tight_layout()
plt.show()

save_lab_figure(fig, ax, f"task6_colorplot_FFT_{float_to_str(A_DRIVE, 3)}")
print("immagine salvata")

ad2.close()