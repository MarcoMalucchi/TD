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
from utils.labplot import save_lab_figure, float_to_str

plt.close('all')

# ==========================================================================
#  FFT + PSD con finestra Hann (come nel codice precedente)
# ==========================================================================
def get_fft_psd(signal, fs):
    n = len(signal)

    sig = signal - np.mean(signal)

    w = np.hanning(n)
    sigw = sig * w

    fft_values = np.fft.rfft(sigw)
    fft_freqs = np.fft.rfftfreq(n, d=1/fs)

    cg = np.mean(w)
    amplitude = (2.0 / n) * np.abs(fft_values) / (cg + 1e-30)

    df = fs / n
    PSD = (amplitude**2) / (2.0 * df)

    return fft_freqs, PSD

# ==========================================================================
#  Parametri sweep in ampiezza
# ==========================================================================
nA = 500
A0 = 0.1
A1 = 1.5
Av = np.linspace(A0, A1, nA)

F_DRIVE = 7300       # frequenza fissa
FS = 1e6
NPT = 16384
SETTLE_S = 0.03

FMIN_PLOT = 80.0
FMAX_PLOT = 2.0e4

# ==========================================================================
#  Configurazione AD2
# ==========================================================================
ad2 = tdwf.AD2()
ad2.vdd = 5
ad2.vss = -5
ad2.power(True)

wavegen = tdwf.WaveGen(ad2.hdwf)
wavegen.w1.freq = F_DRIVE
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
# ==========================================================================
scope.sample()
freq, PSD0 = get_fft_psd(scope.ch1.vals, scope.fs)

mask = (freq >= FMIN_PLOT) & (freq <= FMAX_PLOT)
freq_plot = freq[mask]
nfft = len(freq_plot)

data_db = np.zeros((len(Av), nfft), dtype=float)

eps = 1e-24

for ii, A in enumerate(tqdm(Av, desc="Sweep in ampiezza")):
    wavegen.w1.ampl = float(A)
    time.sleep(SETTLE_S)
    scope.sample()

    _, PSD = get_fft_psd(scope.ch1.vals, scope.fs)

    ASD = np.sqrt(PSD[mask])
    data_db[ii, :] = 20.0 * np.log10(ASD + eps)

# ==========================================================================
#  Visualizzazione (identica logica del codice precedente)
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

vmin = np.percentile(data_db, 5)
vmax = np.percentile(data_db, 99.5)

fig, ax = plt.subplots(figsize=(11, 7))

pcm = ax.pcolormesh(
    Av,
    freq_plot,
    data_db.T,
    shading="auto",
    cmap="jet",
    vmin=vmin,
    vmax=vmax
)

ax.set_yscale("log")

ax.set_xlabel("Driving amplitude [V]", fontweight="bold")
ax.set_ylabel("Spectral frequency [Hz]", fontweight="bold")

ax.set_xlim(A0, A1)
ax.set_ylim(freq_plot[0], freq_plot[-1])

ax.tick_params(axis="both", which="major",
               direction="out", length=10, width=2)
ax.tick_params(axis="both", which="minor",
               direction="out", length=6, width=1.5)

for sp in ax.spines.values():
    sp.set_linewidth(2)
for lab in ax.get_xticklabels() + ax.get_yticklabels():
    lab.set_fontweight("bold")

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label("ASD [dB]", fontweight="bold")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

ax.set_title(f"Color plot della FFT di Vc (f = {F_DRIVE} Hz)",
             fontweight="bold")

plt.tight_layout()
plt.show()

save_lab_figure(fig, ax,
                f"task6_colorplot_ampl_{float_to_str(F_DRIVE, 3)}")

print("immagine salvata")

ad2.close()