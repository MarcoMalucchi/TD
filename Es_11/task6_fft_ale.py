import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import tdwf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import numpy as np
import math
from tqdm import tqdm
from utils.labplot import save_lab_figure, float_to_str

#path = "/home/marco/Desktop/Uni_anno3/TD/Es_11/acquisizioni/"

plt.close('all')

def get_fft(t, signal, fs=200):
    n = len(t)
    sig_detrended = signal - np.mean(signal) # Remove DC offset
    fft_values = np.fft.rfft(sig_detrended)
    fft_freqs = np.fft.rfftfreq(n, d=1/fs)
    amplitude = np.abs(fft_values) * (2.0 / n)
    df = fs/n
    PSD = (amplitude**2)/(df*2)
    return fft_freqs, amplitude, PSD

def osc_sampling_rate(r):
    return 100e6 / math.ceil(100e6 / r)
    
# ==========================================================================
#  Parametri dello script
#nper = 10           # numero periodi usati per la stima   
#npt = 16384          # numero MASSIMO di punti acquisiti
nf = 500            # numero di frequenze nello sweep da f0 a f1   
f0 = 1e3
f1 = 1e4
# vettore delle frequenze
fv = np.logspace(np.log10(f0), np.log10(f1), nf)

# ==========================================================================
#  Configurazione base AD2 (più parametri impostati dopo)
ad2 = tdwf.AD2()
ad2.vdd = 5
ad2.vss = -5
ad2.power(True)

wavegen = tdwf.WaveGen(ad2.hdwf)
wavegen.w1.ampl = 0.6
wavegen.w1.func = tdwf.funcSine
wavegen.w1.start()
scope = tdwf.Scope(ad2.hdwf)
#scope.fs = 1e6
scope.fs = 1e6
scope.npt = 16384
scope.ch1.rng = 5
scope.ch2.rng = 50 
scope.ch1.avg = True
scope.ch2.avg = True 

# ==========================================================================
# Ciclo misura

nfft = scope.npt // 2 + 1
data = np.zeros((len(fv), nfft))

scope.sample()

freq, _, _ = get_fft(scope.time.vals, scope.ch1.vals, scope.fs)

for ii in range(len(fv)):  # Ciclo frequenze
    # if ii % 25 == 0:
    #     print(f"{round((ii/len(fv))*100)}%")
    # Frequenza attuale
    ff = fv[ii]
    # [3b] stima parametri di sampling
    #
    #  COSA VOGLIO: misurare nper con al massimo npt punti acquisizione
    #  DOMANDA: quale è la MASSIMA frequenza di sampling che posso usare?
    #
    #  NOTARE: solo 100MSa/s intero (qui df) è una frequenza ammessa.
    #
    #  SE voglio misurare nper periodi a ff devo misura per un tempo TT = nper/ff
    #  SE misuro ad un rate fs, mi servono npt = fs*nper/ff punti di acquisizione
    #
    #  voglio che fs*nper/ff sia il più alto possibile ma al massimo uguale a npt 
    #  (altrimenti no buco il buffer...), per ottenere questo sceglo un df intero 
    #  in modo che fs = 100MHz/df soddisfi la relazione sopra
    #
    #  => df = celing(100MHz*nper/(npt*ff)) 
    #

    #  Ribadiamo il trigger... 
    #scope.trig(True, hist = 0.01)
    wavegen.w1.freq = ff 
    # [3c] Campionamento e analisi risultati
    scope.sample()

    _, _, PSD = get_fft(scope.time.vals, scope.ch1.vals, scope.fs)
    data[ii, :] = 10 * np.log10(PSD)

# [4] Visualizzazione risultati

import matplotlib.colors as colors

plt.rcParams.update({
    "font.size": 18,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

fig, ax = plt.subplots(figsize=(10, 6))

# === ASSI CORRETTI: x = fv (frequenza di forzante, parametro scansionato), y = freq (frequenza spettrale) ===
pcm = ax.pcolormesh(fv, freq, data.T, shading='auto', cmap='inferno')

# entrambe sono frequenze -> entrambe in log ha senso
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel("Driving frequency [Hz]", fontweight="bold")
ax.set_ylabel("Spectral frequency [Hz]", fontweight="bold")

# limiti coerenti con quelli che avevi (ma ora applicati all'asse corretto)
ax.set_xlim(f0, f1)
ax.set_ylim(f0, 5*1e5)  # oppure freq.min(), freq.max() se preferisci

# tick più visibili e in grassetto
ax.tick_params(axis='both', which='both', direction='in', length=8, width=2)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label("[dB]", fontweight="bold")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')

ax.set_title(f"Color plot della FFT di Vc (A = {wavegen.w1.ampl} V)", fontweight="bold")

plt.show()

save_lab_figure(fig, ax, f"task6_colorplot_FFT_{float_to_str(wavegen.w1.ampl, 3)}")