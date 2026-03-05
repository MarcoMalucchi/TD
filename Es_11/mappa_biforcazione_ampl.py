import sys
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks

# Assicurati che il percorso per tdwf sia corretto per il tuo ambiente
sys.path.append(str(Path(__file__).resolve().parents[1]))
import tdwf

# ==========================================================================
#  PARAMETRI SWEEP IN AMPIEZZA
# ==========================================================================
nA = 600             # Numero di punti sull'asse X (risoluzione dello sweep)
A0 = 0.1             # Ampiezza di partenza [V]
A1 = 3.0             # Ampiezza finale [V] (regola in base al limite del circuito)
Av = np.linspace(A0, A1, nA)

F_DRIVE = 7300       # Frequenza fissa di eccitazione [Hz]
FS = 1e6             # Sample Rate 1MHz
NPT = 16384          # Punti per acquisizione (abbastanza per molti cicli)
SETTLE_S = 0.05      # Tempo di attesa per esaurire i transitori (fondamentale!)

# ==========================================================================
#  CONFIGURAZIONE HARDWARE AD2
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
scope.ch1.rng = 10    # Range alto (10V) per evitare clipping se il segnale cresce
scope.ch1.avg = False # Disabilitato: vogliamo i valori istantanei, non la media

# Liste per accumulare i dati per lo scatter plot
all_amplitudes = []
all_peaks = []

# ==========================================================================
#  CICLO DI MISURA (BIFURCATION MAP)
# ==========================================================================
try:
    print(f"Inizio sweep in ampiezza a {F_DRIVE} Hz...")
    
    for A in tqdm(Av, desc="Generazione Mappa"):
        # 1. Imposta la nuova ampiezza dello stimolo
        wavegen.w1.ampl = float(A)
        
        # 2. Aspetta che il sistema fisico si stabilizzi sul nuovo stato
        time.sleep(SETTLE_S)
        
        # 3. Acquisisci il segnale
        scope.sample()
        signal = scope.ch1.vals
        
        # 4. Detrend per una peak detection più pulita (centra lo zero)
        sig_centered = signal - np.mean(signal)
        
        # 5. Calcolo distanza minima tra i picchi (1.5 volte la freq. di drive)
        # Questo serve a non prendere i piccoli picchi del rumore
        dist_min = max(1, int(FS / (F_DRIVE * 1.5)))
        
        # 6. Trova i massimi locali (picchi)
        # La prominence aiuta a distinguere i veri picchi dal rumore di fondo
        peaks_idx, _ = find_peaks(sig_centered, distance=dist_min, prominence=0.02)
        
        if len(peaks_idx) > 0:
            peaks_vals = signal[peaks_idx] # Usiamo i valori originali (con offset)
            
            # 7. Teniamo solo gli ultimi picchi dell'acquisizione 
            # (per essere sicuri di aver superato ogni transitorio residuo)
            n_to_save = 32
            selected_peaks = peaks_vals[-n_to_save:]
            
            for p in selected_peaks:
                all_amplitudes.append(A)
                all_peaks.append(p)

finally:
    # Rilascia sempre l'hardware
    wavegen.w1.stop()
    ad2.power(False)
    ad2.close()
    print("Hardware spento correttamente.")

# ==========================================================================
#  VISUALIZZAZIONE (Stile Laboratorio)
# ==========================================================================
plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')

# Usiamo lo scatter plot con punti piccolissimi (pixel)
# s=0.1 e alpha=0.3 permettono di vedere le zone più dense (attrattori stabili)
ax.scatter(all_amplitudes, all_peaks, s=0.5, c='blue', alpha=0.3, marker=',')

ax.set_xlabel("Driving Amplitude [V]")
ax.set_ylabel("Signal Peaks (Vc) [V]")
ax.set_title(f"Diagramma di Biforcazione (Sweep in Ampiezza)\nf = {F_DRIVE} Hz")

ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(A0, A1)

# Se i dati sono stati raccolti, adatta l'asse Y
if len(all_peaks) > 0:
    ymin, ymax = np.min(all_peaks), np.max(all_peaks)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)

plt.tight_layout()
plt.show()