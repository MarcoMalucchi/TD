import sys
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks

# Assicurati che il percorso per tdwf sia corretto
sys.path.append(str(Path(__file__).resolve().parents[1]))
import tdwf

# ==========================================================================
#  PARAMETRI SWEEP E ACQUISIZIONE
# ==========================================================================
nf = 500            # Risoluzione asse X
f0 = 1e3
f1 = 10e3
fv = np.logspace(np.log10(f0), np.log10(f1), nf)

A_DRIVE = 0.8       # Prova a variare tra 0.5 e 5V per vedere le biforcazioni
FS = 1e6            # 1 MHz
NPT = 16384         # Abbastanza punti per catturare molti cicli
SETTLE_S = 0.04     # Tempo di assestamento

# ==========================================================================
#  CONFIGURAZIONE HARDWARE AD2
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
scope.ch1.rng = 5   # Imposta un range ragionevole (es. 5V)
scope.ch1.avg = False 

# Liste per i dati del plot
all_freqs = []
all_peaks = []

print(f"Inizio acquisizione... Ampiezza Drive: {A_DRIVE}V")

# ==========================================================================
#  CICLO DI MISURA
# ==========================================================================
try:
    for ff in tqdm(fv, desc="Scansione Biforcazione"):
        wavegen.w1.freq = float(ff)
        time.sleep(SETTLE_S)
        
        scope.sample()
        signal = scope.ch1.vals
        
        # --- LOGICA DI RILEVAMENTO PICCHI ROBUSTA ---
        # 1. Calcoliamo la distanza minima tra i picchi basandoci sulla frequenza attuale
        #    (Vogliamo circa 1 picco per periodo del segnale di drive)
        dist_min = max(1, int(FS / (ff * 1.2)))
        
        # 2. Cerchiamo i picchi. Usiamo una 'prominence' minima per ignorare il rumore di fondo
        #    La prominence assicura che il picco spicchi rispetto al rumore circostante.
        peaks_idx, _ = find_peaks(signal, distance=dist_min, prominence=0.01)
        
        # 3. Fallback: se non trova nulla con la prominence, prova senza restrizioni
        if len(peaks_idx) == 0:
            peaks_idx, _ = find_peaks(signal, distance=dist_min)
            
        # 4. Salvataggio dati
        if len(peaks_idx) > 0:
            peaks_vals = signal[peaks_idx]
            
            # Scartiamo i primi picchi dell'acquisizione (potrebbero essere transitori)
            # e teniamo gli ultimi (es. 20 picchi)
            n_keep = 20
            selected = peaks_vals[-n_keep:]
            
            for p in selected:
                all_freqs.append(ff)
                all_peaks.append(p)
        else:
            # Se continua a non trovare nulla, è un problema di segnale assente
            pass

finally:
    wavegen.w1.stop()
    ad2.power(False)
    ad2.close()
    print("Hardware rilasciato.")

# ==========================================================================
#  VISUALIZZAZIONE FINALE
# ==========================================================================
if len(all_peaks) == 0:
    print("ERRORE: Nessun dato raccolto. Controlla i collegamenti dello Scope.")
else:
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot con punti molto piccoli (marker=',') e trasparenza (alpha)
    # per visualizzare la densità degli stati (mappa di densità)
    ax.scatter(all_freqs, all_peaks, s=0.5, c='lime', alpha=0.3, marker=',')
    
    ax.set_xscale("log")
    ax.set_xlabel("Driving Frequency [Hz]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Signal Peaks [V]", fontsize=12, fontweight='bold')
    ax.set_title(f"Bifurcation Map - Drive Amplitude: {A_DRIVE}V", fontsize=15, fontweight='bold')
    
    ax.grid(True, which="both", alpha=0.2, linestyle='--')
    
    # Imposta i limiti Y in base ai dati reali per evitare grafici vuoti
    ax.set_ylim(np.min(all_peaks)*1.1, np.max(all_peaks)*1.1)
    
    plt.tight_layout()
    plt.show()