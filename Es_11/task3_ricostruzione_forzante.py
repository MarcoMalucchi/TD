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
import time
from utils.labplot import save_lab_figure

path = '/home/marco/Desktop/Uni_anno3/TD/Es_11/'

# -[Configurazione AD2]--------------------------------------------------------
#   1. Connessiene con AD2 e selezione configurazione 
ad2 = tdwf.AD2()
ad2.vdd = 5
ad2.vss = -5
ad2.power(True)
#   3. Configurazione generatore funzioni
wgen = tdwf.WaveGen(ad2.hdwf)
wgen.w1.func = tdwf.funcTriangle
wgen.w1.freq = 1e3
wgen.w1.ampl = 4
#wgen.w1.offs = 1
wgen.w1.start()
#   3. Configurazione oscilloscopio
scope = tdwf.Scope(ad2.hdwf)
scope.fs=1e5
scope.npt=101
scope.ch1.rng = 50
scope.ch2.rng = 50
scope.ch1.avg=True
scope.ch2.avg=True
scope.trig(True, level = 0.5, hist = 0.4, sour = tdwf.trigsrcCh1)
time.sleep(0.1)
scope.sample()

Ch1_1 = scope.ch1.vals
Ch2_1 = scope.ch2.vals

# scope.sample()

# Ch1_2 = scope.ch1.vals
# Ch2_2 = scope.ch2.vals

fig, ax = plt.subplots(1,1, figsize=(10,6), dpi=100)

#ax.plot(Ch2_1, Ch1_1, linestyle='-', marker='.', color="tab:blue", label = "Forzante primo periodo")

block_size = 101

N = len(Ch1_1)

# choose automatic matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, start in enumerate(range(0, N, block_size)):
    
    end = start + block_size
    
    ax.plot(
        Ch2_1[start:end],
        Ch1_1[start:end],
        linestyle='-',
        marker='.',
        color=colors[i % len(colors)],
        label=f'Periodo {i+1}'
    )


#ax.plot(Ch2_2, Ch1_2, linestyle='-', marker='.', color="tab:red", label = "Forzante secondo periodo")
ax.set_xlabel(r'$V_{C}$ [V]')
ax.set_ylabel(r'$V_{W1}$ [V]')
ax.grid()
ax.legend()

save_lab_figure(fig, ax, "task3_ricostruzione_forzante_alta_ampiezza", mode="both", folder=path + "logbook")

plt.show()

ad2.close()
