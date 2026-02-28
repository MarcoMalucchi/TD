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

#path = "/home/marco/Desktop/Uni_anno3/TD/Es_11/acquisizioni/"

# -[Configurazione Analog Discovery 2]-----------------------------------------
#   1. Connessiene con AD2
ad2 = tdwf.AD2()
ad2.vdd = 5
ad2.vss = -5
ad2.power(True)
#   2. Configurazione generatore di funzioni
wgen = tdwf.WaveGen(ad2.hdwf)
wgen.w1.ampl = 1.5     #0.34, 0.35, 0.6 forma elegante che piace ad Alessia, la mettiamo nella presentazione così lei è felice yeeeiiiii
wgen.w1.freq = 7300
wgen.w1.offs = 0.0
wgen.w1.func = tdwf.funcSine
wgen.w1.duty = 50
#wgen.w1.phase = np.pi
#wgen.w1.sync()
wgen.w1.start()

# wgen.w2.ampl = 2.5
# wgen.w2.freq = 5
# wgen.w2.offs = 2.5
# wgen.w2.func = tdwf.funcSquare
# wgen.w2.duty = 50
#wgen.w2.sync()
# wgen.w2.start()

#   3. Configurazione oscilloscopio
scope = tdwf.Scope(ad2.hdwf)
scope.fs=1e5
scope.npt=65536
scope.ch1.rng = 50
scope.ch2.rng = 50
scope.ch1.avg=True
scope.ch2.avg=True
time.sleep(0.1)
scope.trig(True, level = 0.5, hist = 0.4, sour = tdwf.trigsrcCh1, cond=tdwf.trigslopeFall)



#   4. Configurazione powersupply

# -[Funzioni di gestione eventi]-----------------------------------------------
def on_close(event):
    global flag_run
    flag_run = False
def on_key(event):
    global flag_run
    global flag_acq
    if event.key == 'x':  # => export su file

        name = input("Inserire nome per la figura: ")
        save_lab_figure(fig, ax, name, mode="both", folder_standard='logbook', folder_presentation='presentazione')
        print("Figura salvata")

        #filename = input("Esporta dati su file: ")
        # data = np.column_stack((scope.time.vals, scope.ch1.vals, scope.ch2.vals))
        # if scope.npt > 8192:
        #     info =  f"Acquisizione Analog Discovery 2 - Lunga durata\ntime\tch1\tch2"
        # else:
        #     info =  f"Acquisizione Analog Discovery 2\nTimestamp {scope.time.t0}\ntime\tch1\tch2"
        # np.savetxt(filename, data, delimiter='\t', header=info)
    if event.key == ' ':  # => run/pausa misura
        flag_acq = not flag_acq
    if event.key == 'escape':  # => esci dalla misura
        flag_run = False

# -[Ciclo di misura]-----------------------------------------------------------
fig, ax = plt.subplots(figsize=(12,6))
fig.canvas.mpl_connect("close_event", on_close)
fig.canvas.mpl_connect('key_press_event', on_key)
flag_run = True
flag_acq = True
flag_first = True
while flag_run:
    if flag_acq: # l'acquisizione è attiva?
        #time.sleep(0.1)
        scope.sample()
    # Visualizzazione
    if flag_first:
        flag_first = False
        hp1, = plt.plot(scope.ch1.vals, scope.ch2.vals, linestyle='-', marker='.', label="Ch1", color="tab:orange")
        plt.legend()
        plt.grid(True)
        plt.xlabel(r'$V_C$ [V]', fontsize=15)
        plt.ylabel(r'$Ri_L$ [V]', fontsize=15)
        plt.title("User interaction: x|space|escape")
        plt.tight_layout()
        plt.show(block = False)
    else:
        hp1.set_xdata(scope.ch1.vals)
        hp1.set_ydata(scope.ch2.vals)
        fig.canvas.draw()
        fig.canvas.flush_events()

plt.close(fig)
ad2.close()