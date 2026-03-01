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
from utils.labplot import save_lab_figure, float_to_str

#path = "/home/marco/Desktop/Uni_anno3/TD/Es_11/acquisizioni/"

# -[Configurazione Analog Discovery 2]-----------------------------------------
#   1. Connessiene con AD2
ad2 = tdwf.AD2()
ad2.vdd = 5
ad2.vss = -5
ad2.power(True)
#   2. Configurazione generatore di funzioni
wgen = tdwf.WaveGen(ad2.hdwf)
wgen.w1.ampl = 0.82     #0.34, 0.35, 0.6 forma elegante che piace ad Alessia, la mettiamo nella presentazione così lei è felice yeeeiiiii
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
scope.fs=1e6
scope.npt=16384
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
    global flag_rescale
    if event.key == 'a':  # => cambia l'ampiezza della forzante
        wgen.w1.ampl = float(input("Nuova ampiezza forzante: "))
        flag_rescale = True
        fig._suptitle.set_text(f'Risposta circuito con forzante sinusoidale ' f'{wgen.w1.ampl} V, {wgen.w1.freq} Hz')
    if event.key == 'x':  # => salva immagine

        name = f'task5_ampl_{float_to_str(wgen.w1.ampl, 3)}'
        save_lab_figure(fig, [ax1, ax2], name, mode="both", folder_standard='logbook', folder_presentation='presentazione')
        print(f"Figura salvata, nome: {name}")

        #filename = input("Esporta dati su file: ")
        # data = np.column_stack((scope.time.vals, scope.ch1.vals, scope.ch2.vals))
        # if scope.npt > 8192:
        #     info =  f"Acquisizione Analog Discovery 2 - Lunga durata\ntime\tch1\tch2"
        # else:
        #     info =  f"Acquisizione Analog Discovery 2\nTimestamp {scope.time.t0}\ntime\tch1\tch2"
        # np.savetxt(filename, data, delimiter='\t', header=info)
    if event.key == ' ':  # => run/pausa misura
        flag_acq = not flag_acq
        print("ACQ RUN" if flag_acq else "ACQ PAUSED")
    if event.key == 'escape':  # => esci dalla misura
        flag_run = False

#-[Gestione riscalamento assi immagine]----------------------------------------
def limits_with_margin(data, margin=0.05):
    dmin = np.min(data)
    dmax = np.max(data)
    span = dmax - dmin
    return dmin - margin*span, dmax + margin*span

# -[Ciclo di misura]-----------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
fig.canvas.mpl_connect("close_event", on_close)
fig.canvas.mpl_connect('key_press_event', on_key)
flag_run = True
flag_acq = True
flag_first = True
flag_rescale = True
while flag_run:
    if flag_acq: # l'acquisizione è attiva?
        #time.sleep(0.1)
        scope.sample()
    # Visualizzazione
    if flag_first:
        flag_first = False
        hp1, = ax1.plot(scope.time.vals, scope.ch1.vals, linestyle='-', label="forma d'onda", color="tab:blue") 
        hp2, = ax2.plot(scope.ch1.vals, scope.ch2.vals, linestyle='-', label="traiettoria spazio fasi", color="tab:orange")
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        ax1.grid(True)
        ax2.grid(True)
        ax1.set_xlabel(r'time [s]', fontsize=15)
        ax1.set_ylabel(r'$V_C$ [V]', fontsize=15)
        ax2.set_xlabel(r'$V_C$ [V]', fontsize=15)
        ax2.set_ylabel(r'$Ri_L$ [V]', fontsize=15)
        fig.suptitle(f'Risposta circuito con forzante sinusoidale ' f'{wgen.w1.ampl} V, {wgen.w1.freq} Hz')
        plt.tight_layout()
        plt.show(block = False)
    else:
        hp2.set_xdata(scope.ch1.vals)
        hp2.set_ydata(scope.ch2.vals)
        hp1.set_ydata(scope.ch1.vals)
        hp1.set_xdata(scope.time.vals)
        if flag_rescale:

            ax1.set_xlim(*limits_with_margin(scope.time.vals))
            ax1.set_ylim(*limits_with_margin(scope.ch1.vals))

            ax2.set_xlim(*limits_with_margin(scope.ch1.vals))
            ax2.set_ylim(*limits_with_margin(scope.ch2.vals))

            flag_rescale = False
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.01)

plt.close(fig)
ad2.close()