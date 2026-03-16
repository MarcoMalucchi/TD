import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tdwf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
from utils.labplot import save_lab_figure, float_to_str, save_experiment_metadata


#################
# SETTAGGIO AD2.
#################

ad2 = tdwf.AD2()

# ---PER CIRCUITI ANALOGICI!!---
ad2.vdd = 5
ad2.vss = -5

# ---PER CIRCUITI LOGICI!!---
# ad2.vdd = 5
# ad2.vss = 0

ad2.power(True)

wgen = tdwf.WaveGen(ad2.hdwf)
wgen.w1.ampl = 1.5
wgen.w1.freq = 100
wgen.w1.offs = 1.5
wgen.w1.func = tdwf.funcTriangle
wgen.w1.duty = 90
wgen.w1.start()

# wgen.w2.ampl = 2.5
# wgen.w2.freq = 100
# wgen.w2.offs = 2.5
# wgen.w2.func = tdwf.funcSquare
# wgen.w2.duty = 50
#wgen.w2.phase = np.pi
#wgen.w2.sync()
# wgen.w2.start()

scope = tdwf.Scope(ad2.hdwf)
scope.fs = 1e5
scope.npt = 4096
scope.ch1.rng = 50
scope.ch2.rng = 50
scope.ch1.avg = True
scope.ch2.avg = True
scope.trig(True, level=0.5, hist=0.4,
           sour=tdwf.trigsrcCh1, cond=tdwf.trigslopeFall)

time.sleep(0.1)


#######################
# LIVE PLOT FUNCTIONS
#######################

def get_title():
    title_str = f'W1: {wgen.w1.ampl} V, {wgen.w1.freq} Hz, {wgen.w1.offs} V, {wgen.w1.duty} %'

    try:
        if hasattr(wgen, "w2") and wgen.w2 is not None:
            title_str += f'; W2: {wgen.w2.ampl} V, {wgen.w2.freq} Hz, {wgen.w2.offs} V, {wgen.w2.duty} %'
    except:
        pass

    title_str += f'; Scope: {scope.fs} Sa/s, {scope.npt} Sa'

    return title_str


def on_close(event):
    global flag_run
    flag_run = False


def on_key(event):
    global flag_run, flag_acq
    global req_freq, req_ampl, req_offset, req_duty, req_fs, req_points

    if event.key == 'f':
        req_freq = True

    if event.key == 'a':
        req_ampl = True

    if event.key == 'o':
        req_offset = False

    if event.key == 'd':
        req_duty = True

    if event.key == 's':
        req_fs = True

    if event.key == 'p':
        req_points = True

    if event.key == 'x':
        name = f'esame_ampl_{float_to_str(wgen.w1.ampl,3)}_freq_{float_to_str(wgen.w1.freq,3)}'

        metadata = {
            "w1_amplitude": wgen.w1.ampl,
            "w1_frequency": wgen.w1.freq,
            "sampling_rate": scope.fs,
            "points": scope.npt
        }

        try:
            metadata["w2_amplitude"] = wgen.w2.ampl
            metadata["w2_frequency"] = wgen.w2.freq
        except AttributeError:
            pass

        save_experiment_metadata(
            fig=fig,
            axes=[ax1],
            name=name,
            data=np.column_stack([scope.time.vals,
                                  scope.ch1.vals,
                                  scope.ch2.vals]),
            header='time\tch1\tch2',
            metadata=metadata
        )

        print("Dati e immagine salvati.")

    if event.key == ' ':
        flag_acq = not flag_acq
        print("ACQ RUN" if flag_acq else "ACQ PAUSED")

    if event.key == 'escape':
        flag_run = False

    if event.key == 'h':
        print(
            "Controls:\n"
            "SPACE : start/stop acquisition\n"
            "a     : change amplitude\n"
            "f     : change frequency\n"
            "o     : change offset\n"
            "d     : change duty cycle\n"
            "s     : change sampling rate\n"
            "p     : change number of points\n"
            "x     : save acquisition\n"
            "ESC   : quit\n"
            "h     : help"
        )


def limits_with_margin(data, margin=0.05):
    dmin = np.min(data)
    dmax = np.max(data)
    span = dmax - dmin

    if span == 0:
        span = abs(dmin) if dmin != 0 else 1

    return dmin - margin*span, dmax + margin*span


###################
# CICLO DI MISURA
###################

fig, ax1 = plt.subplots(figsize=(12, 6))

fig.canvas.mpl_connect("close_event", on_close)
fig.canvas.mpl_connect("key_press_event", on_key)

flag_run = True
flag_acq = True
flag_first = True

flag_rescale = False
flag_freq = False
flag_fs = False
flag_points = False

req_freq = False
req_ampl = False
req_offset = False
req_duty = False
req_fs = False
req_points = False

try:

    while flag_run:

        # ---- HANDLE USER REQUESTS ----

        if req_freq:
            control = int(input("Generatore (1,2,3=entrambi)? "))

            if control in (1, 3):
                wgen.w1.freq = float(input("Nuova frequenza W1: "))

            if control in (2, 3):
                wgen.w2.freq = float(input("Nuova frequenza W2: "))

            flag_freq = True
            title.set_text(get_title())
            req_freq = False

        if req_ampl:
            control = int(input("Generatore (1,2,3=entrambi)? "))

            if control in (1, 3):
                wgen.w1.ampl = float(input("Nuova ampiezza W1: "))

            if control in (2, 3):
                wgen.w2.ampl = float(input("Nuova ampiezza W2: "))

            flag_rescale = True
            title.set_text(get_title())
            req_ampl = False

        if req_offset:
            control = int(input("Generatore (1,2,3=entrambi)? "))

            if control in (1, 3):
                wgen.w1.offset = float(input("Nuovo offset W1: "))

            if control in (2, 3):
                wgen.w2.offset = float(input("Nuovo offset W2: "))

            flag_rescale = True
            title.set_text(get_title())
            req_offset = False

        if req_duty:
            control = int(input("Generatore (1,2,3=entrambi)? "))

            if control in (1, 3):
                wgen.w1.duty = float(input("Nuovo duty cycle W1: "))

            if control in (2, 3):
                wgen.w2.duty = float(input("Nuovo duty cycle W2: "))

            flag_rescale = True
            title.set_text(get_title())
            req_duty = False

        if req_fs:
            scope.fs = float(input("Nuova frequenza di campionamento: "))
            flag_fs = True
            title.set_text(get_title())
            req_fs = False

        if req_points:
            scope.npt = int(input("Nuovo numero punti acquisiti: "))
            flag_points = True
            title.set_text(get_title())
            req_points = False

        # ---- ACQUISITION ----

        if flag_acq:
            scope.sample()

        # ---- PLOT ----

        if flag_first:

            flag_first = False

            hp1, = ax1.plot(scope.time.vals,
                            scope.ch1.vals,
                            label="Ch1",
                            color="tab:orange")

            hp2, = ax1.plot(scope.time.vals,
                            scope.ch2.vals,
                            label="Ch2",
                            color="tab:blue")

            ax1.legend(loc="upper right")
            ax1.grid(True)

            ax1.set_xlabel("time [s]", fontsize=15)
            ax1.set_ylabel("$V_C$ [V]", fontsize=15)

            title = fig.suptitle(get_title())

            ax1.set_xlim(*limits_with_margin(scope.time.vals))
            ydata = np.concatenate([scope.ch1.vals, scope.ch2.vals])
            ax1.set_ylim(*limits_with_margin(ydata))

            plt.show(block=False)

        else:

            hp1.set_xdata(scope.time.vals)
            hp1.set_ydata(scope.ch1.vals)

            hp2.set_xdata(scope.time.vals)
            hp2.set_ydata(scope.ch2.vals)

            if flag_rescale or flag_freq or flag_fs or flag_points:

                ax1.set_xlim(*limits_with_margin(scope.time.vals))
                ydata = np.concatenate([scope.ch1.vals, scope.ch2.vals])
                ax1.set_ylim(*limits_with_margin(ydata))

                flag_rescale = False
                flag_freq = False
                flag_fs = False
                flag_points = False

            fig.canvas.draw()
            fig.canvas.flush_events()

        #plt.pause(0.01)
        time.sleep(0.001)

finally:

    plt.close(fig)
    ad2.close()