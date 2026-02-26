import tdwf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# ======================
# User option
# ======================
save = True   # <-- activate/deactivate saving
path = '/home/marco/Desktop/Uni_anno3/TD/Es_11/'

V0 = -5
V1 = 5
nV = 100
npt = 100
fs  = 1e6

Vv = np.linspace(V0,V1,nV)
Vv = np.concatenate([Vv, Vv[-2::-1]])
print(Vv, '\n')

# -[Configurazione AD2]--------------------------------------------------------
ad2 = tdwf.AD2()
ad2.vdd = 5
ad2.vss = -5
ad2.power(True)

wgen = tdwf.WaveGen(ad2.hdwf)
wgen.w1.func = tdwf.funcDC
wgen.w1.start()

scope = tdwf.Scope(ad2.hdwf)
scope.fs = fs
scope.npt = npt
scope.ch1.rng = 50
scope.ch2.rng = 50

# -[Live Plot]-----------------------------------------------------------
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
fig.canvas.manager.set_window_title('Spazzata voltaggio')
ax2.yaxis.tick_right()

Nsweep = len(Vv)

Ch1v = np.full(Nsweep, np.nan)
Ch2v = np.full(Nsweep, np.nan)

for ii, V in enumerate(Vv):
    wgen.w1.offs = V
    time.sleep(0.1)

    scope.sample()
    Ch1v[ii] = np.mean(scope.ch1.vals)
    Ch2v[ii] = np.mean(scope.ch2.vals)

    # ---- LEFT PANEL ----
    ax1.clear()
    ax1.plot(Vv[:ii+1], Ch1v[:ii+1], ".", color="tab:orange", label="Ch1")
    ax1.plot(Vv[:ii+1], Ch2v[:ii+1], ".", color="tab:blue", label="Ch2")
    ax1.grid(True)
    ax1.set_xlabel("W1 [V]", fontsize=15)
    ax1.set_ylabel("Signals [V]", fontsize=15)
    ax1.legend()

    # ---- RIGHT PANEL ----
    ax2.clear()
    ax2.plot(Ch1v[:ii+1], Ch2v[:ii+1], ".", color="tab:orange")
    ax2.grid(True)
    ax2.set_xlabel(r"$V_Q$ [V]", fontsize=15)
    ax2.set_ylabel(r"$V_{W1} [V]", fontsize=15)
    ax2.yaxis.set_label_position('right')

    plt.tight_layout()
    plt.pause(0.001)

# ======================
# Close hardware
# ======================
ad2.close()

# ======================
# Saving block
# ======================
if save:

    folder = path + "logbook"
    os.makedirs(folder, exist_ok=True)

    # ---- Standard version ----
    fig.savefig(
        os.path.join(folder, "profilo_forzante_1.png"),
        dpi=300
    )

    folder = path + "presentazione"

    # ---- Presentation version ----
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both',
                       labelsize=16,
                       width=2,
                       length=8)

        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)

        if ax.get_legend():
            ax.legend(fontsize=16)

    fig.set_size_inches(14, 8)

    fig.savefig(
        os.path.join(folder, "profilo_forzante_1.png"),
        dpi=300
    )

plt.show()