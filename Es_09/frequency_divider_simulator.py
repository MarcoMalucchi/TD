import numpy as np
from scipy.signal import square
import matplotlib.pyplot as plt

plt.close('all')

# Parameters
f1 = 6
f2 = 2                 # frequency in Hz
duration = 1.0        # seconds
sampling_rate = 1000  # Hz (samples per second)

# Time array
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Square wave
x1 = square(2 * np.pi * f1 * t)

x2 = square(2 * np.pi * f2 * t, duty=0.6666)  # duty cycle 2/3

# Plot
fig = plt.figure(figsize=(10, 6), dpi=100)
ax = fig.subplots()
ax.plot(t, x2, color = 'red', label="$Q_2$ (2 Hz)")
ax.plot(t, x1, linestyle='dashed', color='black', label="CLK (6 Hz)")
ax.set_xlabel("Tempo [s]")
ax.set_ylabel("Ampiezza [a.u.]")
ax.set_title("Diagramma temporale del divisore in frequenza")
ax.legend(loc='upper right')

savefig = False

if savefig:
    path1 = "/home/marco/Desktop/Uni_anno3/TD/Es_09/logbook/"
    path2 = "/home/marco/Desktop/Uni_anno3/TD/Es_09/presentazione/"

    plt.savefig(path1+"freq_divider_terzi_sim.png", dpi=300)

    ax.set_xlabel("Tempo [s]", fontsize='18')
    ax.set_ylabel("Ampiezza [a.u.]", fontsize='18')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='upper right', fontsize='16')

    plt.savefig(path2+"freq_divider_terzi_sim.png", dpi=300)

plt.show()


'''
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(t, x2, color = 'red', label="$Q_2$ (2 Hz)")
plt.plot(t, x1, linestyle='dashed', color='black', label="CLK (6 Hz)")
plt.xlabel("Tempo [s]")
plt.ylabel("Ampiezza [a.u.]")
plt.title("Diagramma temporale del divisore in frequenza")
plt.legend(loc='upper right')
plt.show()

savefig = False

if savefig:
    path1 = "/home/marco/Desktop/Uni_anno3/TD/Es_09/logbook/"
    path2 = "/home/marco/Desktop/Uni_anno3/TD/Es_09/presentazione/"

    plt.savefig(path1+"freq_divider_terzi_sim.png", dpi=300)

    plt.xlabel("Tempo [s]", fontsize='18')
    plt.ylabel("Ampiezza [a.u.]", fontsize='18')
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.legend(fontsize='16')

    plt.savefig(path2+"freq_divider_terzi_sim.png", dpi=300)

plt.show()
'''
