import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Frequency axis
f = np.linspace(0.1, 16, 4000)
w = 2*np.pi*f

# Complex mechanical FRF (displacement form)
def complex_mode(w, w0, zeta, R):
    return R / (w0**2 - w**2 + 1j*2*zeta*w0*w)

def total_frf(w, w01, z1, R1, w02, z2, R2):
    H1 = complex_mode(w, w01, z1, R1)
    H2 = complex_mode(w, w02, z2, R2)
    return H1 + H2

# Initial parameters (roughly inspired by your sweep)
f01_init = 1.0
f02_init = 6.0
z1_init = 0.03
z2_init = 0.02
R1_init = 1.0
R2_init = -0.5   # negative residue to allow dip

# Convert to angular frequency
w01_init = 2*np.pi*f01_init
w02_init = 2*np.pi*f02_init

# Initial FRF
H = total_frf(w, w01_init, z1_init, R1_init,
                 w02_init, z2_init, R2_init)

# Figure
fig, (ax_mag, ax_phase) = plt.subplots(2,1, figsize=(10,8))
plt.subplots_adjust(left=0.1, bottom=0.35)

line_mag, = ax_mag.plot(f, np.abs(H))
line_phase, = ax_phase.plot(f, np.angle(H))

ax_mag.set_yscale('log')
ax_mag.set_ylabel("Magnitude")
ax_mag.grid(True, which='both', alpha=0.3)

ax_phase.set_ylabel("Phase [rad]")
ax_phase.set_xlabel("Frequency [Hz]")
ax_phase.grid(True, alpha=0.3)

# Sliders
def slider_axis(ypos):
    return plt.axes([0.15, ypos, 0.7, 0.03])

sf01 = Slider(slider_axis(0.28), 'f01', 0.5, 3.0, valinit=f01_init)
sf02 = Slider(slider_axis(0.24), 'f02', 4.0, 10.0, valinit=f02_init)
sz1  = Slider(slider_axis(0.20), 'zeta1', 0.001, 0.1, valinit=z1_init)
sz2  = Slider(slider_axis(0.16), 'zeta2', 0.001, 0.1, valinit=z2_init)
sR1  = Slider(slider_axis(0.12), 'R1', -2.0, 2.0, valinit=R1_init)
sR2  = Slider(slider_axis(0.08), 'R2', -2.0, 2.0, valinit=R2_init)

def update(val):
    w01 = 2*np.pi*sf01.val
    w02 = 2*np.pi*sf02.val
    z1  = sz1.val
    z2  = sz2.val
    R1  = sR1.val
    R2  = sR2.val

    H = total_frf(w, w01, z1, R1, w02, z2, R2)

    line_mag.set_ydata(np.abs(H))
    line_phase.set_ydata(np.angle(H))

    fig.canvas.draw_idle()

for s in [sf01, sf02, sz1, sz2, sR1, sR2]:
    s.on_changed(update)

plt.show()
