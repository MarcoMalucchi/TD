import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Frequency axis
f = np.linspace(0.2, 17, 6000)
w = 2*np.pi*f

def mode(w, w0, zeta, R):
    return R / (w0**2 - w**2 + 1j*2*zeta*w0*w)

def total_frf(w, params):
    H = 0
    for w0, z, R in params:
        H += mode(w, w0, z, R)
    return H

# Modal frequencies
f_modes = [0.88, 4.75, 5.88, 15.3]
w_modes = [2*np.pi*fi for fi in f_modes]

# Initial parameters
R_init = [1.0, 0.4, -0.6, 0.1]
z_init = [0.03, 0.02, 0.02, 0.015]

params = list(zip(w_modes, z_init, R_init))
H = total_frf(w, params)

fig, (ax_mag, ax_phase) = plt.subplots(2,1, figsize=(12,10))
plt.subplots_adjust(left=0.1, bottom=0.55)

line_mag, = ax_mag.plot(f, np.abs(H))
line_phase, = ax_phase.plot(f, np.unwrap(np.angle(H)))

ax_mag.set_yscale('log')
ax_mag.set_ylabel("Magnitude")
ax_mag.grid(True, which='both', alpha=0.3)

ax_phase.set_ylabel("Phase [rad]")
ax_phase.set_xlabel("Frequency [Hz]")
ax_phase.grid(True, alpha=0.3)

sliders_R = []
sliders_z = []

y_base = 0.45

for i in range(4):

    # Residue sliders
    axR = plt.axes([0.15, y_base - i*0.06, 0.7, 0.025])
    sR = Slider(axR, f'Residue R{i+1}', -2.0, 2.0,
                valinit=R_init[i], valfmt='%1.3f')
    sliders_R.append(sR)

    # Damping sliders (below)
    axZ = plt.axes([0.15, y_base - i*0.06 - 0.035, 0.7, 0.02])
    sZ = Slider(axZ, f'Damping ζ{i+1}', 0.001, 0.1,
                valinit=z_init[i], valfmt='%1.4f')
    sliders_z.append(sZ)

def update(val):
    new_params = []
    for i in range(4):
        w0 = w_modes[i]
        z  = sliders_z[i].val
        R  = sliders_R[i].val
        new_params.append((w0, z, R))

    H = total_frf(w, new_params)
    line_mag.set_ydata(np.abs(H))
    line_phase.set_ydata(np.unwrap(np.angle(H)))
    fig.canvas.draw_idle()

for s in sliders_R + sliders_z:
    s.on_changed(update)

plt.show()
