import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

# ================= PATH =================

target_path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/spazzata_completa_x/FRF/'

if "_x" in target_path.lower():
    axis = "X"
elif "_y" in target_path.lower():
    axis = "Y"
else:
    raise ValueError("Cannot detect axis from path.")

# ================= LOAD DATA =================

data = np.loadtxt(os.path.join(target_path, "SWEEP_FRF.csv"),
                  delimiter=",", skiprows=1)

data = data[np.argsort(data[:,0])]
f_exp = data[:,0]

if axis == "X":
    H_exp = data[:,1] * np.exp(1j*data[:,5])
    f_modes = [0.88, 4.75, 5.88, 15.3]
else:
    H_exp = data[:,3] * np.exp(1j*data[:,7])
    f_modes = [1.76, 4.75, 5.88, 15.3]

# ================= MODEL =================

def mode(w, w0, zeta, R):
    return R / (w0**2 - w**2 + 1j*2*zeta*w0*w)

def total_frf_acc(w, params):
    H = 0
    for w0, z, R in params:
        H += mode(w, w0, z, R)
    return w**2 * H   # critical correction


f = np.linspace(f_exp.min(), f_exp.max(), 6000)
w = 2*np.pi*f
w_modes = [2*np.pi*fi for fi in f_modes]

R_init = [1.0, 0.5, -0.5, 0.1]
z_init = [0.03, 0.02, 0.02, 0.015]

params = list(zip(w_modes, z_init, R_init))
H = total_frf_acc(w, params)

# ================= PLOT =================

fig, (ax_mag, ax_phase) = plt.subplots(2,1, figsize=(12,10))
plt.subplots_adjust(left=0.1, bottom=0.55)

line_mag, = ax_mag.plot(f, np.abs(H))
line_phase, = ax_phase.plot(f, np.unwrap(np.angle(H)))

ax_mag.scatter(f_exp, np.abs(H_exp), s=15, color='black')
ax_phase.scatter(f_exp, np.unwrap(np.angle(H_exp)), s=15, color='black')

ax_mag.set_yscale('log')
ax_mag.set_ylabel("Amplitude ratio")
ax_mag.grid(True, which='both')

ax_phase.set_ylabel("Phase [rad]")
ax_phase.set_xlabel("Frequency [Hz]")
ax_phase.grid(True)

# ================= SLIDERS =================

sliders_R = []
sliders_z = []

y_base = 0.45

for i in range(4):

    axR = plt.axes([0.15, y_base - i*0.06, 0.7, 0.025])
    sR = Slider(axR, f'Residue R{i+1}', -5.0, 5.0,
                valinit=R_init[i], valfmt='%1.3f')
    sliders_R.append(sR)

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

    H = total_frf_acc(w, new_params)
    line_mag.set_ydata(np.abs(H))
    line_phase.set_ydata(np.unwrap(np.angle(H)))
    fig.canvas.draw_idle()

    print("\nCurrent guess:")
    for i in range(4):
        print(f"Mode {i+1}: f={f_modes[i]:.3f} Hz | ζ={sliders_z[i].val:.4f} | R={sliders_R[i].val:.4f}")

for s in sliders_R + sliders_z:
    s.on_changed(update)

plt.show()
