import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.labelweight': 'bold'
})

# ================= PATH =================

target_path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/spazzata_completa_y/FRF/'

# Detect axis automatically
if "_x" in target_path.lower():
    axis = "X"
elif "_y" in target_path.lower():
    axis = "Y"
else:
    raise ValueError("Cannot detect axis from path name.")

print(f"\nDetected sweep axis: {axis}")

# ================= MODEL =================

def modal_complex_model(f, params):
    w = 2*np.pi*f
    n_modes = len(params)//3
    H = 0
    for k in range(n_modes):
        wk, Rk, zk = params[3*k], params[3*k+1], params[3*k+2]
        H += Rk / (wk**2 - w**2 + 1j*2*zk*wk*w)
    return H

def residuals(params, f, H_exp):
    H_model = modal_complex_model(f, params)
    return np.concatenate([
        np.real(H_model - H_exp),
        np.imag(H_model - H_exp)
    ])

# ================= LOAD DATA =================

data = np.loadtxt(os.path.join(target_path, "SWEEP_FRF.csv"),
                  delimiter=",", skiprows=1)

data = data[np.argsort(data[:,0])][10:]  # sort + discard first 10

f = data[:,0]

if axis == "X":
    H_mag = data[:,1]
    sH    = data[:,2]
    phase = data[:,5]
    expected_fs = [0.88, 4.75, 5.88, 15.3]
else:
    H_mag = data[:,3]
    sH    = data[:,4]
    phase = data[:,7]
    expected_fs = [1.76, 4.75, 5.88, 15.3]

# Threshold selection
mask = H_mag > 1e-3
f_fit = f[mask]
H_complex = H_mag[mask] * np.exp(1j*phase[mask])

# ================= INITIAL GUESS =================

p0 = []
for fs in expected_fs:
    p0 += [2*np.pi*fs, 1.0, 0.02]
p0 = np.array(p0)

# ================= FIT =================

result = least_squares(
    residuals,
    p0,
    args=(f_fit, H_complex),
    max_nfev=40000
)

popt = result.x

# ================= STATISTICS =================

H_fit_complex = modal_complex_model(f_fit, popt)
res_complex = H_complex - H_fit_complex

chi2 = np.sum(np.abs(res_complex)**2)
dof = 2*len(f_fit) - len(popt)
red_chi2 = chi2 / dof
sigma_chi2 = np.sqrt(2/dof)

# ================= PRINT RESULTS =================

print("\n" + "="*70)
print(f"MODAL IDENTIFICATION RESULTS — AXIS {axis}")
print(f"DOF = {dof}")
print(f"Reduced Chi² = {red_chi2:.5f} ± {sigma_chi2:.5f}")
print("-"*70)
print(f"{'Mode':<6}{'f_n [Hz]':<12}{'Residue R':<14}{'Damping ζ':<12}")

for k in range(len(expected_fs)):
    fn = popt[3*k]/(2*np.pi)
    Rk = popt[3*k+1]
    zk = popt[3*k+2]
    print(f"{k+1:<6}{fn:<12.4f}{Rk:<14.5f}{zk:<12.5f}")

print("="*70 + "\n")

# ================= FULL MODEL =================

H_full = modal_complex_model(f, popt)

# ================= PLOT 1 — AMPLITUDE =================

fig1, (ax1, ax2) = plt.subplots(2,1, figsize=(9,8), sharex=True)

# Top: fit + data
ax1.set_yscale('log')
ax1.plot(f, np.abs(H_full), 'r', lw=2, label='Model')
ax1.scatter(f, H_mag, s=15, alpha=0.6, label='Data')
ax1.set_ylabel("Amplitude ratio")
ax1.legend()
ax1.grid(True, which='both')

# Bottom: residuals
amp_res = H_mag - np.abs(H_full)
ax2.axhline(0, color='k', lw=1)
ax2.scatter(f, amp_res, s=10)
ax2.set_yscale('symlog', linthresh=1e-4)
ax2.set_ylabel("Residuals")
ax2.set_xlabel("Frequency [Hz]")
ax2.grid(True, which='both')

fig1.suptitle(f"Amplitude Fit — Axis {axis}")
plt.show()

# ================= PLOT 2 — PHASE =================

fig2, (ax3, ax4) = plt.subplots(2,1, figsize=(9,8), sharex=True)

phase_model = np.unwrap(np.angle(H_full))

# Top: phase fit
ax3.plot(f, phase_model, 'r', lw=2, label='Model')
ax3.scatter(f, phase, s=15, alpha=0.6, label='Data')
ax3.set_ylabel("Phase [rad]")
ax3.legend()
ax3.grid(True)

# Bottom: residuals
phase_res = phase - phase_model
ax4.axhline(0, color='k', lw=1)
ax4.scatter(f, phase_res, s=10)
ax4.set_yscale('symlog', linthresh=1e-2)
ax4.set_ylabel("Residuals")
ax4.set_xlabel("Frequency [Hz]")
ax4.grid(True)

fig2.suptitle(f"Phase Fit — Axis {axis}")
plt.show()
