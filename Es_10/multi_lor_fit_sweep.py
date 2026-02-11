import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

plt.close('all')

# ================= MODEL (FRF amplitude) =================

def single_frf(f, f0, K, zeta):
    num = K * f**2
    den = np.sqrt((f0**2 - f**2)**2 + (2*zeta*f0*f)**2)
    return num / den

def multi_frf(f, *params):
    res = np.zeros_like(f)
    for i in range(0, len(params), 3):
        res += single_frf(f, params[i], params[i+1], params[i+2])
    return res


# ================= LOAD FRF DATA =================

path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/spazzata_completa_x/FRF/'
data = np.loadtxt(os.path.join(path, "SWEEP_FRF.csv"), delimiter=",", skiprows=1)

f_data = data[:,0]
Hx = data[:,1]
sHx = data[:,2]
Hy = data[:,3]
sHy = data[:,4]

# -------- choose axis ----------
H_data = Hx
sigma_data = sHx
axis_name = 'X'

# ================= SORT BY FREQUENCY =================

order = np.argsort(f_data)

f_data = f_data[order]
H_data = H_data[order]
sigma_data = sigma_data[order]



# ================= PEAK SELECTION (same logic as PSD script) =================

base_threshold = np.max(H_data) * 0.05
init_p, _ = find_peaks(H_data, height=base_threshold, distance=10)

if len(init_p) > 0:
    buffer_limit = base_threshold + (0.05 * (np.max(H_data) - base_threshold))
    peaks = [p for p in init_p if (f_data[p] >= 12.0 or H_data[p] > buffer_limit)]
    peaks = np.array(peaks)
else:
    peaks, buffer_limit = np.array([]), base_threshold

print(f"Detected peaks at: {f_data[peaks]}")


# ================= INITIAL GUESSES =================

guesses = []
for p in peaks:
    f0_g = f_data[p]
    K_g = H_data[p] * (f0_g**2)
    zeta_g = 0.02
    guesses.extend([f0_g, K_g, zeta_g])


# ================= BOUNDS =================

bounds_l, bounds_u = [], []
for i in range(len(peaks)):
    f0_g, K_g, zeta_g = guesses[3*i:3*i+3]
    bounds_l.extend([f0_g - 0.5, 0, 0.001])
    bounds_u.extend([f0_g + 0.5, K_g * 10, 0.2])


# ================= FIT =================

popt, pcov = curve_fit(
    multi_frf,
    f_data,
    H_data,
    p0=guesses,
    sigma=sigma_data,
    absolute_sigma=True,
    bounds=(bounds_l, bounds_u),
    maxfev=50000
)

perr = np.sqrt(np.diag(pcov))
fit_curve = multi_frf(f_data, *popt)
residuals = H_data - fit_curve


# ================= RESULTS + CHI SQUARE =================

print("\n=========== SWEEP FRF FIT RESULTS ===========")
for i in range(0, len(popt), 3):
    f0, K, zeta = popt[i:i+3]
    sf0, sK, sz = perr[i:i+3]
    Q = 1/(2*zeta)
    print(f"Mode {i//3+1}:")
    print(f"  f0   = {f0:.4f} ± {sf0:.4f} Hz")
    print(f"  zeta = {zeta:.5f} ± {sz:.5f}")
    print(f"  Q    = {Q:.2f}")

dof = len(f_data) - len(popt)
red_chi = np.sum((residuals / sigma_data)**2) / dof
print(f"\nReduced Chi-Square: {red_chi:.4f} (Expected Std Dev: {np.sqrt(2/dof):.4f})")


# ================= FIGURE A (fit + symlog residuals) =================

fig_comb, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

# --- DATA + FIT (semilog y) ---
ax1.set_yscale('log')
ax1.errorbar(
    f_data, H_data,
    yerr=sigma_data,
    fmt='o',
    markersize=3,
    alpha=0.6,
    label='Data'
)
ax1.plot(f_data, fit_curve, 'r-', linewidth=2, label='FRF Fit')

ax1.axhspan(base_threshold, buffer_limit, color='gray', alpha=0.15, label='Buffer Zone')
ax1.axvline(12, color='b', linestyle=':', alpha=0.5, label='12 Hz Filter')

ax1.set_ylabel("|H(f)|")
ax1.grid(True, which="both", alpha=0.2)
ax1.legend()

# --- RESIDUALS SYMLOG ---
ax2.set_yscale('symlog', linthresh=1e-4)
ax2.errorbar(
    f_data,
    residuals,
    yerr=sigma_data,
    fmt='o',
    markersize=3,
    alpha=0.6
)
ax2.axhline(0, color='k', linestyle='--')
ax2.set_ylabel("Res. (Symlog)")
ax2.grid(True, which="both", alpha=0.2)

ax2.set_xlabel("Frequency [Hz]")


# ================= FIGURE B (linear residuals) =================

fig_lin = plt.figure(figsize=(10, 5))
plt.errorbar(
    f_data,
    residuals,
    yerr=sigma_data,
    fmt='o',
    markersize=3,
    alpha=0.8
)
plt.axhline(0, color='black', linestyle='--')
plt.ylabel("Residuals (linear)")
plt.xlabel("Frequency [Hz]")
plt.grid(True, alpha=0.3)
plt.title(f"Linear Residuals - Sweep FRF ({axis_name})")

plt.show()
