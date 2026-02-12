import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ================= PATH =================

target_path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/spazzata_completa_y/FRF/'

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
    amp_exp   = data[:,1]
    amp_err   = data[:,2]
    phase_exp = data[:,5]
    phase_err = data[:,6]
    f_modes = [0.88, 4.75, 5.88, 15.3]
else:
    amp_exp   = data[:,3]
    amp_err   = data[:,4]
    phase_exp = data[:,7]
    phase_err = data[:,8]
    f_modes = [1.76, 4.75, 5.88, 15.3]

phase_exp = np.unwrap(phase_exp)

# Clean uncertainties
amp_err = np.abs(amp_err)
phase_err = np.abs(phase_err)

amp_err[amp_err == 0] = np.median(amp_err)
phase_err[phase_err == 0] = np.median(phase_err)

# Safety floor
amp_err = np.maximum(amp_err, 0.05*np.median(amp_exp))
phase_err = np.maximum(phase_err, 0.02)


# ================= MODEL =================

def modal_acc_model(f, *params):
    w = 2*np.pi*f
    n_modes = len(params)//3
    H = 0
    for k in range(n_modes):
        wk = params[3*k]
        Rk = params[3*k+1]
        zk = params[3*k+2]
        H += Rk / (wk**2 - w**2 + 1j*2*zk*wk*w)
    return w**2 * H   # acceleration model (SIGN VERIFIED)

def model_amplitude(f, *params):
    return np.abs(modal_acc_model(f, *params))

def model_phase(f, *params):
    return np.unwrap(np.angle(modal_acc_model(f, *params)))

# ================= INITIAL GUESS =================

p0 = []
for fi in f_modes:
    p0.extend([2*np.pi*fi, 1.0, 0.02])  # wk, Rk, zk

# ================= AMPLITUDE FIT =================

popt_amp, pcov_amp = curve_fit(
    model_amplitude,
    f_exp,
    amp_exp,
    p0=p0,
    sigma=amp_err,
    absolute_sigma=True,
    maxfev=30000
)

perr_amp = np.sqrt(np.diag(pcov_amp))

res_amp = amp_exp - model_amplitude(f_exp, *popt_amp)
chi2_amp = np.sum((res_amp/amp_err)**2)
dof_amp = len(f_exp) - len(popt_amp)
chi2red_amp = chi2_amp / dof_amp
chi2red_err_amp = np.sqrt(2/dof_amp)

# ================= PHASE FIT =================

popt_phase, pcov_phase = curve_fit(
    model_phase,
    f_exp,
    phase_exp,
    p0=p0,
    sigma=phase_err,
    absolute_sigma=True,
    maxfev=30000
)

perr_phase = np.sqrt(np.diag(pcov_phase))

res_phase = phase_exp - model_phase(f_exp, *popt_phase)
chi2_phase = np.sum((res_phase/phase_err)**2)
dof_phase = len(f_exp) - len(popt_phase)
chi2red_phase = chi2_phase / dof_phase
chi2red_err_phase = np.sqrt(2/dof_phase)

# ================= WEIGHTED AVERAGE =================

p_avg = []
p_avg_err = []

for i in range(len(popt_amp)):
    w1 = 1/perr_amp[i]**2
    w2 = 1/perr_phase[i]**2
    avg = (popt_amp[i]*w1 + popt_phase[i]*w2)/(w1+w2)
    err = np.sqrt(1/(w1+w2))
    p_avg.append(avg)
    p_avg_err.append(err)

p_avg = np.array(p_avg)
p_avg_err = np.array(p_avg_err)

# ================= PRINT RESULTS =================

def print_table(title, popt, perr, chi2, chi2err):
    print(f"\n================ {title} =================")
    print(f"Reduced chi^2 = {chi2:.4f} ± {chi2err:.4f}\n")
    print(f"{'Mode':<6}{'f_k [Hz]':<18}{'R_k':<18}{'ζ_k':<12}")
    print("-"*54)
    for i in range(len(popt)//3):
        fk = popt[3*i]/(2*np.pi)
        Rk = popt[3*i+1]
        zk = popt[3*i+2]
        ef = perr[3*i]/(2*np.pi)
        eR = perr[3*i+1]
        ez = perr[3*i+2]
        print(f"{i+1:<6}"
              f"{fk:>8.4f} ± {ef:<8.2e}"
              f"{Rk:>8.4f} ± {eR:<8.2e}"
              f"{zk:>8.4f} ± {ez:<8.2e}")

print_table("AMPLITUDE FIT", popt_amp, perr_amp,
            chi2red_amp, chi2red_err_amp)

print_table("PHASE FIT", popt_phase, perr_phase,
            chi2red_phase, chi2red_err_phase)

print_table("WEIGHTED AVERAGE", p_avg, p_avg_err,
            0, 0)

# ================= PLOTS =================

f_dense = np.linspace(f_exp.min(), f_exp.max(), 6000)

# -------- AMPLITUDE FIGURE --------

fig1, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(10,8))

ax1.errorbar(f_exp, amp_exp, yerr=amp_err,
             fmt='.', color='black',
             ecolor='black', capsize=3)

ax1.plot(f_dense, model_amplitude(f_dense, *popt_amp))
ax1.set_yscale('log')
ax1.set_title(f"Amplitude Fit - Axis {axis}")
ax1.set_ylabel("Amplitude ratio")
ax1.grid(True, which='both')

ax2.errorbar(f_exp, res_amp, yerr=amp_err,
             fmt='.', color='black',
             ecolor='black', capsize=3)

ax2.axhline(0)
ax2.set_yscale('symlog')
ax2.set_ylabel("Residuals (Amplitude ratio)")
ax2.set_xlabel("Frequency [Hz]")
ax2.grid(True)

# -------- PHASE FIGURE --------

fig2, (ax3, ax4) = plt.subplots(2,1, sharex=True, figsize=(10,8))

ax3.errorbar(f_exp, phase_exp, yerr=phase_err,
             fmt='.', color='black',
             ecolor='black', capsize=3)

ax3.plot(f_dense, model_phase(f_dense, *popt_phase))
ax3.set_title(f"Phase Fit - Axis {axis}")
ax3.set_ylabel("Phase [rad]")
ax3.grid(True)

ax4.errorbar(f_exp, res_phase, yerr=phase_err,
             fmt='.', color='black',
             ecolor='black', capsize=3)

ax4.axhline(0)
ax4.set_ylabel("Residuals [rad]")
ax4.set_xlabel("Frequency [Hz]")
ax4.grid(True)

plt.tight_layout()
plt.show()
