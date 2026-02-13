import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ================= STILE GRAFICO =================

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2
})

# ================= PATH =================

target_path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/spazzata_completa_x/FRF/'
save = True   # <-- metti True per salvare

folder_name = os.path.basename(os.path.dirname(target_path.rstrip("/")))

if "_x" in folder_name.lower():
    axis = "X"
elif "_y" in folder_name.lower():
    axis = "Y"
else:
    raise ValueError("Impossibile determinare l'asse dal percorso.")

# ================= CARICAMENTO DATI =================

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

amp_err = np.abs(amp_err)
phase_err = np.abs(phase_err)

amp_err[amp_err == 0] = np.median(amp_err)
phase_err[phase_err == 0] = np.median(phase_err)

amp_err = np.maximum(amp_err, 0.05*np.median(amp_exp))
phase_err = np.maximum(phase_err, 0.02)

# ================= MODELLO MODALE =================

def modal_acc_model(f, *params):
    w = 2*np.pi*f
    n_modes = len(params)//3
    H = 0
    for k in range(n_modes):
        wk = params[3*k]
        Rk = params[3*k+1]
        zk = params[3*k+2]
        H += Rk / (wk**2 - w**2 + 1j*2*zk*wk*w)
    return w**2 * H

def model_amp(f, *params):
    return np.abs(modal_acc_model(f, *params))

def model_phase(f, *params):
    return np.unwrap(np.angle(modal_acc_model(f, *params)))

# ================= PARAMETRI INIZIALI =================

p0 = []
for fi in f_modes:
    p0.extend([2*np.pi*fi, 1.0, 0.02])

# ================= FIT AMPIEZZA =================

popt_amp, pcov_amp = curve_fit(
    model_amp, f_exp, amp_exp,
    p0=p0, sigma=amp_err,
    absolute_sigma=True, maxfev=30000
)

perr_amp = np.sqrt(np.diag(pcov_amp))

res_amp = amp_exp - model_amp(f_exp, *popt_amp)
chi2_amp = np.sum((res_amp/amp_err)**2)
dof_amp = len(f_exp) - len(popt_amp)
chi2red_amp = chi2_amp/dof_amp

# ================= FIT FASE =================

popt_phase, pcov_phase = curve_fit(
    model_phase, f_exp, phase_exp,
    p0=p0, sigma=phase_err,
    absolute_sigma=True, maxfev=30000
)

perr_phase = np.sqrt(np.diag(pcov_phase))

res_phase = phase_exp - model_phase(f_exp, *popt_phase)
chi2_phase = np.sum((res_phase/phase_err)**2)
dof_phase = len(f_exp) - len(popt_phase)
chi2red_phase = chi2_phase/dof_phase

# ================= MEDIA PESATA =================

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

# ================= PARAMETRI FINALI =================

n_modes = len(p_avg)//3
final_results = []

print("\n=========== PARAMETRI FINALI MEDIATI ===========\n")

for k in range(n_modes):

    wk  = p_avg[3*k]
    Rk  = p_avg[3*k+1]
    zk  = p_avg[3*k+2]

    wk_err = p_avg_err[3*k]
    Rk_err = p_avg_err[3*k+1]
    zk_err = p_avg_err[3*k+2]

    fk = wk/(2*np.pi)
    fk_err = wk_err/(2*np.pi)

    Qk = 1/(2*zk)
    Qk_err = zk_err/(2*zk**2)

    final_results.append((fk, fk_err, Rk, Rk_err, zk, zk_err, Qk, Qk_err))

    print(f"Modo {k+1}")
    print(f"  Frequenza f_k = {fk} ± {fk_err} Hz")
    print(f"  Residuo R_k   = {Rk} ± {Rk_err}")
    print(f"  Smorzamento ζ = {zk} ± {zk_err}")
    print(f"  Fattore Q     = {Qk} ± {Qk_err}\n")

# ================= SALVATAGGIO =================

if save:
    output_file = os.path.join(target_path, f"PARAMETRI_FINALI_{axis}.txt")
    with open(output_file, "w") as f:
        f.write("===== PARAMETRI FINALI MEDIATI =====\n\n")
        for k, res in enumerate(final_results):
            fk, fk_err, Rk, Rk_err, zk, zk_err, Qk, Qk_err = res
            f.write(f"Modo {k+1}\n")
            f.write(f"f_k (Hz) = {fk} , {fk_err}\n")
            f.write(f"R_k      = {Rk} , {Rk_err}\n")
            f.write(f"zeta     = {zk} , {zk_err}\n")
            f.write(f"Q_k      = {Qk} , {Qk_err}\n\n")

    print("File salvato in:", output_file)

# ================= GRAFICI =================

f_dense = np.linspace(f_exp.min(), f_exp.max(), 6000)

# ---- AMPIEZZA ----
fig1, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(10,8),
    constrained_layout=True
)

fig1.suptitle(
    f"{folder_name} - Fit Guadagno Asse {axis} "
    f"(χ² ridotto = {chi2red_amp:.3f})",
    fontweight='bold'
)

# --- dati
ax1.errorbar(
    f_exp, amp_exp,
    yerr=amp_err,
    fmt='o',
    markersize=4,
    color='black',          # <-- marker neri
    ecolor='black',
    capsize=4
)

ax1.plot(
    f_dense,
    model_amp(f_dense, *popt_amp),
    color='red',
    linewidth=2
)

ax1.set_yscale('log')
ax1.set_ylabel("Guadagno FRF")
ax1.grid(True, which='both', linewidth=0.8, alpha=0.3)

# --- residui (NO symlog)
ax2.errorbar(
    f_exp, res_amp,
    yerr=amp_err,
    fmt='o',
    markersize=4,
    color='black',
    ecolor='black',
    capsize=4
)

ax2.axhline(0, color='red', linestyle='--')

# limiti automatici con margine
ymax = np.max(np.abs(res_amp))
ax2.set_ylim(-1.2*ymax, 1.2*ymax)

ax2.set_ylabel("Residui")
ax2.set_xlabel("Frequenza [Hz]")
ax2.grid(True, linewidth=0.8, alpha=0.3)

# ---- FASE ----
fig2, (ax3, ax4) = plt.subplots(
    2, 1, sharex=True, figsize=(10,8),
    constrained_layout=True
)

fig2.suptitle(
    f"{folder_name} - Fit Fase Asse {axis} "
    f"(χ² ridotto = {chi2red_phase:.3f})",
    fontweight='bold'
)

ax3.errorbar(
    f_exp, phase_exp,
    yerr=phase_err,
    fmt='o',
    markersize=4,
    color='black',
    ecolor='black',
    capsize=4
)

ax3.plot(
    f_dense,
    model_phase(f_dense, *popt_phase),
    color='red',
    linewidth=2
)

ax3.set_ylabel("Fase [rad]")
ax3.grid(True, linewidth=0.8, alpha=0.3)

# residui fase
ax4.errorbar(
    f_exp, res_phase,
    yerr=phase_err,
    fmt='o',
    markersize=4,
    color='black',
    ecolor='black',
    capsize=4
)

ax4.axhline(0, color='red', linestyle='--')

ymax_phase = np.max(np.abs(res_phase))
ax4.set_ylim(-1.2*ymax_phase, 1.2*ymax_phase)

ax4.set_ylabel("Residui [rad]")
ax4.set_xlabel("Frequenza [Hz]")
ax4.grid(True, linewidth=0.8, alpha=0.3)

# ================= SALVATAGGIO IMMAGINI =================

if save:
    fig1.savefig(
        os.path.join(target_path, f"FIT_AMPIEZZA_{axis}.png"),
        dpi=300,
        bbox_inches='tight'
    )

    fig2.savefig(
        os.path.join(target_path, f"FIT_FASE_{axis}.png"),
        dpi=300,
        bbox_inches='tight'
    )

    print("Immagini salvate correttamente.")
    
plt.show()
