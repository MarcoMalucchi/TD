import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# --- GLOBAL PLOT STYLE FOR BEAMER ---
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


# --- CONFIGURATION ---
root_path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/'
plt.close('all')
save = True  # Set to False to disable all disk writing
unit_label = r"$\mathrm{g^2/Hz}$"

def single_lorentzian(f, f0, A, gamma):
    return A / (1 + ((f - f0) / gamma)**2)

def multi_lorentzian(f, *params):
    res = np.zeros_like(f)
    for i in range(0, len(params), 3):
        res += single_lorentzian(f, params[i], params[i+1], params[i+2])
    return res

# --- BATCH PROCESSING ---
subdirs = [os.path.join(root_path, d) for d in os.listdir(root_path) 
           if os.path.isdir(os.path.join(root_path, d, 'FFT'))]

all_modes = []   # Will contain dictionaries

for folder in subdirs:
    # 1. Validation: Check for exactly 10 .bin files in the parent folder
    bin_files = [f for f in os.listdir(folder) if f.endswith('.bin')]
    if len(bin_files) != 10:
        print(f"Skipping {os.path.basename(folder)}: Found {len(bin_files)} .bin files.")
        continue

    path_fft = os.path.join(folder, 'FFT')
    csv_path = os.path.join(path_fft, "AVERAGED_XY_RESONANCE.csv")
    if not os.path.exists(csv_path): continue

    data_all = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    # Tasks: (Column Mean, Column Std, Axis Name)
    tasks = [(1, 2, 'X'), (3, 4, 'Y')]

    for psd_idx, sig_idx, axis_name in tasks:
        print(f"\n--- Analyzing {os.path.basename(folder)} | Axis {axis_name} ---")
        
        mask = (data_all[:, 0] >= 0.5) & (data_all[:, 0] <= 20)
        f_data = data_all[mask, 0]
        psd_data = data_all[mask, psd_idx]
        sigma_data = data_all[mask, sig_idx]

        # 2. Peak Selection with Buffer Zone
        base_threshold = 5e-3
        init_p, _ = find_peaks(psd_data, height=base_threshold, distance=10)
        
        if len(init_p) > 0:
            buffer_limit = base_threshold + (0.05 * (np.max(psd_data) - base_threshold))
            peaks = [p for p in init_p if (f_data[p] >= 12.0 or psd_data[p] > buffer_limit)]
            peaks = np.array(peaks)
        else:
            peaks, buffer_limit = np.array([]), base_threshold

        # 3. Fit Calculation
        guesses = []
        for p in peaks:
            guesses.extend([f_data[p], psd_data[p], 0.02 * f_data[p]])

        if len(guesses) > 0:
            bounds_l, bounds_u = [], []
            for i in range(len(peaks)):
                f0_g, A_g = guesses[i*3], guesses[i*3+1]
                bounds_l.extend([f0_g - 0.5, A_g * 0.5, 0.001])
                bounds_u.extend([f0_g + 0.5, A_g * 2.0, 5.0])
            
            try:
                popt, pcov = curve_fit(multi_lorentzian, f_data, psd_data, p0=guesses, 
                                       sigma=sigma_data, absolute_sigma=True, 
                                       bounds=(bounds_l, bounds_u), maxfev=50000)
                perr = np.sqrt(np.diag(pcov))
                fit_curve = multi_lorentzian(f_data, *popt)
                residuals = psd_data - fit_curve

                # --- 4. Parameter Results Construction ---
                output = [f"Analysis: {os.path.basename(folder)} | Axis: {axis_name}", "="*95]
                header = f"{'MODE':<5} | {'f0 [Hz]':<15} | {'Gamma [Hz]':<15} | {'Height':<15} | {'Zeta':<15} | {'Q Factor'}"
                output.append(header); output.append("-" * 95)

                for i in range(0, len(popt), 3):
                    f0, A, gam = popt[i], popt[i+1], popt[i+2]
                    sf0, sA, sgam = perr[i], perr[i+1], perr[i+2]

                    all_modes.append({
                        'folder': os.path.basename(folder),
                        'axis': axis_name,
                        'f0': f0,
                        'sf0': sf0,
                        'gamma': gam,
                        'sgamma': sgam,
                        'A': A,
                        'sA': sA
                    })

                    # Error Propagation
                    zeta = gam / f0
                    szeta = np.sqrt((sgam/f0)**2 + ((-gam/f0**2)*sf0)**2)
                    q = 1 / (2 * zeta)
                    sq = np.sqrt((sf0/(2*gam))**2 + ((-f0/(2*gam**2))*sgam)**2)

                    line = f"{i//3+1:<5} | {f0:.4f}±{sf0:.4f} | {gam:.4f}±{sgam:.4f} | {A:.4e}±{sA:.4e} | {zeta:.5f}±{szeta:.5f} | {q:.2f}±{sq:.2f}"
                    output.append(line)

                dof = len(f_data) - len(popt)
                red_chi = np.sum((residuals / sigma_data)**2) / dof
                output.append("-" * 95)
                output.append(f"Reduced Chi-Square: {red_chi:.4f} (Expected Std Dev: {np.sqrt(2/dof):.4f})")
                plot_title = f"{os.path.basename(folder)} | Asse {axis_name} | $\\chi^2_\\nu$ = {red_chi:.3f}"

                print("\n".join(output))

                # --- 5. Generate Separated Figures ---
                # Figure A: Fit + Symlog Residuals
                fig_comb, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                fig_comb.suptitle(plot_title, fontweight='bold')
                ax1.semilogy(f_data, psd_data, color='black', marker='o', markersize=4, alpha=0.3, label='Dati')
                ax1.semilogy(f_data, fit_curve, 'r-', linewidth=2, label='Fit lorentziano')
                ax1.axhspan(base_threshold, buffer_limit, color=(0.35, 0.35, 0.35), alpha=0.3, label='Fascia di buffer') 
                ax1.axvline(12, linestyle=':', linewidth=2, color='blue', alpha=0.5, label='Soglia 12 Hz')
                ax1.set_ylabel(f"PSD [{unit_label}]"); ax1.grid(True, which="both", linewidth=0.8, alpha=0.3); ax1.legend()
                
                ax2.set_yscale('symlog', linthresh=1e-4)
                ax2.errorbar(f_data, residuals, yerr=sigma_data, fmt='o', markersize=4, color='black', ecolor='gray', alpha=0.7)
                ax2.axhline(0, color='red', linestyle='--'); ax2.set_ylabel(f"Res. [{unit_label}]"); ax2.grid(True, which="both", linewidth=0.8, alpha=0.3)
                ax2.set_xlabel("Frequenza [Hz]")

                
                # Figure B: Linear Residuals
                fig_lin = plt.figure(figsize=(10, 5))
                plt.errorbar(f_data, residuals, yerr=sigma_data, fmt='o', markersize=4, color='black', ecolor='black', elinewidth=1.2, capsize=4, capthick=1.2, alpha=0.7)
                plt.axhline(0, color='red', linestyle='--')
                plt.ylabel(f"Residui [{unit_label}]"); plt.xlabel("Frequenza [Hz]"); plt.grid(True, alpha=0.3)
                plt.grid(True, which="both", linewidth=0.8, alpha=0.3)
                plt.title(plot_title, fontweight='bold')

                # --- 6. SAVER BLOCK (Separate Folders) ---
                if save:
                    # Logic: fit_results / X / or / Y /
                    axis_dir = os.path.join(folder, 'fit_results_pres', axis_name)
                    if not os.path.exists(axis_dir): os.makedirs(axis_dir)
                    
                    # Save summary.txt inside the axis folder
                    with open(os.path.join(axis_dir, 'summary_pres.txt'), 'w') as f:
                        f.write("\n".join(output))
                    
                    # Save both plots inside the axis folder
                    fig_comb.savefig(os.path.join(axis_dir, 'fit_combined_pres.png'), dpi=300)
                    fig_lin.savefig(os.path.join(axis_dir, 'linear_residuals_pres.png'), dpi=300)
                    print(f"Saved axis {axis_name} results to: {axis_dir}")

                plt.show()

            except Exception as e:
                print(f"Fit failed for axis {axis_name} in {folder}: {e}")


# ==========================================================
# GLOBAL WEIGHTED AVERAGE OF MODES
# ==========================================================

if len(all_modes) == 0:
    print("No fitted modes found. Skipping global averaging.")
    exit()

FREQ_TOL = 0.2  # Hz tolerance

def weighted_mean(values, errors):
    weights = 1.0 / (errors**2)
    mean = np.sum(values * weights) / np.sum(weights)
    error = np.sqrt(1.0 / np.sum(weights))
    return mean, error


# ---- Step 1: Extract and sort ----
frequencies = np.array([m['f0'] for m in all_modes])
sorted_indices = np.argsort(frequencies)
sorted_modes = [all_modes[i] for i in sorted_indices]

# ---- Step 2: Group by proximity ----
groups = []
current_group = [sorted_modes[0]]

for i in range(1, len(sorted_modes)):
    if abs(sorted_modes[i]['f0'] - current_group[-1]['f0']) < FREQ_TOL:
        current_group.append(sorted_modes[i])
    else:
        groups.append(current_group)
        current_group = [sorted_modes[i]]

groups.append(current_group)


# ---- Step 3: Compute weighted averages ----
global_output = []
global_output.append("GLOBAL WEIGHTED AVERAGES")
global_output.append("="*60)

for idx, group in enumerate(groups):

    f_vals = np.array([m['f0'] for m in group])
    sf_vals = np.array([m['sf0'] for m in group])

    g_vals = np.array([m['gamma'] for m in group])
    sg_vals = np.array([m['sgamma'] for m in group])

    A_vals = np.array([m['A'] for m in group])
    sA_vals = np.array([m['sA'] for m in group])

    # ---- Weighted means ----
    f_mean, sf_mean = weighted_mean(f_vals, sf_vals)
    g_mean, sg_mean = weighted_mean(g_vals, sg_vals)
    A_mean, sA_mean = weighted_mean(A_vals, sA_vals)

    # ---- Derived quantities ----
    zeta = g_mean / f_mean
    szeta = np.sqrt(
        (sg_mean / f_mean)**2 +
        ((-g_mean / f_mean**2) * sf_mean)**2
    )

    Q = f_mean / (2 * g_mean)
    sQ = np.sqrt(
        (sf_mean / (2 * g_mean))**2 +
        ((-f_mean / (2 * g_mean**2)) * sg_mean)**2
    )

    # ---- DEBUG PRINT ----
    print("\n====================================================")
    print(f"MODE GROUP {idx+1}")
    print("Frequencies used:", f_vals)
    print("Gammas used:", g_vals)
    print("Amplitudes used:", A_vals)
    print("-> Weighted f0 =", f_mean)
    print("-> Weighted gamma =", g_mean)
    print("-> Weighted A =", A_mean)
    print("-> Derived zeta =", zeta)
    print("-> Derived Q =", Q)
    print("====================================================")

    # ---- Save raw values (no rounding) ----
    global_output.append(f"\nMODE GROUP {idx+1}")
    global_output.append(f"N_contributions = {len(group)}")

    global_output.append(f"f0_avg = {f_mean}")
    global_output.append(f"sf0_avg = {sf_mean}")

    global_output.append(f"gamma_avg = {g_mean}")
    global_output.append(f"sgamma_avg = {sg_mean}")

    global_output.append(f"A_avg = {A_mean}")
    global_output.append(f"sA_avg = {sA_mean}")

    global_output.append(f"zeta_avg = {zeta}")
    global_output.append(f"szeta_avg = {szeta}")

    global_output.append(f"Q_avg = {Q}")
    global_output.append(f"sQ_avg = {sQ}")


# ---- Step 4: Save to file ----
if save:
    output_path = os.path.join(root_path, "global_weighted_averages_pres.txt")
    with open(output_path, 'w') as f:
        f.write("\n".join(global_output))

    print("\nSaved global weighted averages to:", output_path)
