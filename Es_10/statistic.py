import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

#======
# NOTA.
#======

# Qui si assume che il rumore sia gaussiano, ma prendendo i bin dell'istrogramma pari alla risoluzione data dal ADC del sensore fare direttamente un fit gaussinao 
#sulle occorrenze in ogni bin è sbagliato. Infatti, assumendo il rumore gaussiano, bisogno ulteriormente chiedersi quale sia la probabilità che il rumore caschi in
#un certo bin. Questa probabilità è uniforme sull'intevallo del bin, cioè dibende dalla risoluzione dell'ADC.
# Di conseguenza il problema che si sta caratterizzando statisticamente è: qual è la probabilità che un campione gaussiano caschi in un certo bin?
# La risposta è data dalla convoluzione del rumore gaussiano con la distribuzione uniforme sull'intervallo del bin. Nel concreto integro la distribuzione gaussina
#sull'intervallo del bin.

# --- Helper Functions ---

def read_synchronized_log(filename):
    packet_fmt = "<HhhhI" 
    packet_size = struct.calcsize(packet_fmt)
    sync_word = 0xAAAA
    results = []
    try:
        with open(filename, "rb") as f:
            content = f.read()
    except FileNotFoundError:
        return np.array([])
    i = 0
    start_found = False
    while i <= len(content) - (packet_size * 2):
        header, = struct.unpack_from("<H", content, i)
        if header == sync_word:
            next_header, = struct.unpack_from("<H", content, i + packet_size)
            if next_header == sync_word:
                start_found = True
                break
        i += 1 
    if not start_found: return np.array([])
    while i <= len(content) - packet_size:
        header, = struct.unpack_from("<H", content, i)
        if header == sync_word:
            _, x, y, z, timestamp = struct.unpack_from(packet_fmt, content, i)
            results.append([timestamp / 1e6, x, y, z])
            i += packet_size 
        else: i += 1
    return np.array(results)

def get_outliers_zscore(time_arr, data_arr, threshold=5):
    mu, sigma = np.mean(data_arr), np.std(data_arr)
    z_scores = np.abs((data_arr - mu) / sigma)
    return z_scores < threshold, z_scores >= threshold

# def gaussian(x, amp, mu, sigma):
#     return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

from scipy.special import erf

def gaussian_bin_integral(xc, N, mu, sigma, delta):
    a = (xc - delta/2 - mu) / (np.sqrt(2) * sigma)
    b = (xc + delta/2 - mu) / (np.sqrt(2) * sigma)
    return 0.5 * N * (erf(b) - erf(a))


def get_physical_bins(data_array, res):
    kmin = np.floor(np.min(data_array) / res)
    kmax = np.ceil(np.max(data_array) / res)
    return np.arange(kmin, kmax + 1) * res


# --- Main Loop ---

path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/statistica/'
names = sorted([f for f in os.listdir(path) if f.endswith('.bin') and f.startswith('S0')])
LSB_res = 4.0 / 65536.0

p_best_fit = np.array([])

for current_name in names:
    data = read_synchronized_log(os.path.join(path, current_name))
    if data.size == 0: continue

    # Extraction & Trimming
    start_trim = 20
    t_raw, t_sync = data[:-1, 0], data[-1, 0]
    t_raw = np.where(t_raw < 0, t_raw + (2**32)/1e6, t_raw)
    t_sync = np.where(t_sync < 0, t_sync + (2**32)/1e6, t_sync)
    t = t_raw[start_trim:] - t_sync
    x = data[start_trim:-1, 1] / 16384.0
    y = data[start_trim:-1, 2] / 16384.0

    m_cx, m_ox = get_outliers_zscore(t, x, threshold=5)
    m_cy, m_oy = get_outliers_zscore(t, y, threshold=5)

    # ---------------------------------------------------------
    # FIGURE 1: OVERVIEW (Bins parallel to Acceleration axis)
    # ---------------------------------------------------------
    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 8), sharey='row', gridspec_kw={'width_ratios': [3, 1]})
    fig1.suptitle(f"Overview Analysis: {current_name}")

    for i, (sig, mask, lbl, col) in enumerate([(x, m_cx, 'X', 'blue'), (y, m_cy, 'Y', 'green')]):
        axs1[i, 0].plot(t[mask], sig[mask], color=col, lw=0.5, label='Clean')
        axs1[i, 0].set_ylabel(f"{lbl}-Axis Acceleration [g]")
        axs1[i, 0].set_xlabel("Time [s]")
        axs1[i, 1].hist(sig[mask], bins=50, orientation='horizontal', color=col, alpha=0.6)
        axs1[i, 1].set_title(f"μ={np.mean(sig[mask]):.4f}, σ={np.std(sig[mask]):.4e}")

    fig1.tight_layout()

    # ---------------------------------------------------------
    # FIGURE 2: STATISTICAL FIT (Normal Orientation)
    # ---------------------------------------------------------
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10), sharex='col')
    fig2.suptitle(f"Detailed Statistical Fit (Normal Orientation): {current_name}")

    for i, (sig, mask, lbl, col) in enumerate([(x[m_cx], m_cx, 'X', 'blue'), (y[m_cy], m_cy, 'Y', 'green')]):
        bins_p = get_physical_bins(sig, LSB_res)
        counts, edges = np.histogram(sig, bins=bins_p)
        centers = (edges[:-1] + edges[1:]) / 2
        delta = edges[1] - edges[0]

        valid = counts > 0
        counts = counts[valid]
        centers = centers[valid]
        y_err = np.sqrt(counts)

        try:
            # 3. The Fit
            popt, pcov = curve_fit(
                lambda x, N, mu, sigma: gaussian_bin_integral(x, N, mu, sigma, delta),
                centers,
                counts,
                p0=[np.sum(counts), np.mean(sig), np.std(sig)],
                sigma=y_err,
                absolute_sigma=True
            )
            N_f, mu_f, sigma_f = popt
            perr = np.sqrt(np.diag(pcov))
            
            # 4. Generate Fit Line for Plotting
            #x_fit = np.linspace(centers[0], centers[-1], 500)
            #y_fit_line = gaussian(x_fit, *popt)     # High resolution for smooth curve
            y_model_bins = gaussian_bin_integral(centers, *popt, delta) # At bin centers for residuals
            
            # 5. Top Plot: Histogram + Fit
            axs2[0, i].bar(centers, counts, width=delta, color=col, alpha=0.3, label='Data')
            axs2[0, i].bar(centers, y_model_bins, width=delta, edgecolor='red', fill=False, lw=2, label='Model')
            axs2[0, i].set_title(f"{lbl} Fit\nμ={mu_f:.5f}, σ={sigma_f:.5e}")

            # 6. Bottom Plot: Residuals
            residuals = counts - y_model_bins
            axs2[1, i].stem(centers, residuals, linefmt=col, markerfmt=' ', basefmt="k-")
            
            # Chi-Square Calculation
            chi_sq = np.sum((residuals**2) / y_err**2)
            dof = len(counts) - 3
            #print(np.sum(counts), N_f)
            axs2[1, i].set_title(f"Residuals (χ²/DoF: {chi_sq/dof:.2f} +/- {np.sqrt(2/dof):.2f})")
            axs2[1, i].axhline(0, color='black', lw=1, linestyle='--')

        except Exception as e:
            print(f"Fit failed for {lbl}: {e}")

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()