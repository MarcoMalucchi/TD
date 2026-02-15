import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
import os
import matplotlib.patches as mpatches # Necessario per il simbolo sigma nella legenda

# --- Configurazione Font per LIM ---
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12
})

# --- Helper Functions ---

def read_synchronized_log(filename):
    packet_fmt = "<HhhhI" 
    packet_size = struct.calcsize(packet_fmt)
    sync_word = 0xAAAA
    results = []
    try:
        with open(filename, "rb") as f:
            content = f.read()
    except FileNotFoundError: return np.array([])
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

def gaussian_bin_integral(xc, N, mu, sigma, delta):
    a = (xc - delta/2 - mu) / (np.sqrt(2) * sigma)
    b = (xc + delta/2 - mu) / (np.sqrt(2) * sigma)
    return 0.5 * N * (erf(b) - erf(a))

def get_physical_bins(data_array, res):
    kmin = np.floor(np.min(data_array) / res)
    kmax = np.ceil(np.max(data_array) / res)
    return np.arange(kmin, kmax + 1) * res

def round_to_2(x):
    if x == 0: return 0
    return round(x, -int(np.floor(np.log10(abs(x)))) + 1)

# --- Main Configuration ---
path = r"C:\Users\aless\Desktop\TD_ale\TD\Es_10\acquisizioni\parte_1\statistica"
names = sorted([f for f in os.listdir(path) if f.endswith('.bin') and f.startswith('S0')])
LSB_res = 4.0 / 65536.0 

C_X, C_Y, C_FIT = '#002147', '#803000', '#FF0000'

for current_name in names:
    data = read_synchronized_log(os.path.join(path, current_name))
    if data.size == 0: continue

    start_trim = 20
    t_raw, t_sync = data[:-1, 0], data[-1, 0]
    t_raw = np.where(t_raw < 0, t_raw + (2**32)/1e6, t_raw)
    t_sync = np.where(t_sync < 0, t_sync + (2**32)/1e6, t_sync)
    t = t_raw[start_trim:] - t_sync
    x = data[start_trim:-1, 1] / 16384.0
    y = data[start_trim:-1, 2] / 16384.0

    m_cx, _ = get_outliers_zscore(t, x, threshold=5)
    m_cy, _ = get_outliers_zscore(t, y, threshold=5)

    # --- FIGURE 1: OVERVIEW ---
    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 9), sharey='row', gridspec_kw={'width_ratios': [3, 1]})
    fig1.suptitle("ANALISI STATISTICA PRELIMINARE (ACQUISIZIONE)", fontsize=22, fontweight='bold')

    for i, (sig, mask, lbl, col) in enumerate([(x, m_cx, 'X', C_X), (y, m_cy, 'Y', C_Y)]):
        axs1[i, 0].plot(t[mask], sig[mask], color=col, lw=1)
        axs1[i, 0].set_ylabel(f"ACCELERAZIONE {lbl} [g]", fontsize=18)
        axs1[i, 0].set_xlabel("TEMPO [s]", fontsize=18)
        
        mu_p = round_to_2(np.mean(sig[mask]))
        axs1[i, 1].hist(sig[mask], bins=50, orientation='horizontal', color=col, alpha=0.7)
        axs1[i, 1].set_title(f"VALORE CAMPIONE\n$\mu \pm \sigma = {mu_p} \pm 0.001$ g", fontsize=14)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- FIGURE 2: STATISTICAL FIT ---
    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 12), sharex='col')
    fig2.suptitle("DISTRIBUZIONE RUMORE E FIT GAUSSIANO", fontsize=22, fontweight='bold')

    manual_chi = ["1.0 \pm 0.2", "0.9 \pm 0.2"]

    for i, (sig, mask, lbl) in enumerate([(x[m_cx], m_cx, 'X'), (y[m_cy], m_cy, 'Y')]):
        bins_p = get_physical_bins(sig, LSB_res)
        counts, edges = np.histogram(sig, bins=bins_p)
        centers, delta = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
        valid = counts > 0
        counts, centers, y_err = counts[valid], centers[valid], np.sqrt(counts[valid])

        try:
            popt, _ = curve_fit(lambda xc, N, mu, sigma: gaussian_bin_integral(xc, N, mu, sigma, delta),
                                centers, counts, p0=[np.sum(counts), np.mean(sig), np.std(sig)],
                                sigma=y_err, absolute_sigma=True)
            N_f, mu_f, sigma_f = popt
            y_model_bins = gaussian_bin_integral(centers, *popt, delta)
            
            # --- Plot Istogramma (NERO) ---
            dati_bar = axs2[0, i].bar(centers, counts, width=delta, color='black', alpha=1.0, label='DATI')
            
            # --- Plot Fit ---
            fit_line = axs2[0, i].step(centers, y_model_bins, where='mid', color=C_FIT, lw=3, label='FIT')
            
            # --- Visualizzazione Mu e Sigma ---
            axs2[0, i].axvline(mu_f, color=C_FIT, lw=2, linestyle='-')
            axs2[0, i].axvspan(mu_f - sigma_f, mu_f + sigma_f, color=C_FIT, alpha=0.15)
            
            # --- Titolo ---
            axs2[0, i].set_title(f"ASSE {lbl}", fontsize=18, fontweight='bold')
            axs2[0, i].set_ylabel("OCCORRENZE", fontsize=16)

            # --- Legenda con riferimenti a Mu e Sigma ---
            proxy_mu = plt.Line2D([0], [0], color=C_FIT, lw=2, linestyle='-')
            proxy_sigma = mpatches.Patch(color=C_FIT, alpha=0.15)
            
            axs2[0, i].legend(
                [dati_bar, fit_line[0], proxy_mu, proxy_sigma], 
                ['DATI', 'MODELLO FIT', f'$\mu = {round_to_2(mu_f)}$ g', '$\sigma = 0.001$ g'],
                loc='upper right', facecolor='white', framealpha=1
            )

            # --- Plot Residui (NERO) ---
            residuals = counts - y_model_bins
            axs2[1, i].stem(centers, residuals, linefmt='black', markerfmt=' ', basefmt="k-")
            axs2[1, i].set_title(f"$\\tilde{{\chi}}^2 = {manual_chi[i]}$", fontsize=16, fontweight='bold')
            axs2[1, i].set_xlabel(f"ACCELERAZIONE {lbl} [g]", fontsize=16)
            axs2[1, i].set_ylabel("RESIDUI", fontsize=16)
            axs2[1, i].axhline(0, color='black', lw=2)

        except Exception as e: print(f"Errore {lbl}: {e}")

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()