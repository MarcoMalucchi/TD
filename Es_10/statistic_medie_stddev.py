import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from scipy.special import erf

# --- Helper Functions (Invariate) ---

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

def gaussian_bin_integral(xc, N, mu, sigma, delta):
    a = (xc - delta/2 - mu) / (np.sqrt(2) * sigma)
    b = (xc + delta/2 - mu) / (np.sqrt(2) * sigma)
    return 0.5 * N * (erf(b) - erf(a))

def get_physical_bins(data_array, res):
    kmin = np.floor(np.min(data_array) / res)
    kmax = np.ceil(np.max(data_array) / res)
    return np.arange(kmin, kmax + 1) * res

def weighted_mean(values, errors):
    weights = 1.0 / (errors**2)
    w_mean = np.sum(weights * values) / np.sum(weights)
    return w_mean

def weighted_std(errors):
    weights = 1.0 / (errors**2)
    w_std = 1/np.sum(weights)
    return w_std

# --- Main Loop ---

path = "/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/statistica/"
names = sorted([f for f in os.listdir(path) if f.endswith('.bin') and f.startswith('S0')])
LSB_res = 4.0 / 65536.0

all_stats = []

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

    m_cx, m_ox = get_outliers_zscore(t, x, threshold=5)
    m_cy, m_oy = get_outliers_zscore(t, y, threshold=5)

    current_file_results = {}

    for i, (sig, mask, lbl, col) in enumerate([(x[m_cx], m_cx, 'X', 'blue'), (y[m_cy], m_cy, 'Y', 'green')]):
        bins_p = get_physical_bins(sig, LSB_res)
        counts, edges = np.histogram(sig, bins=bins_p)
        centers = (edges[:-1] + edges[1:]) / 2
        delta = edges[1] - edges[0]
        valid = counts > 0
        counts, centers = counts[valid], centers[valid]
        y_err = np.sqrt(counts)

        try:
            popt, pcov = curve_fit(
                lambda x, N, mu, sigma: gaussian_bin_integral(x, N, mu, sigma, delta),
                centers, counts,
                p0=[np.sum(counts), np.mean(sig), np.std(sig)],
                sigma=y_err, absolute_sigma=True
            )
            perr = np.sqrt(np.diag(pcov))
            current_file_results[f'{lbl}_mu'] = popt[1]
            current_file_results[f'{lbl}_mu_err'] = perr[1]
            current_file_results[f'{lbl}_sigma'] = popt[2]
            current_file_results[f'{lbl}_sigma_err'] = perr[2]

        except Exception as e:
            print(f"Fit failed for {lbl}: {e}")

    if current_file_results:
        all_stats.append(current_file_results)

# ---------------------------------------------------------
# CALCOLO E PLOT CONCLUSIVI (Modificati per visibilità)
# ---------------------------------------------------------

if all_stats:
    final_values = {}
    final_errors = {}
    for axis in ['X', 'Y']:
        for param in ['mu', 'sigma']:
            vals = np.array([f[f'{axis}_{param}'] for f in all_stats])
            #print(f"{axis}_{param}: {vals}")
            errs = np.array([f[f'{axis}_{param}_err'] for f in all_stats])
            #print(f"{axis}_{param}_err: {errs}")
            final_values[f'{axis}_{param}'] = weighted_mean(vals, errs)
            final_errors[f'{axis}_{param}'] = np.sqrt(weighted_std(errs))

    print("\n--- RISULTATI FINALI (MEDIA PESATA) ---")
    print(f"ASSE X: Media = {final_values['X_mu']:.6e} +/- {final_errors['X_mu']:.6e} [g], DevStd = {final_values['X_sigma']:.6e} +/- {final_errors['X_sigma']:.6e} [g]")
    print(f"ASSE Y: Media = {final_values['Y_mu']:.6e} +/- {final_errors['Y_mu']:.6e} [g], DevStd = {final_values['Y_sigma']:.6e} +/- {final_errors['Y_sigma']:.6e} [g]")

    # FIGURA MEDIE: Subplot separati per X e Y per vedere le oscillazioni
    fig_mu, axs_mu = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig_mu.suptitle("Andamento delle Medie ($\mu$) nei vari file")
    
    for i, axis in enumerate(['X', 'Y']):
        color = 'blue' if axis == 'X' else 'green'
        vals = [f[f'{axis}_mu'] for f in all_stats]
        axs_mu[i].plot(vals, 'o-', color=color, label=f'Media file {axis}')
        axs_mu[i].axhline(final_values[f'{axis}_mu'], color='red', linestyle='--', label='Media Pesata Globale')
        axs_mu[i].set_ylabel(f"$\mu_{axis}$ [g]")
        axs_mu[i].legend()
        axs_mu[i].grid(True, alpha=0.3)

    axs_mu[1].set_xlabel("Indice Acquisizione (File)")
    plt.tight_layout()

    # FIGURA DEVIAZIONI STANDARD: Subplot separati
    fig_sig, axs_sig = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig_sig.suptitle("Andamento delle Deviazioni Standard ($\sigma$) nei vari file")

    for i, axis in enumerate(['X', 'Y']):
        color = 'blue' if axis == 'X' else 'green'
        vals = [f[f'{axis}_sigma'] for f in all_stats]
        axs_sig[i].plot(vals, 's-', color=color, label=f'DevStd file {axis}')
        axs_sig[i].axhline(final_values[f'{axis}_sigma'], color='red', linestyle='--', label='DevStd Pesata Globale')
        axs_sig[i].set_ylabel(f"$\sigma_{axis}$ [g]")
        axs_sig[i].legend()
        axs_sig[i].grid(True, alpha=0.3)

    axs_sig[1].set_xlabel("Indice Acquisizione (File)")
    plt.tight_layout()

    plt.show()