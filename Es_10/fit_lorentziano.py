import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit


# -----------------------------
# Modello: Lorentziana + offset
# -----------------------------
def lorentzian(x, A, x0, gamma, y0):
    """
    y(x) = y0 + A * gamma^2 / ((x-x0)^2 + gamma^2)

    A     : ampiezza sopra baseline
    x0    : centro del picco
    gamma : HWHM (Half-Width at Half-Maximum)
    y0    : offset (baseline costante)
    """
    return y0 + A * (gamma**2) / ((x - x0)**2 + gamma**2)


# -----------------------------
# Caricamento dati
# -----------------------------
def load_csv_two_cols_or_named(path):
    """
    Carica un CSV con header.
    Se trova colonne nominate (es. Freq_Hz, Mean_PSD_X, ...), usa:
      x = prima colonna
      y = seconda colonna
    """
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names is None or len(data.dtype.names) < 2:
        raise ValueError("Il CSV deve avere almeno 2 colonne con header.")
    col_x = data.dtype.names[0]
    col_y = data.dtype.names[1]
    x = data[col_x].astype(float)
    y = data[col_y].astype(float)
    return x, y, col_x, col_y


# -----------------------------
# Estrazione regione del picco
# -----------------------------
def extract_peak_region(y, peak_idx, rel_height=0.98, margin_pts=10):
    """
    Usa peak_widths per stimare i bordi sinistro/destro del picco
    (intersezione a rel_height) e aggiunge un margine di punti per
    includere meglio le code.
    """
    w = peak_widths(y, [peak_idx], rel_height=rel_height)
    left_ip = w[2][0]   # indice float interpolato
    right_ip = w[3][0]  # indice float interpolato

    left = int(np.floor(left_ip)) - margin_pts
    right = int(np.ceil(right_ip)) + margin_pts

    left = max(left, 0)
    right = min(right, len(y) - 1)
    return left, right


# -----------------------------
# Fit singolo picco
# -----------------------------
def fit_single_peak_lorentzian(xw, yw):
    """
    Fit lorentziano su una finestra (xw, yw) con stime iniziali semplici.
    """
    # baseline: mediana di un po' di punti ai bordi
    k = max(5, len(yw) // 10)
    y0_guess = float(np.median(np.r_[yw[:k], yw[-k:]]))

    # centro: massimo locale nella finestra
    i_max = int(np.argmax(yw))
    x0_guess = float(xw[i_max])

    # ampiezza
    A_guess = float(max(yw[i_max] - y0_guess, 1e-12))

    # gamma: stima grezza
    gamma_guess = float(max((xw[-1] - xw[0]) / 10.0, 1e-12))

    p0 = [A_guess, x0_guess, gamma_guess, y0_guess]

    # bounds ragionevoli
    bounds = (
        [0.0, xw.min(), 1e-15, -np.inf],   # A>=0, gamma>0
        [np.inf, xw.max(), np.inf, np.inf]
    )

    popt, pcov = curve_fit(
        lorentzian, xw, yw,
        p0=p0, bounds=bounds,
        maxfev=20000
    )
    return popt, pcov


# -----------------------------
# Pipeline: trova picchi, filtra, fit, plot
# -----------------------------
def fit_lorentzians_on_peaks(
    x, y,
    peak_threshold=5e-3,
    prominence=None,
    rel_height=0.98,
    margin_pts=10,
    plot_each=True,
    plot_global=True
):
    """
    1) trova picchi (find_peaks)
    2) filtra per altezza > peak_threshold
    3) per ogni picco: estrai finestra (cima+code) e fai fit lorentziano
    4) plot punti+fit (per picco) e plot globale opzionale
    """
    # Trova picchi
    peaks, _ = find_peaks(y, prominence=prominence)

    # Filtra per altezza soglia
    peaks = [p for p in peaks if y[p] > peak_threshold]

    results = []

    # Plot globale: dati
    if plot_global:
        plt.figure()
        plt.plot(x, y, ".", ms=3, label="dati")

    for idx, p in enumerate(peaks, start=1):
        # Estrai regione del picco
        left, right = extract_peak_region(y, p, rel_height=rel_height, margin_pts=margin_pts)
        xw = x[left:right + 1]
        yw = y[left:right + 1]

        # Fit
        try:
            popt, pcov = fit_single_peak_lorentzian(xw, yw)
        except Exception as e:
            print(f"[WARN] Fit fallito sul picco #{idx} (index={p}, x={x[p]}): {e}")
            continue

        A, x0, gamma, y0 = popt
        fwhm = 2.0 * gamma

        results.append({
            "peak_number": idx,
            "peak_index": int(p),
            "x_peak_raw": float(x[p]),
            "A": float(A),
            "x0": float(x0),
            "gamma": float(gamma),
            "FWHM": float(fwhm),
            "y0": float(y0),
            "window_left": int(left),
            "window_right": int(right),
        })

        # Curve per il plot
        xf = np.linspace(xw.min(), xw.max(), 800)
        yf = lorentzian(xf, *popt)

        # Plot singolo picco
        if plot_each:
            plt.figure()
            plt.plot(xw, yw, "o", ms=4, label="punti del picco")
            plt.plot(xf, yf, "-", lw=2, label="fit lorentziano")
            plt.axvline(x0, ls="--", lw=1)
            plt.title(f"Picco #{idx} | x0={x0:.6g}  FWHM={fwhm:.6g}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True, alpha=0.3)
            plt.legend()

        # Plot globale: sovrapponi fit
        if plot_global:
            plt.plot(xf, yf, "-", lw=2, label=f"fit #{idx}")

    # Chiudi plot globali
    if plot_global:
        plt.title("Dati + fit lorentziani sui picchi selezionati")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.legend(ncols=2, fontsize=9)

    plt.show()
    return results


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # File (quello che mi hai caricato)
    path = "C:\\Users\\aless\\Desktop\\TD_ale\\TD\\Es_10\\AVERAGED_XY_RESONANCE.csv"  # se lo metti nella stessa cartella
    # In alternativa, metti il path completo:
    # path = "/mnt/data/AVERAGED_XY_RESONANCE.csv"

    x, y, col_x, col_y = load_csv_two_cols_or_named(path)
    print(f"Colonne usate: x='{col_x}', y='{col_y}'")

    results = fit_lorentzians_on_peaks(
        x, y,
        peak_threshold=5e-3,  # soglia richiesta
        prominence=None,      # se serve, prova ad es. 1e-4
        rel_height=0.98,      # vicino alla baseline (include code)
        margin_pts=10,        # margine per includere le code
        plot_each=True,       # plot per ogni picco
        plot_global=True      # plot globale con fit sovrapposti
    )

    print("\n=== RISULTATI FIT ===")
    for r in results:
        print(r)
