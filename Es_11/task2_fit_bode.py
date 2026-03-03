import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# PATH DATI
FILE_NEG = r"C:\Users\aless\Desktop\TD_ale\TD\Es_11\acquisizioni\spazzata_minimo_negativo.txt"
FILE_POS = r"C:\Users\aless\Desktop\TD_ale\TD\Es_11\acquisizioni\spazzata_minimo_positivo.txt"

# PARAMETRI INIZIALI
# ordine: [f0, tau, offs, cost]
p0 = np.array([7.34e3, 1e-5, 3e-2, 0.98], dtype=float)

# STILE GRAFICO
# =========================================================
plt.rcParams.update({
    "font.size": 16,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

DATA_COLOR = "black"
FIT_COLOR  = "#00B5FF"   # azzurro
RES_COLOR  = "#7B002C"   # bordeaux
ZERO_COLOR = "black"

# LETTURA FILE:
# colonne: f, gain_mean, gain_std, phase_mean(rad), phase_std(rad)
# (fase qui NON la fittiamo)
def load_bode_onefile(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File non trovato: {p}")

    df = pd.read_csv(p, comment="#", header=None, sep=",", engine="python")
    if df.shape[1] < 5:
        raise ValueError(f"Attese >= 5 colonne nel file {p}, trovate {df.shape[1]}")

    df = df.iloc[:, :5].copy()
    df.columns = ["f_Hz", "gain_mean", "gain_std", "phi_mean", "phi_std"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("f_Hz").reset_index(drop=True)

    f = df["f_Hz"].to_numpy(float)
    gain_mean = df["gain_mean"].to_numpy(float)
    gain_std  = df["gain_std"].to_numpy(float)

    # protezione su std
    eps = 1e-30
    gain_std = np.where(gain_std > 0, gain_std, eps)
    return f, gain_mean, gain_std


# MODELLO:
# G(p,x) = p3 + p4 / sqrt( (1-(x/p1)^2)^2 + (p2*x*2*pi)^2 )
# con p = [f0, tau, offs, cost]
def G_model(f, f0, tau, offs, cost):
    re = 1.0 - (f / f0)**2
    im = (2.0 * np.pi * tau * f)
    return offs + cost / np.sqrt(re**2 + im**2)


def fit_gain(f, gain_mean, gain_std, p0):
    # curve_fit usa sigma come deviazione standard dei dati
    popt, pcov = curve_fit(
        G_model,
        f, gain_mean,
        p0=p0,
        sigma=gain_std,
        absolute_sigma=True,   # IMPORTANTISSIMO: pcov in unità corrette
        maxfev=20000
    )
    perr = np.sqrt(np.diag(pcov))

    # chi^2 e chi^2 ridotto (come il tuo chi2n)
    resid = gain_mean - G_model(f, *popt)
    chi2 = np.sum((resid / gain_std)**2)
    dof = len(f) - len(popt)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    return popt, perr, resid, chi2_red, dof


def plot_gain_and_residuals(f, gain_mean, gain_std, popt, resid, chi2_red, dof, title, out_pdf):
    # curva fit sui punti (come MATLAB) + anche versione smooth per estetica
    f_smooth = np.logspace(np.log10(f.min()), np.log10(f.max()), 1500)
    g_smooth = G_model(f_smooth, *popt)

    g_fit_pts = G_model(f, *popt)
    resn = resid / gain_std

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(title, fontweight="bold")

    # --- Gain (log-log)
    ax = axs[0]
    ax.errorbar(f, gain_mean, yerr=gain_std, fmt=".", color=DATA_COLOR, ecolor=DATA_COLOR,
                capsize=2, label="Punti sperimentali")
    ax.plot(f_smooth, g_smooth, "-", lw=2.8, color=FIT_COLOR, label="Fit")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Guadagno")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="lower left", framealpha=0.95)

    ax.set_title(rf"$\chi^2_\nu = {chi2_red:.3f}$")

    # --- Residui normalizzati
    ax = axs[1]
    ax.errorbar(f, resn, yerr=np.ones_like(resn), fmt=".", color=RES_COLOR, ecolor=RES_COLOR, capsize=2)
    ax.axhline(0.0, color=ZERO_COLOR, lw=2.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Frequenza [Hz]")
    ax.set_ylabel("Residui normalizzati")
    ax.grid(True, which="both", alpha=0.35)

    # tick/spine più visibili
    for ax in axs:
        ax.tick_params(which="both", direction="in", length=8, width=2.0)
        for sp in ax.spines.values():
            sp.set_linewidth(2.0)


    fig.savefig(out_pdf, bbox_inches="tight")  # PDF vettoriale
    plt.show()


def run_one(path, label, out_pdf):
    f, gm, sgm = load_bode_onefile(path)
    popt, perr, resid, chi2_red, dof = fit_gain(f, gm, sgm, p0)

    print(f"\n\nFIT SU {label.upper()}:")
    names = ["f0", "tau", "offs", "cost"]
    for n, v, e in zip(names, popt, perr):
        print(f"{n} = {v:.8g} ± {e:.3g}")
    print(f"chi2n = {chi2_red:.8g}")

    plot_gain_and_residuals(
        f, gm, sgm,
        popt, resid, chi2_red, dof,
        title=f"Plot di Bode sul punto di equilibrio {label}",
        out_pdf=out_pdf
    )


if __name__ == "__main__":
    run_one(FILE_NEG, r"a -$2V^*$ (minimo negativo)", "bodeeqnegativosecondo01.pdf")
    run_one(FILE_POS, r"a +$2V^*$ (minimo positivo)", "bodeeqpositivosecondo01.pdf")