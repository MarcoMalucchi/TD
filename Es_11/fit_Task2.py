import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares

# =========================================================
# PATH (Windows) — i tuoi
# =========================================================
FILE_NEG = r"C:\Users\aless\Desktop\TD_ale\TD\Es_11\acquisizioni\spazzata_minimo_negativo.txt"
FILE_POS = r"C:\Users\aless\Desktop\TD_ale\TD\Es_11\acquisizioni\spazzata_minimo_positivo.txt"

# (Se vuoi testare nell'ambiente dove li hai già caricati)
# FILE_NEG = "/mnt/data/spazzata_minimo_negativo.txt"
# FILE_POS = "/mnt/data/spazzata_minimo_positivo.txt"

# =========================================================
# STILE "DA SLIDE": grande, bold, tick visibili, contrasto
# =========================================================
plt.rcParams.update({
    "figure.figsize": (14, 10),
    "font.size": 18,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

C_DATA = "black"
C_FIT  = "#00B5FF"   # azzurro acceso
C_ZERO = "#FFB000"   # arancione acceso

# =========================================================
# LETTURA: file unico con 5 colonne
# f, G_mean, G_std, phi_mean(rad), phi_std(rad)
# =========================================================
def load_sweep(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File non trovato: {p}")

    # i tuoi file: separati da virgola, righe commentate con '#'
    df = pd.read_csv(p, comment="#", header=None, sep=",", engine="python")

    if df.shape[1] < 5:
        raise ValueError(f"Attese >=5 colonne, trovate {df.shape[1]} in {p}")

    df = df.iloc[:, :5].copy()
    df.columns = ["f_Hz", "G_mean", "G_std", "phi_mean_rad", "phi_std_rad"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("f_Hz").reset_index(drop=True)

    # unwrap SOLO per visualizzare e come riferimento iniziale;
    # per i residui useremo differenza wrapped (robusta)
    df["phi_unwrapped_rad"] = np.unwrap(df["phi_mean_rad"].to_numpy())
    return df

# =========================================================
# MODELLO screenshot
# G(f)= A / sqrt((1-(f/f0)^2)^2 + (2*pi*tau*f)^2) + offs
# phi(f)= phi0 - atan2(2*pi*tau*f, 1-(f/f0)^2)
# =========================================================
def gain_model(f, A, f0, tau, offs):
    re = 1.0 - (f / f0)**2
    im = 2.0 * np.pi * tau * f
    return A / np.sqrt(re**2 + im**2) + offs

def phase_model(f, f0, tau, phi0):
    re = 1.0 - (f / f0)**2
    im = 2.0 * np.pi * tau * f
    return phi0 - np.arctan2(im, re)

def angle_diff(a, b):
    """Differenza angolare robusta in (-pi, pi]."""
    return np.angle(np.exp(1j*(a - b)))

# =========================================================
# INIZIAL GUESS (robusto)
# =========================================================
def initial_guess(f, G, phi_u):
    f = np.asarray(f)
    G = np.asarray(G)

    # offs: stima fondo (non negativa)
    offs0 = max(0.0, 0.5*np.min(G))

    # A: plateau low-f (mediana 20% più basso) - offs
    ord_idx = np.argsort(f)
    fs = f[ord_idx]
    Gs = G[ord_idx]
    low = fs <= np.percentile(fs, 20)
    if not np.any(low):
        low = np.arange(max(1, int(0.2*len(fs))))
    A0 = float(np.median(Gs[low]) - offs0)
    A0 = max(A0, 1e-9)

    # f0: al massimo del (G-offs)
    Gcorr = np.clip(G - offs0, 1e-12, None)
    f0_0 = float(f[np.argmax(Gcorr)])
    f0_0 = max(f0_0, 1e-6)

    # tau: guess grezzo da larghezza -3dB (solo per inizializzare)
    peak = float(np.max(Gcorr))
    target = peak/np.sqrt(2.0)
    ip = int(np.argmax(Gcorr))
    left = np.where(Gcorr[:ip] <= target)[0]
    right = np.where(Gcorr[ip:] <= target)[0]
    if len(left) and len(right):
        fL = float(f[left[-1]])
        fR = float(f[ip + right[0]])
        df = max(fR - fL, 1e-9)
        tau0 = max(1e-9, 1.0/(2.0*np.pi*max(df, 1.0)))
    else:
        tau0 = 1e-5

    # phi0: mediana della fase a bassa frequenza
    n0 = max(3, int(0.1*len(phi_u)))
    phi0 = float(np.median(phi_u[:n0]))

    return np.array([A0, f0_0, tau0, offs0, phi0], float)

# =========================================================
# FIT CONGIUNTO CORRETTO:
#  - gain in dB (con sigma propagata)
#  - fase con differenza ANGOLARE WRAPPED (no unwrap nei residui)
# =========================================================
def fit_gain_phase(df: pd.DataFrame):
    f = df["f_Hz"].to_numpy(float)
    G = df["G_mean"].to_numpy(float)
    sG = df["G_std"].to_numpy(float)
    phi = df["phi_mean_rad"].to_numpy(float)          # fase "raw"
    phi_u = df["phi_unwrapped_rad"].to_numpy(float)   # solo per guess/plot
    sphi = df["phi_std_rad"].to_numpy(float)

    # protezioni su std
    sG = np.where(sG > 0, sG, np.median(sG[sG > 0]) if np.any(sG > 0) else 1.0)
    sphi = np.where(sphi > 0, sphi, np.median(sphi[sphi > 0]) if np.any(sphi > 0) else 1.0)

    # guess
    x0 = np.array([
    1.1,        # A
    7340.0,     # f0 [Hz]
    1.6e-5,     # tau [s]
    0.0,        # offs
    2.8         # phi0 [rad]
])

    # BOUNDS per evitare f0 fuori banda (causa tipica di fit "low-pass")
    fmin, fmax = float(np.min(f)), float(np.max(f))
    # offs: metti range sensato; se sai che deve essere ~0, stringilo ancora
    lb = np.array([1e-12, 0.5*fmin, 1e-12, -0.5, -np.inf], float)
    ub = np.array([np.inf,  2.0*fmax, np.inf,  +0.5,  np.inf], float)

    # Precompute gain in dB data + sigma in dB
    G_safe = np.clip(G, 1e-18, None)
    G_dB = 20*np.log10(G_safe)
    sG_dB = (20/np.log(10.0))*(sG/G_safe)
    sG_dB = np.where(sG_dB > 0, sG_dB, 1.0)

    def residuals(x):
        A, f0, tau, offs, phi0 = x

        # modello
        Gm = gain_model(f, A, f0, tau, offs)
        phim = phase_model(f, f0, tau, phi0)

        # === residuo gain in dB ===
        Gm_safe = np.clip(Gm, 1e-18, None)
        Gm_dB = 20*np.log10(Gm_safe)
        rG = (Gm_dB - G_dB) / sG_dB

        # === residuo fase wrapped ===
        dphi = angle_diff(phim, phi)     # (-pi, pi]
        rP = dphi / sphi

        return np.concatenate([rG, rP])

    res = least_squares(residuals, x0, bounds=(lb, ub), method="trf", loss="huber")

    A, f0, tau, offs, phi0 = res.x

    # predizioni sui punti misurati
    Ghat = gain_model(f, A, f0, tau, offs)
    phat = phase_model(f, f0, tau, phi0)

    # chi2 coerente con residui definiti sopra (dB + fase wrapped)
    rr = residuals(res.x)
    chi2 = float(np.sum(rr**2))
    Nd = 2*len(f)
    Np = 5
    dof = Nd - Np
    chi2_red = chi2/dof if dof > 0 else np.nan

    return {
        "popt": res.x,
        "success": res.success,
        "message": res.message,
        "f": f,
        "G": G,
        "sG": sG,
        "phi": phi,
        "sphi": sphi,
        "Ghat": Ghat,
        "phat": phat,
        "G_dB": G_dB,
        "sG_dB": sG_dB,
        "chi2": chi2,
        "chi2_red": chi2_red,
        "dof": dof,
    }

# =========================================================
# PLOT: 2x2 -> (Gain+fit, Phase+fit, Resid gain, Resid phase)
# =========================================================
def plot_fit_and_residuals(res, title, out_pdf=None):
    f = res["f"]
    G = res["G"]
    phi = res["phi"]
    sphi = res["sphi"]
    A, f0, tau, offs, phi0 = res["popt"]

    # Smooth curves
    f_fit = np.logspace(np.log10(np.min(f)), np.log10(np.max(f)), 2000)
    G_fit = gain_model(f_fit, A, f0, tau, offs)
    phi_fit = phase_model(f_fit, f0, tau, phi0)

    # Gain in dB + sigma(dB)
    G_dB = res["G_dB"]
    sG_dB = res["sG_dB"]
    Ghat_dB = 20*np.log10(np.clip(res["Ghat"], 1e-18, None))
    G_fit_dB = 20*np.log10(np.clip(G_fit, 1e-18, None))

    # Residui
    rG_dB = G_dB - Ghat_dB
    rP = angle_diff(phi, res["phat"])  # wrapped residual in rad
    rP_deg = np.rad2deg(rP)
    sP_deg = np.rad2deg(sphi)

    # Phase plot in degrees (wrapped nicely around)
    # Visualizzazione: usiamo unwrap SOLO per la curva in figura, non per i residui.
    phi_plot = np.unwrap(phi)
    phat_plot = np.unwrap(res["phat"])
    phi_fit_plot = np.unwrap(phi_fit)

    phi_deg = np.rad2deg(phi_plot)
    phat_deg = np.rad2deg(phat_plot)
    phi_fit_deg = np.rad2deg(phi_fit_plot)

    # χ²
    chi2_red = res["chi2_red"]
    dof = res["dof"]

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.85], hspace=0.30, wspace=0.22)

    # --- Gain
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(f, G_dB, yerr=sG_dB, fmt="o", ms=5, capsize=3,
                 color=C_DATA, ecolor=C_DATA, label="Data")
    ax1.plot(f_fit, G_fit_dB, lw=3.2, color=C_FIT, label="Fit")
    ax1.set_xscale("log")
    ax1.grid(True, which="both", alpha=0.35)
    ax1.set_title(f"{title} — Gain\n" + r"$\chi^2_\nu$" + f" = {chi2_red:.3f} (dof={dof})")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Gain [dB]")
    ax1.legend(loc="best", frameon=True)
    ax1.tick_params(which="both", direction="in", length=9, width=2.0)

    # --- Phase
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(f, phi_deg, yerr=sP_deg, fmt="o", ms=5, capsize=3,
                 color=C_DATA, ecolor=C_DATA, label="Data")
    ax2.plot(f_fit, phi_fit_deg, lw=3.2, color=C_FIT, label="Fit")
    ax2.set_xscale("log")
    ax2.grid(True, which="both", alpha=0.35)
    ax2.set_title(f"{title} — Phase")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Phase [deg]")
    ax2.legend(loc="best", frameon=True)
    ax2.tick_params(which="both", direction="in", length=9, width=2.0)

    # --- Residuals gain (dB)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.errorbar(f, rG_dB, yerr=sG_dB, fmt="o", ms=5, capsize=3,
                 color=C_DATA, ecolor=C_DATA)
    ax3.axhline(0, lw=3.0, color=C_ZERO)
    ax3.set_xscale("log")
    ax3.grid(True, which="both", alpha=0.35)
    ax3.set_title(f"{title} — Residuals (gain)")
    ax3.set_xlabel("Frequency [Hz]")
    ax3.set_ylabel("Residual [dB]")
    ax3.tick_params(which="both", direction="in", length=9, width=2.0)

    # --- Residuals phase (deg)  [WRAPPED]
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.errorbar(f, rP_deg, yerr=sP_deg, fmt="o", ms=5, capsize=3,
                 color=C_DATA, ecolor=C_DATA)
    ax4.axhline(0, lw=3.0, color=C_ZERO)
    ax4.set_xscale("log")
    ax4.grid(True, which="both", alpha=0.35)
    ax4.set_title(f"{title} — Residuals (phase)")
    ax4.set_xlabel("Frequency [Hz]")
    ax4.set_ylabel("Residual [deg]")
    ax4.tick_params(which="both", direction="in", length=9, width=2.0)

    # Box parametri (ben leggibile)
    txt = (
        f"A = {A:.5g}\n"
        f"f0 = {f0:.5g} Hz\n"
        f"τ = {tau:.5g} s\n"
        f"offs = {offs:.5g}\n"
        f"φ0 = {phi0:.5g} rad"
    )
    fig.text(
        0.02, 0.02, txt,
        fontsize=15, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.95)
    )

    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")

    plt.show()

# =========================================================
# RUN: un file -> fit + plot
# =========================================================
def run_one(path, label):
    df = load_sweep(path)
    res = fit_gain_phase(df)

    print(f"\n=== {label} ===")
    print("success:", res["success"], "|", res["message"])
    A, f0, tau, offs, phi0 = res["popt"]
    print(f"chi2_red = {res['chi2_red']:.6f} (dof={res['dof']})")
    print(f"A={A:.6g}, f0={f0:.6g} Hz, tau={tau:.6g} s, offs={offs:.6g}, phi0={phi0:.6g} rad")

    plot_fit_and_residuals(res, title=label, out_pdf=f"{label.replace(' ', '_')}_fit_residuals.pdf")
    return res

# =========================================================
# ESECUZIONE SU ENTRAMBI I FILE
# =========================================================
res_neg = run_one(FILE_NEG, "Spazzata minimo negativo")
res_pos = run_one(FILE_POS, "Spazzata minimo positivo")