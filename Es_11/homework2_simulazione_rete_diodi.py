import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import numpy as np
import matplotlib.pyplot as plt
from utils.labplot import save_lab_figure

# =========================
# 1) Caricamento dati
# =========================
# Cambia questo path se necessario:
filename = "/home/marco/Desktop/Uni_anno3/TD/Es_11/logbook/dati_TINA.txt"

# genfromtxt gestisce bene header e separatori vari
data = np.genfromtxt(filename, comments="#", skip_header=1)

# Se il file avesse righe vuote, filtriamole
data = data[~np.isnan(data).any(axis=1)]

V_C = data[:, 0]
V_plus = data[:, 1]

# (opzionale ma consigliato) assicuriamoci che siano ordinati per V_C crescente
idx = np.argsort(V_C)
V_C = V_C[idx]
V_plus = V_plus[idx]

# =========================
# 2) Intersezioni: V_plus = 0.5 * V_C
# =========================
f = V_plus - 0.5 * V_C

x_int = []
y_int = []

for i in range(len(V_C) - 1):
    # Caso in cui un punto sia esattamente sulla retta (raro con float)
    if f[i] == 0.0:
        x_int.append(V_C[i])
        y_int.append(V_plus[i])
        continue

    # Cambio di segno => esiste uno zero nell'intervallo
    if f[i] * f[i + 1] < 0:
        x1, x2 = V_C[i], V_C[i + 1]
        y1, y2 = f[i], f[i + 1]

        # Interpolazione lineare dello zero di f tra (x1,y1) e (x2,y2)
        x0 = x1 - y1 * (x2 - x1) / (y2 - y1)
        x_int.append(x0)
        y_int.append(0.5 * x0)  # perché sul vincolo V_plus = 0.5*V_C

x_int = np.array(x_int)
y_int = np.array(y_int)

# =========================
# 3) Plot
# =========================
fig, ax = plt.subplots(figsize = (10,6))
ax.plot(V_C, V_plus, label=r"$V_+(V_C)$")
ax.plot(V_C, 0.5 * V_C, label=r"$V_+ = \frac{1}{2}V_C$")

if len(x_int) > 0:
    ax.scatter(x_int, y_int, label="Intersezioni")
    
ax.set_xlabel(r"$V_C$ [V]")
ax.set_ylabel(r"$V_+$ [V]")
plt.title(r"Intersezioni tra $V_+(V_C)$ e $V_+ = \frac{1}{2}V_C$")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
save_lab_figure(fig, ax, "homework2_simulazione_rete_diodi", mode='presentation')

# =========================
# 4) Stampa risultati
# =========================
print("Intersezioni trovate (V_C, V_+):")
if len(x_int) == 0:
    print("  Nessuna intersezione trovata nel range dei dati.")
else:
    for xi, yi in zip(x_int, y_int):
        print(f"  V_C = {xi:.8f} V,  V_+ = {yi:.8f} V")