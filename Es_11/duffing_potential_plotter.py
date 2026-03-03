import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import numpy as np
import matplotlib.pyplot as plt
from utils.labplot import save_lab_figure, float_to_str

# -----------------------------
# Duffing potential definition
# -----------------------------
def duffing_potential(x, alpha, beta):
    return 0.5 * alpha * x**2 + 0.25 * beta * x**4


# -----------------------------
# Parameters (EDIT THESE)
# -----------------------------
alpha = -1.0   # try +1.0 for single-well
beta = 1.0     # must be > 0 for stability

# -----------------------------
# Domain
# -----------------------------
x = np.linspace(-2.5, 2.5, 1000)
V = duffing_potential(x, alpha, beta)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, V, color='red')
ax.axhline(0, linestyle='--', color='black')
ax.axvline(0, linestyle='--', color='black')

# Mark equilibrium points
if alpha < 0 and beta > 0:
    x_eq = np.sqrt(abs(alpha)/beta)
    ax.scatter([x_eq, -x_eq], 
                duffing_potential(np.array([x_eq, -x_eq]), alpha, beta))
elif alpha > 0:
    ax.scatter([0], [0])

plt.title(f"Duffing Potential: alpha = {alpha}, beta = {beta}")
ax.set_xlabel("x")
ax.set_ylabel("V(x)")
ax.grid(True)
plt.show()
save_lab_figure(fig, ax, f"duffing_potential_alpha_{float_to_str(alpha, 3)}_beta_{float_to_str(beta, 3)}")