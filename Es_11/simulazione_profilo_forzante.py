import numpy as np
import matplotlib.pyplot as plt
import os

# ======================
# User option
# ======================
path = '/home/marco/Desktop/Uni_anno3/TD/Es_11/'
save = True   # <-- set False to disable saving

# ======================
# Define function
# ======================
x = np.linspace(-2, 2, 1000)

F = np.piecewise(
    x,
    [x < -0.5, (x >= -0.5) & (x <= 0.5), x > 0.5],
    [
        lambda x: -(x + 1),
        lambda x: x,
        lambda x: -(x - 1)
    ]
)

# ======================
# Standard plot
# ======================
fig, ax = plt.subplots(figsize=(10,6))

ax.plot(x, F, label='Forzante teorica')

ax.set_xlabel('x [a.u]')
ax.set_ylabel('F(x) [a.u.]')

ax.legend()
ax.grid(True)

plt.tight_layout()

# ======================
# Saving block
# ======================
if save:

    folder = path + "logbook"
    os.makedirs(folder, exist_ok=True)

    # ---- Standard version ----
    fig.savefig(
        os.path.join(folder, "forzante_standard.png"),
        dpi=300
    )

    folder = path + "presentazione"

    # ---- Presentation version ----
    ax.tick_params(axis='both',
                   labelsize=16,
                   width=2,
                   length=8)

    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)

    ax.legend(fontsize=16)

    fig.set_size_inches(10, 7)

    fig.savefig(
        os.path.join(folder, "forzante_presentation.png"),
        dpi=300
    )

# ======================
# Show figure
# ======================
plt.show()