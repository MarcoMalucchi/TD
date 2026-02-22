import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh

# ==========================================
# 1. SETUP DEL SISTEMA (Identico allo script precedente)
# ==========================================
# ==========================================
# 1. PARAMETRI MODIFICABILI (INPUT)
# ==========================================
L = 0.973           # Lunghezza asta [m]
diam = 0.004      # Diametro asta [m]
E = 70e9          # Modulo Young Alluminio [Pa]
rho = 2700        # Densità [kg/m3]

# MASSE CONCENTRATE (Peso in Newton)
P1 = 3.75           # Massa 1 (Top)
P2 = 1.85           # Massa 2 (Mid)
pos_m1 = 1.0 * L    # Posizione Massa 1
pos_m2 = 0.502 * L  # Posizione Massa 2

# MOLLA FLESSIONALE (Rotazionale)
k_rot = 300.0     # Valore in Nm/rad
pos_k = 0.502 * L   # Posizione della molla

# Parametri FEM
n_el = 50
n_nodes = n_el + 1
dx = L / n_el
nodes = np.linspace(0, L, n_nodes)
I = (np.pi * diam**4) / 64; A = np.pi * (diam/2)**2
EI = E * I; m_lin = rho * A

# --- Assemblaggio Matrici (FEM) ---
K = np.zeros((2*n_nodes, 2*n_nodes))
M = np.zeros((2*n_nodes, 2*n_nodes))

def beam_element(EI, m_lin, l):
    ke = (EI/l**3) * np.array([[12, 6*l, -12, 6*l],[6*l, 4*l**2, -6*l, 2*l**2],[-12, -6*l, 12, -6*l],[6*l, 2*l**2, -6*l, 4*l**2]])
    me = (m_lin*l/420) * np.array([[156, 22*l, 54, -13*l],[22*l, 4*l**2, 13*l, -3*l**2],[54, 13*l, 156, -22*l],[-13*l, -3*l**2, -22*l, 4*l**2]])
    return ke, me

for i in range(n_el):
    ke, me = beam_element(EI, m_lin, dx)
    idx = slice(2*i, 2*i + 4)
    K[idx, idx] += ke
    M[idx, idx] += me

node_m1 = int(round(pos_m1/dx)); M[2*node_m1, 2*node_m1] += P1 / 9.81
node_m2 = int(round(pos_m2/dx)); M[2*node_m2, 2*node_m2] += P2 / 9.81
node_k = int(round(pos_k/dx)); K[2*node_k + 1, 2*node_k + 1] += k_rot

dofs_to_remove = [0, 1, 2*n_el + 1] # Vincoli Incastro-Pattino
active_dofs = np.delete(np.arange(2*n_nodes), dofs_to_remove)
K_red = K[np.ix_(active_dofs, active_dofs)]
M_red = M[np.ix_(active_dofs, active_dofs)]

# Risoluzione
evals, evecs = eigh(K_red, M_red)
freqs = np.sqrt(evals) / (2 * np.pi)

# ==========================================
# 2. ESTRAZIONE DATI MODE_IDX° MODO
# ==========================================
MODE_IDX = 2  # Indice 1 per il secondo modo
freq_modo = freqs[MODE_IDX]

# Ricostruzione forma modale
u_full = np.zeros(2*n_nodes)
u_full[active_dofs] = evecs[:, MODE_IDX]
w_shape = u_full[0::2] # Solo spostamenti laterali

# Normalizzazione e Scala Visiva
amp_visual_max = 0.15 * L  # Ampiezza massima dell'oscillazione nel grafico
w_shape_norm = w_shape / np.max(np.abs(w_shape)) * amp_visual_max

# ==========================================
# 3. SETUP ANIMAZIONE MATPLOTLIB
# ==========================================
fig, ax = plt.subplots(figsize=(8, 10))

# --- Elementi Statici (Vincoli, testi) ---
ax.axvline(0, color='k', alpha=0.2, lw=1) # Asse neutro
ax.plot(0, 0, 'ks', ms=15, zorder=1) # Incastro
ax.hlines(L, -amp_visual_max*1.5, amp_visual_max*1.5, colors='k', lw=4, zorder=1) # Pattino
ax.annotate(r'Molla $k_\theta$', xy=(0, pos_k), xytext=(-amp_visual_max*1.3, pos_k),
            arrowprops=dict(arrowstyle='->', color='purple', lw=2), color='purple')

ax.set_xlim(-amp_visual_max*1.6, amp_visual_max*1.6)
ax.set_ylim(-0.05, L*1.05)
ax.set_title(f"Animazione {MODE_IDX+1}° Modo di Vibrazione\nFrequenza: {freq_modo:.2f} Hz (Velocità Rallentata)", fontsize=14)
ax.set_xlabel("Spostamento Laterale [m]")
ax.set_ylabel("Altezza [m]")
ax.grid(True, alpha=0.3)

# --- Elementi Dinamici (da inizializzare vuoti) ---
# Colore arancione per il 2° modo (coerente con i grafici precedenti)
color_mode = '#ff7f0e'
line_beam, = ax.plot([], [], lw=4, color=color_mode)
marker_m1, = ax.plot([], [], 'o', ms=12, color=color_mode, mec='k', zorder=5)
marker_m2, = ax.plot([], [], 'o', ms=10, color=color_mode, mec='k', zorder=5)

# --- Funzione di Inizializzazione ---
def init():
    line_beam.set_data([], [])
    marker_m1.set_data([], [])
    marker_m2.set_data([], [])
    return line_beam, marker_m1, marker_m2

# --- Funzione di Aggiornamento (Core dell'animazione) ---
# frame va da 0 a frames-1
def update(frame):
    # Calcolo del fattore tempo per un ciclo coseno perfetto
    # Usiamo un coseno per partire dalla massima elongazione
    time_factor = np.cos(2 * np.pi * frame / total_frames)

    # Calcolo della posizione istantanea
    w_current = w_shape_norm * time_factor

    # Aggiornamento linea dell'asta
    line_beam.set_data(w_current, nodes)

    # Aggiornamento posizione masse
    marker_m1.set_data([w_current[node_m1]], [nodes[node_m1]])
    marker_m2.set_data([w_current[node_m2]], [nodes[node_m2]])

    return line_beam, marker_m1, marker_m2

# --- Parametri e Avvio Animazione ---
fps = 30
duration_sec = 2/freq_modo+0.5 # Durata di un ciclo visivo
total_frames = int(fps * duration_sec)

ani = FuncAnimation(fig, update, frames=total_frames,
                    init_func=init, blit=True, interval=1000/fps)

plt.tight_layout()
plt.show()

# Per salvare l'animazione come GIF (richiede imagemagick installato):
# ani.save('modo2_vibrazione.gif', writer='imagemagick', fps=30)