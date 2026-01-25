import numpy as np
from numpy.linalg import inv, pinv

def read_elab_file_numpy(filename):
    print(filename)
    # Legge il numero di record
    with open(filename, "rb") as f:
        # Legge primo uint32
        #num = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        #print(num)
        # Legge tutto il resto come float32
        data = np.fromfile(f, dtype=np.float32)

    if data.size % 9 != 0:
        raise ValueError("File non allineato: numero di float non multiplo di 9")

    n = data.size // 9
    data = data.reshape(n, 9)
    # prime 6 colonne → covarianze
    cov = np.zeros((n,3,3))
    i, j = np.triu_indices(3)
    cov[:, i, j] = data[:, 0:6]
    cov[:, j, i] = data[:, 0:6]

    # ultime 3 colonne → medie
    avg = data[:, 6:10].astype(np.float64)


    return cov.astype(np.float64), avg.astype(np.float64)


# ---------------------------------------------
# Funzioni di utilità
# ---------------------------------------------

def build_design_matrix(obs):
    """
    Costruisce la matrice D per la forma quadratica dell'ellissoide:
        x^T Q x + 2 p^T x + f
    """
    x = obs[:,0]
    y = obs[:,1]
    z = obs[:,2]
    D = np.column_stack([
        x*x, y*y, z*z, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z
    ])
    return D

def unpack_params(a):
    """
    Da vettore 9 o 10 parametri a Q, p, f
    """
    Q = np.array([[a[0], a[3], a[4]],
                  [a[3], a[1], a[5]],
                  [a[4], a[5], a[2]]], dtype=np.float64)
    p = 0.5 * np.array([a[6], a[7], a[8]], dtype=np.float64)
    f = a[9] if len(a) == 10 else None
    return Q, p, f

def initial_guess_sphere(obs, R0=256):
    """
    Stima iniziale: sfera centrata sulla media delle osservazioni
    """
    cx, cy, cz = obs.mean(axis=0)
    a0 = a1 = a2 = 1.0
    a3 = a4 = a5 = 0.0
    a6, a7, a8 = -2*cx, -2*cy, -2*cz
    a9 = cx*cx + cy*cy + cz*cz - R0*R0
    return np.array([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9], dtype=float)

# ---------------------------------------------
# Fit iterativo a blocchi
# ---------------------------------------------

def fit_ellipsoid_iterative(obs, sigma=None, max_iter=20, tol=1e-6):
    """
    Fit robusto ellissoide usando blocchi Q,p ↔ f
    obs: Nx3 array di dati
    sigma: optional array di pesi (varianze) per ponderazione
    """
    N = obs.shape[0]

    # normalizzazione per stabilità numerica
    mean_obs = obs.mean(axis=0)
    std_obs = obs.std(axis=0)
    obs_n = (obs - mean_obs) / std_obs

    # stima iniziale (sfera)
    a = initial_guess_sphere(obs_n, R0=1.0)  # R0=1 per norma unitaria
    Q, p, f = unpack_params(a)

    for iteration in range(max_iter):
        # --------- Fit Q,p con f fisso ---------
        D = build_design_matrix(obs_n)
        y = -np.ones(N)*f  # f fisso

        if sigma is not None:
            W = np.diag(1.0/sigma**2)
            # soluzione dei minimi quadrati pesati
            lhs = D.T @ W @ D
            rhs = D.T @ W @ y
            a_Qp = np.linalg.solve(lhs, rhs)
        else:
            # minimi quadrati non pesati
            a_Qp = np.linalg.lstsq(D, y, rcond=None)[0]

        # aggiorna Q e p
        Q, p, _ = unpack_params(np.append(a_Qp,0))  # f rimane fisso

        # --------- Fit f con Q,p fissati ---------
        f_new = np.mean(-np.sum(obs_n @ Q * obs_n, axis=1) - 2*obs_n @ p)
        # convergenza
        if np.abs(f_new - f) < tol:
            f = f_new
            break
        f = f_new

    # --------- Riportare i parametri alla scala originale ---------
    S = np.diag(std_obs)
    Q_s = inv(S) @ Q @ inv(S)
    p_s = inv(S) @ (p - Q @ mean_obs)
    f_s = f + mean_obs @ Q @ mean_obs - 2*p @ mean_obs

    # centro e autovalori
    try:
        b = -inv(Q_s) @ p_s
    except np.linalg.LinAlgError:
        b = -pinv(Q_s) @ p_s

    eigvals, eigvecs = np.linalg.eigh(Q_s)

    return {
        'Q': Q_s,
        'p': p_s,
        'f': f_s,
        'center': b,
        'eigvals': eigvals,
        'eigvecs': eigvecs
    }

# ---------------------------------------------
# Esempio di utilizzo
# ---------------------------------------------
if __name__ == "__main__":
    # obs = Nx3 array dei dati raw
    # sigma opzionale
    path = "/home/stefano/Documenti/SCUOLA/LAB3/MPU6050_11/"
    fileName = "MPU6050_processData.bin"

    cov_raw, obs = read_elab_file_numpy(path + fileName)

    result = fit_ellipsoid_iterative(obs)

    print("Centro stimato:", result['center'])
    print("Q:", result['Q'])
    print("p:", result['p'])
    print("f:", result['f'])
    print("Autovalori:", result['eigvals'])
