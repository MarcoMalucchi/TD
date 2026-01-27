import numpy as np
import inspect
from scipy.optimize import curve_fit
import functools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def read_elab_file_numpy(filename):
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

    avg_mean = np.mean(np.linalg.norm(avg, axis=1))
    avg = avg/avg_mean
    cov=cov/(avg_mean*avg_mean)
    avg_mean = np.mean(np.linalg.norm(avg, axis=1))

    return cov, avg, avg_mean

#Equazione del curve fit
def Eq_elli(obs, a11, a12, a13, a22, a23, a33, p1, p2, p3):
    x = obs[:,0]
    y = obs[:,1]
    z = obs[:,2]

    D = np.column_stack([
        x*x,
        y*y,
        z*z,
        2*x*y,
        2*x*z,
        2*y*z,
        2*x,
        2*y,
        2*z,
        np.ones_like(x)
    ])

    a = np.array([a11,a12,a13,a22,a23,a33,p1,p2,p3,-1])

    return D @ a

# Condizioni iniziali a partire da sfera
def initial_guess_sphere(obs, R0=0):
    cx, cy, cz = obs.mean(axis=0)

    a11 = 1
    a12 = 1
    a13 = 1
    a22 = 0
    a23 = 0
    a33 = 0
    p1 = -2*cx
    p2 = -2*cy
    p3 = -2*cz
    if R0 != 0:
        f = cx*cx + cy*cy + cz*cz - R0*R0
    else:
        f=-1

    p0 = np.array([a11,a12,a13,a22,a23,a33,p1,p2,p3,f], dtype=float)
    return p0

def unpack_a(a):
    Q = np.array([
        [a[0], a[3], a[4]],
        [a[3], a[1], a[5]],
        [a[4], a[5], a[2]]
    ])
    p = 0.5*np.array([a[6], a[7], a[8]])
    return Q,p

def center (M):
    A = M[:3,:3]
    b = M[:3, 3]
    x0 = -np.linalg.solve(A, b)
    return x0

def shift (M, x0):
    S = np.identity(4, dtype=float)
    S[:3,3]=-np.asarray(x0, float)
    return S

def map_ellipsoid_to_unit_sphere(M):
    """
    M: 4x4 quadric matrix of an ellipsoid (assumes A3 positive definite).
    Returns H such that for points y = H @ x:
      y^T B y = x^T M x
    with B = diag(I3, -1). In other terms M_new = (H^{-1})^T M H^{-1} = B.
    """
    # 1) center
    x0 = center(M)

    # 2) translate to origin (points)
    T = shift(M, x0)   # y = T @ x moves center to origin

    # 3) compute centered quadric
    T_inv = np.linalg.inv(T)
    M1 = T_inv.T @ M @ T_inv
    M1 = 0.5*(M1 + M1.T) #rende la matrice simmetrica a causa di errori computazionali
    M1 = M1/M[3,3]*-1 #normalizza ad M1[3,3]=-1

    A3c = M1[:3,:3]   # should be SPD for ellipsoid

    # 4) Cholesky A3c = C C^T (C lower)
    C = np.linalg.cholesky(A3c)     # raises error if not SPD
    L = C.T                         # L^T L = A3c

    # 5) build H = [L 0; 0 1] @ T  (apply translation then L)
    L_block = np.identity(4)
    L_block[:3,:3] = L
    H = L_block @ T

    # verification (optional)
    B = np.diag([1.,1.,1.,-1.])
    Hinv = np.linalg.inv(H)
    B_check = Hinv.T @ M @ Hinv
    B_check = 0.5*(B_check + B_check.T)

    return H, B_check, x0, L, A3c, C

def rotation_matrix(axis, theta):
    """
    Restituisce la matrice di rotazione 3×3 che ruota di un angolo theta (rad)
    attorno all'asse 'axis' (vettore 3D).
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)  # normalizzazione

    ux, uy, uz = axis
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.identity(4, dtype=float)
    R[:3, :3] = np.array([
        [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
    ])

    return R

def rotation_matrixv1_v2(v1, v2):
    c=np.cross(v1,v2)
    d=np.inner(v1,v2)
    theta = np.arccos(d/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    r=rotation_matrix(c,theta)
    return r[:3,:3]


if __name__ == "__main__":
    plt.close('all')
    # Example: read raw accelerometer bin file average and covariance
    vrs=np.identity(4) #Versori degli assi standard
    Vspace=1.0 #Dimensione finestra plottaggio
    path = "/home/marco/Desktop/Uni_anno3/TD/PENDOLO/MPU6050_11/"
    fileName = "MPU6050_processData.bin"

    cov_raw, avg_raw, scale = read_elab_file_numpy(path + fileName) #lettura osservazioni
    #scale=1.0
    samples = avg_raw/scale
    cov=cov_raw/(scale**2)

    sigma = np.sqrt(cov[:,0,0] + cov[:,1,1] + cov[:,2,2])

    p0 = initial_guess_sphere(samples)
    popt, pcov = curve_fit(
        Eq_elli,
        samples,                      # xdata
        np.zeros(len(samples)),       # ydata  = 0
        p0[:-1],
        sigma=sigma,
        absolute_sigma=True,
        maxfev=10000
    )

    Q, p = unpack_a(popt)
    f= p0[9]

    M = np.block([
        [Q,        p.reshape(3,1)],
        [p.reshape(1,3), np.array([[f]])]
    ])

    H, B_check, x0, L, A3c, C = map_ellipsoid_to_unit_sphere(M)
    np.set_printoptions(precision=8, suppress=True)
    print("H (trasformazione):\n", H)
    print("\nB_check (dovrebbe essere diag(1,1,1,-1)):\n", B_check)
    print("\nCentro trovato:", x0)
    print("\nL (3x3) usato:\n", L)

    #per disegno
    omg=colonna = np.ones((samples.shape[0], 1))
    avg_raw_omg = np.hstack((samples, colonna))
    AC = (H @ avg_raw_omg.T).T
    Ds=np.linalg.norm(samples-AC[:,:3], axis=1) #distanza tra i punti osservati e quelli trasformati7

    e_vl, e_vc = np.linalg.eig(Q) #autovettori direzioni assi principali
    #sort eigenvector max main diagonal
    idx = [1,2,3]
    for j in range(3):
        max = np.max(abs(e_vc[j,:]))
        for i in range(3):
            if abs(e_vc[j,i]) == max:
                idx[j]=i
                break
    # Riordinare gli autovalori e gli autovettori usando gli indici ottenuti
    eigen_sorted = e_vl[idx]
    # Riordinare le colonne degli autovettori usando gli stessi indici
    vec_sorted = e_vc[:, idx]
    #if np.cross(vec_sorted[:,1],vec_sorted[:,2]) @ vec_sorted[:,0] < 0:
    #    vec_sorted[:,0] *= -1
    for j in range(3):
        if vec_sorted[:, j]@vrs[:3,j] < 0:
            vec_sorted[:, j] *= -1

    print("Matrice ellissoide punti osservati\n", M)

    v_cal = (C @ (samples-x0).T).T
    Rt1 = rotation_matrixv1_v2(v_cal[0],vrs[:3,2])
    v_cal_rot = (Rt1 @ v_cal.T).T

    print("Eigenvalue\n", eigen_sorted)
    print("Eigenvector\n", vec_sorted)

    fig=plt.figure('Ellissoide', figsize=(10., 8.), dpi=300)
    ax=fig.add_subplot(111, projection ='3d')
    ax.scatter(samples[:, 0], samples[:,1], samples[:, 2], s=1, c='r', marker='.')
    ax.scatter(AC[:, 0], AC[:,1], AC[:, 2], s=1, c='b', marker='.')

    ax.quiver(0, 0, 0, 1, 0, 0, color='g', length= 0.5, arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(0, 0, 0, 0, 1, 0, color='r', length= 0.5, arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(0, 0, 0, 0, 0, 1, color='y', length= 0.5, arrow_length_ratio=0.2, linewidths=0.3)
#orientazioni assi iniziali
    ax.quiver(x0[0], x0[1],x0[2], vec_sorted[0,0], vec_sorted[1,0], vec_sorted[2,0],
        color='g', normalize=False, length= 1, arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(x0[0], x0[1],x0[2], vec_sorted[0,1], vec_sorted[1,1], vec_sorted[2,1], color='r', normalize=False, length= 1, arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(x0[0], x0[1],x0[2], vec_sorted[0,2], vec_sorted[1,2], vec_sorted[2,2], color='y', normalize=False, length= 1, arrow_length_ratio=0.2, linewidths=0.3)


    ax.axes.set_xlim3d (left=-Vspace, right=Vspace)
    ax.axes.set_ylim3d (top=Vspace, bottom=-Vspace)
    ax.axes.set_zlim3d (top=Vspace, bottom=-Vspace)

# Personalizzazione della griglia
    ax.xaxis._axinfo["grid"]['color'] = 'gray'
    ax.xaxis._axinfo["grid"]['linewidth'] = 0.2
    ax.yaxis._axinfo["grid"]['color'] = 'gray'
    ax.yaxis._axinfo["grid"]['linewidth'] = 0.2
    ax.zaxis._axinfo["grid"]['color'] = 'gray'
    ax.zaxis._axinfo["grid"]['linewidth'] = 0.2

# Personalizzazione del colore dei pannelli (sfondo assi) per maggiore contrasto
    ax.xaxis.pane.set_color((1.0, 1.0, 1.0, 0.8)) # Bianco semi-trasparente
    ax.yaxis.pane.set_color((1.0, 1.0, 1.0, 0.8))
    ax.zaxis.pane.set_color((1.0, 1.0, 1.0, 0.8))

# Modificare il font e la dimensione delle etichette degli assi
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)

# Modificare la dimensione del font dei tick su tutti gli assi
    ax.tick_params(axis='both', which='major', labelsize=4)
    ax.tick_params(axis='z', which='major', labelsize=4) # Specifico per l'asse Z se necessario

    ax.axes.set_aspect('equal')
    ax.view_init(elev=30, azim=-45)
    plt.show()
    fig2=plt.figure('Ellissoide1', figsize=(10., 8.), dpi=300)
    ax2=fig2.add_subplot(111, projection ='3d')
    for j in range(len(avg_raw)):
        _, vct = np.linalg.eig(cov[j])
        ax2.quiver(0,0,0, vct[0,2], vct[1,2], vct[2,2], color='b', normalize=False, length= 1, arrow_length_ratio=0.2, linewidths=0.3)
    ax2.axes.set_xlim3d (left=-Vspace, right=Vspace)
    ax2.axes.set_ylim3d (top=Vspace, bottom=-Vspace)
    ax2.axes.set_zlim3d (top=Vspace, bottom=-Vspace)




    plt.show()
