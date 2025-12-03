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
        num = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        print(num)
        # Legge tutto il resto come float32
        data = np.fromfile(f, dtype=np.float32)

    if data.size % 12 != 0:
        raise ValueError("File non allineato: numero di float non multiplo di 12")

    n = data.size // 12
    data = data.reshape(n, 12)

    # prime 9 colonne → covarianze
    cov = data[:, 0:9].reshape(n, 3, 3).astype(np.float64)

    # ultime 3 colonne → medie
    avg = data[:, 9:12].astype(np.float64)

    avg_mean = np.mean(np.linalg.norm(avg, axis=1))
    avg = avg/avg_mean
    cov=cov/(avg_mean*avg_mean)

    return cov, avg

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



if __name__ == "__main__":
    # Example: read raw accelerometer bin file medie e covarianze
    vrs=np.identity(4)
    Vspace = 1.2
    cov_raw, avg_raw = read_elab_file_numpy("dati1.bin")
    sigma = np.sqrt(cov_raw[:,0,0] + cov_raw[:,1,1] + cov_raw[:,2,2])

    p0 = initial_guess_sphere(avg_raw)
    popt, pcov = curve_fit(
        Eq_elli,
        avg_raw,                      # xdata
        np.zeros(len(avg_raw)),       # ydata  = 0
        p0[:-1],
        sigma=sigma,
        absolute_sigma=True,
        maxfev=10000
    )

    Q, p = unpack_a(popt)
    f= p0[9]

    Fit_Observed = np.block([
        [Q,        p.reshape(3,1)],
        [p.reshape(1,3), np.array([[f]])]
    ])
    c = np.linalg.solve(Q, -p)
    e_vl, e_vc = np.linalg.eig(Q)

    #Matrice di shift
    Sft = np.identity(4)
    Sft[:3,3]=-c
    SftInv = np.linalg.inv(Sft)
    #matrice dell'ellissoide traslata nell'origine
    MM = SftInv.T@Fit_Observed@SftInv
    MM = (MM.T+MM)*0.5
    MM = MM/MM[3,3]*-1

    # Ottenere gli indici che ordinerebbero gli autovalori in maniera da rendere piccolo l'angolo di rotazione
    idx = [1,2,3]
    for j in range(3):
        max = np.max(abs(e_vc[j,:]))
        for i in range(3):
            if abs(e_vc[j,i]) == max:
                idx[j]=i
                break
    # Riordinare gli autovalori e gli autovettori usando gli indici ottenuti
    eigen_sorted = e_vl[idx]
    vec_sorted = e_vc[:, idx]
    #vicino ai versori evitare angoli di 180°
    for j in range(3):
        if vec_sorted[:, j]@vrs[:3,j] < 0:
            vec_sorted[:, j] *= -1

    #Matrice di rotazione
    R1 = np.identity(4)
    R1[:3,:3]=vec_sorted
    R1Inv = R1.T#np.linalg.inv(R1) #inversa=trasposta

    #Matrice di scala
    Sc = np.diag(np.append(np.sqrt(np.diag(MM[:3,:3])), 1))
    ScInv=np.linalg.inv(Sc)

    Calib=Sc@(R1.T@Sft)
    omg=colonna = np.ones((avg_raw.shape[0], 1))
    avg_raw_omg = np.hstack((avg_raw, colonna))
    AC = (Calib @ avg_raw_omg.T).T
    AC=AC.T
    print("Matrice ellissoide punti osservati\n", Fit_Observed)

    print("Center\n", c)
    print("Eigenvalue\n", eigen_sorted)
    print("Eigenvector\n", vec_sorted)





    fig=plt.figure('Ellissoide', figsize=(10., 7.), dpi=300)
    ax=fig.add_subplot(projection ='3d')
    ax.scatter(avg_raw[:, 0], avg_raw[:,1], avg_raw[:, 2], s=0.25, c='r', marker='.')
    ax.scatter(AC[:, 0], AC[:,1], AC[:,2], s=1, c='b', marker='.')

    ax.quiver(0, 0, 0, 1, 0, 0, color='g', length= 2, arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(0, 0, 0, 0, 1, 0, color='r', length= 2, arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(0, 0, 0, 0, 0, 1, color='y', length= 2, arrow_length_ratio=0.2, linewidths=0.3)

    ax.quiver(c[0], c[1],c[2],
        vec_sorted[0,0], vec_sorted[1,0], vec_sorted[2,0],
        color='g', normalize=False, length= 2,
        arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(c[0], c[1],c[2], vec_sorted[0,1], vec_sorted[1,1], vec_sorted[2,1], color='r', normalize=False, length= 2, arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(c[0], c[1],c[2], vec_sorted[0,2], vec_sorted[1,2], vec_sorted[2,2], color='y', normalize=False, length= 2, arrow_length_ratio=0.2, linewidths=0.3)


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
    ax.set_xlabel('X', fontsize=4, fontfamily='serif')
    ax.set_ylabel('Y', fontsize=4, fontfamily='serif')
    ax.set_zlabel('Z', fontsize=4, fontfamily='serif')

# Modificare la dimensione del font dei tick su tutti gli assi
    ax.tick_params(axis='both', which='major', labelsize=4)
    ax.tick_params(axis='z', which='major', labelsize=4) # Specifico per l'asse Z se necessario

    ax.set_box_aspect([6, 6, 6])
    ax.view_init(elev=30. )
    plt.show()
