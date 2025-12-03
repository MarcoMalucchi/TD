import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

def make_sfera(num):
       sfera = np.empty([3, num], dtype=float)
       aureo = np.pi*(3-np.sqrt(5))
       for index in range(num):
           sfera[2,index]=1-2*index/num
           theta=np.arccos(sfera[2,index])
           fi=(index*aureo)%(2*np.pi)
           sfera[0,index]=np.sin(theta)*np.sin(fi)
           sfera[1,index]=np.sin(theta)*np.cos(fi)
       return np.append(sfera, [np.ones(num)], axis=0)

def rotation_matrix_axis(axis, theta):
    return None

if __name__ =="__main__":
    num=200
    vrs = np.identity(4)
    H=np.array([[1, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 3, 0],
                [0, 0, 0, 1]])
    S=np.array([[1, 0, 0, 0.5],
                [0, 1, 0, 0.35],
                [0, 0, 1, 0.75],
                [0, 0, 0, 1]])


    axis =[1, 0, 0]
    theta=np.radians(5)
    R = rotation_matrix(axis, theta)

    T = R @ H
    sfera=make_sfera(num)

    ellipse=T @ sfera
    M_sfera = np.identity(4, dtype=float)
    M_sfera[3,3] = -1
    #aa = np.einsum('ij,jk,ik->i', sfera.T, Msfera, sfera.T)

    T_inv = np.linalg.inv(T)

    M_ellipse = (T_inv.T @ M_sfera) @ T_inv
    eigen, vec = np.linalg.eig(M_ellipse)
    N_ell = np.linalg.inv(R) @ ellipse

    Es = S @ ellipse
    M_S = np.block([
        [S[:3,:3],      S[:3,3].reshape(3,1)],
        [np.array([0,0,0]).reshape(1,3), np.array([[1]])]
    ])
    M_Sinv = np.linalg.inv(M_S)

    # Ottenere gli indici che ordinerebbero gli autovalori in ordine crescente
    # argsort() restituisce gli indici ordinati
    idx = [1,2,3,4]
    for j in range(4):
        max = np.max(abs(vec[j,:]))
        for i in range(4):
            if abs(vec[j,i]) == max:
                idx[j]=i
                break
    # Riordinare gli autovalori e gli autovettori usando gli indici ottenuti
    eigen_sorted = eigen[idx]
    # Riordinare le colonne degli autovettori usando gli stessi indici
    vec_sorted = vec[:, idx]
    #vicino ai versori
    for j in range(3):
        if vec_sorted[:, j]@vrs[:,j] < 0:
            vec_sorted[:, j] *= -1

    #uu = (vec.T @ M_ellipse) @ vec
    #cc=np.linalg.inv(uu)
    print("mareice sfera\n", M_sfera)
    print("Matrice scala\n", H)
    print("Matrice rotazione\n", R)
    print("Mareice composta RH\n", T)
    print("T inversa\n", T_inv)
    print("Matrice ellisoide\n", M_ellipse)
    print("Autovalori\n", eigen_sorted)
    print("Autovettori\n", vec_sorted)

    fig=plt.figure('Ellissoide', figsize=(10., 7.), dpi=300)
    ax=fig.add_subplot(projection ='3d')
    #for index in range(num):
    ax.scatter(sfera[0,:], sfera[1,:], sfera[2,:], s=0.25, c='r', marker='.')
    ax.scatter(ellipse[0,:], ellipse[1,:], ellipse[2,:], c='b', s=0.25, marker='.')
    ax.scatter(Es[0,:], Es[1,:], Es[2,:], c='g', s=0.25, marker='.')

    ax.quiver(0, 0, 0, vec_sorted[0,0], vec_sorted[1,0], vec_sorted[2,0], color='g', normalize=False, length= 2, arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(0, 0, 0, vec_sorted[0,1], vec_sorted[1,1], vec_sorted[2,1], color='r', normalize=False, length= 2, arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(0, 0, 0, vec_sorted[0,2], vec_sorted[1,2], vec_sorted[2,2], color='y', normalize=False, length= 2, arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(0, 0, 0, 1, 0, 0, color='g', length= 2, arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(0, 0, 0, 0, 1, 0, color='r', length= 2, arrow_length_ratio=0.2, linewidths=0.3)
    ax.quiver(0, 0, 0, 0, 0, 1, color='y', length= 2, arrow_length_ratio=0.2, linewidths=0.3)


    ax.axes.set_xlim3d (left=-3, right=3)
    ax.axes.set_ylim3d (top=-3, bottom=3)
    ax.axes.set_zlim3d (top=-3, bottom=3)
# Modificare la dimensione del font dei tick su tutti gli assi
    ax.tick_params(axis='both', which='major', labelsize=4)
    ax.tick_params(axis='z', which='major', labelsize=4) # Specifico per l'asse Z
# Modificare il font e la dimensione delle etichette degli assi
    ax.set_xlabel('X', fontsize=4, fontfamily='serif')
    ax.set_ylabel('Y', fontsize=4, fontfamily='serif')
    ax.set_zlabel('Z', fontsize=4, fontfamily='serif')
    ax.set_box_aspect([6, 6, 6])
    ax.view_init(elev=-0. , azim=0.)
    plt.show()
