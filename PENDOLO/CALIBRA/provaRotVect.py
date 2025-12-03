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

def rotation_matrixv1_v2(v1, v2):
    c=np.cross(v1,v2)
    d=np.inner(v1,v2)
    theta = np.arccos(d/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    r=rotation_matrix(c,theta)
    return r[:3,:3], theta




a=np.array([0.04108221, 0.01551886, 1.00625981])
a= np.array([0, 1, 1.001])
b=np.array([0,0,-1])
Rt1, th1 = rotation_matrixv1_v2(a, b)
a1 = Rt1@a
print(a, np.linalg.norm(a))
print(a1,  np.linalg.norm(a1))
