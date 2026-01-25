import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    return cov, avg

path = "/home/stefano/Documenti/SCUOLA/LAB3/MPU6050_11/"
fileName = "MPU6050_processData.bin"
cov_raw, avg_raw = read_elab_file_numpy(path + fileName)
assi = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]])
Vspace = 1.2
print(cov_raw.shape)   # (n,3,3)
print(avg_raw.shape)   # (n,3)

cnt = np.mean(avg_raw, axis=0)
avg_cnt= avg_raw-cnt

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(avg_raw[:, 0], avg_raw[:, 1], avg_raw[:, 2], c='blue', s=3, label='Inside')
ax.scatter(avg_cnt[:, 0], avg_cnt[:, 1], avg_cnt[:, 2], c='purple', s=3, label='Inside')

ax.scatter(assi[:,0], assi[:,1], assi[:,2],c='red')
ax.scatter(cnt[0], cnt[1], cnt[2],c='blue')
ax.quiver(assi[0,0], assi[0,1], assi[0,2],assi[1,0], assi[1,1], assi[1,2], color='r', length= 1.0, arrow_length_ratio=0.2, linewidths=0.3)
ax.quiver(assi[0,0], assi[0,1], assi[0,2],assi[3,0], assi[3,1], assi[3,2], color='y', length= 1.0, arrow_length_ratio=0.2, linewidths=0.3)
ax.quiver(assi[0,0], assi[0,1], assi[0,2],assi[5,0], assi[5,1], assi[5,2], color='g', length= 1.0, arrow_length_ratio=0.2, linewidths=0.3)

ax.axes.set_xlim3d (left=-Vspace, right=Vspace)
ax.axes.set_ylim3d (top=Vspace, bottom=-Vspace)
ax.axes.set_zlim3d (top=Vspace, bottom=-Vspace)
ax.axes.set_aspect(1.0)

plt.show()
