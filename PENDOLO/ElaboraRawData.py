import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as mt
import scipy.stats as stats
from scipy.optimize import curve_fit

def mahalanobis_distance(x, mu, inv_cov):
    diff = x - mu
    # Formula: sqrt( (x-mu)^T * InvCov * (x-mu) )
    return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))


def read_file(filename):
    # Legge il numero di record
    with open(filename, "rb") as f:
        # Legge primo uint32
        #num = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        #print(num)
        # Legge tutto il resto come float32
        data = np.fromfile(f, dtype=np.int16)

    if data.size % 3 != 0:
        raise ValueError("File non allineato: numero di float non multiplo di 9")

    n = data.size // 3
    data = data.reshape(n, 3)
    return data

path = "/home/marco/Desktop/Uni_anno3/TD/PENDOLO/MPU6050_11/"
fileName = "MPU6050_rawData00000.bin"

data='MPU6050_listFile.txt'

NomiFile=[]
Dati = open(path+data, 'r')

while True:
    Line = Dati.readline()
    if Line == '': break
    NomiFile.append(Line.strip('\n'))
Dati.close()

raw1 = []
for i in range(len(NomiFile)):
    raw1.append(read_file(path + NomiFile[i]))

acc_raw = np.array(raw1)
means = np.mean(acc_raw, axis=1) # Risultato 402x3
stds = np.std(acc_raw, axis=1, ddof=1) # Risultato 402x3
covs = np.array([np.cov(acc_raw[i], rowvar=False) for i in range(acc_raw.shape[0])])
inv_covs=np.linalg.inv(covs[:])

dist_m = np.zeros((acc_raw.shape[0], acc_raw.shape[1]))
for i in range(acc_raw.shape[0]):
    # Recuperiamo i parametri del gruppo i-esimo
    mu = means[i]
    inv_c = inv_covs[i]

    for j in range(acc_raw.shape[1]):
        # Singola osservazione j del gruppo i
        x = acc_raw[i, j]
        diff = x - mu
        # Calcolo Mahalanobis: sqrt( diff * InvCov * diff.T )
        dist_m[i, j] = np.sqrt(np.dot(np.dot(diff, inv_c), diff.T))

# Soglia (es. 3.0 o calcolata con Chi-quadro come visto prima)
soglia = 3.5

# Crea n maschere in un colpo solo (n x k)
masks = dist_m < soglia

avgf = np.zeros((acc_raw.shape[0], 3))    # Array per n medie (3 variabili)
stdf = np.zeros((acc_raw.shape[0], 3))    # Array per n std (3 variabili)
covf = np.zeros((acc_raw.shape[0], 3, 3)) # Array per n matrici di covarianza (3x3)

for i in range(acc_raw.shape[0]):
    # Estraiamo solo i dati validi per l'esperimento i
    dati_validi = acc_raw[i][masks[i]]

    # Controllo di sicurezza: servono almeno 2 punti per la covarianza (ddof=1)
    if len(dati_validi) > 1:
        avgf[i] = np.mean(dati_validi, axis=0)
        stdf[i] = np.std(dati_validi, axis=0, ddof=1)
        covf[i] = np.cov(dati_validi, rowvar=False)
    else:
        # Gestione errore se i dati sono troppo sporchi o insufficienti
        avgf[i] = np.nan
        stdf[i] = np.nan
        covf[i] = np.nan


Ris = 100
i=2
j=350
Ds = np.max(acc_raw[j,masks[j],i])-np.min(acc_raw[j,masks[j],i])
x=np.linspace(0, 1000, acc_raw[j,masks[j],i].shape[0])
fig1=plt.figure('Distribuzione2', figsize=(10, 6), dpi=Ris)
ax1 = fig1.subplots()
ax1.scatter(x,acc_raw[j,masks[j],i] , c='blue', s=3, label='Inside')

fig2=plt.figure('Distribuzione1', figsize=(10, 6), dpi=Ris)
ax2 = fig2.subplots()

n, bins, patches = ax2.hist(x=acc_raw[j,masks[j],i], bins='auto', color = 'blue', edgecolor = 'gray', density=True)

ax2.plot(bins, stats.norm.pdf(bins, avgf[j,i], stdf[j,i]), '--', color='orange')


plt.show()
