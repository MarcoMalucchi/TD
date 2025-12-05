import tdwf
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np

plt.close('all')

filenames = np.loadtxt('/home/marco/Desktop/Uni_anno3/TD/Es_09/acquisizioni/elencofile.txt', dtype=str)

string = []

for i in range(len(filenames)):

    t, ch1, ch2 = np.loadtxt('/home/marco/Desktop/Uni_anno3/TD/Es_09/acquisizioni/'+filenames[i], unpack=True)

    plt.figure(i, figsize=(10,6), dpi=100)
    plt.title(filenames[i].rsplit('.',1)[0])
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.plot(t, ch1, color='orange', label='Ch1')
    plt.plot(t, ch2, color='blue', label='Ch2')
    plt.legend()
    plt.grid()
    plt.savefig('/home/marco/Desktop/Uni_anno3/TD/Es_09/logbook/'+filenames[i].rsplit('.',1)[0]+'_nuove.png')

plt.show()


