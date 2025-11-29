import tdwf
import numpy as np
import matplotlib.pyplot as plt
import time

plt.close('all')
path='/home/marco/Desktop/Uni_anno3/TD/Es_01/misure'  #percorso in cui salvare i dati

ad2 = tdwf.AD2()    #connessione all'hardware

'''Configurazione dell'oscilloscopio'''
scope = tdwf.Scope(ad2.hdwf)
scope.fs = 1e6
scope.npt = 8192
scope.ch1.rng = 50.0
scope.ch2.rng = 5.0
#scope.trig(True, level=2, sour = tdwf.trigsrcCh1)
scope.ch1.avg = False
scope.ch2.avg = False

#V = np.linspace(-2.0, 2.0, 10)  #Vettore dei valori di tensione da impostare

'''Configurazione di W1'''
wgen = tdwf.WaveGen(ad2.hdwf)   #inizializzazione del generatore
wgen.w1.config(offs=2, func=tdwf.funcDC)    #impostazione dell'offset del segnale e del tipo di funzione in output
wgen.w1.start() #avvio del generatore

time.sleep(0.5)
scope.sample()

'''Acquisizione dati'''
data = np.column_stack((scope.time.vals, scope.ch1.vals, scope.ch2.vals))
#np.savetxt(path+'/misura50.txt', data, delimiter='\t', header='Time [s], Ch1 [V], Ch2 [V]')

'''Plot acquisizioni'''

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(scope.time.vals, scope.ch1.vals,'.', label='Ch1')
plt.plot(scope.time.vals, scope.ch2.vals, '.', label='Ch2')
plt.xlabel("Tempo [s]")
plt.ylabel("Ch1 [V]")
plt.legend()


plt.show()



