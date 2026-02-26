import tdwf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import numpy as np
import scipy.optimize as so
import math
    
# ==========================================================================
#  Parametri dello script
nspaz = 10          # numero di spazzate
nper = 2           # numero periodi usati per la stima   
npt = 8192          # numero MASSIMO di punti acquisiti
nf = 500           # numero di frequenze nello sweep da f0 a f1   
f0 = 2e3
f1 = 1.6e4
flag_return = False  # A/R? o solo andata?
#flag_show = True    # Visualizza i segnali nel tempo? (è uguale...)
# vettore delle frequenze
fv = np.logspace(np.log10(f0), np.log10(f1), nf) 

# ==========================================================================
#  Configurazione base AD2 (più parametri impostati dopo)
ad2 = tdwf.AD2()
ad2.vdd = 5
ad2.vss = -5
ad2.power(True)

wavegen = tdwf.WaveGen(ad2.hdwf)
wavegen.w1.ampl = 0.3   #ampiezza 0.3V e offset 0.0V per oscillazioni in minimo positivo
wavegen.w1.freq = 1e3
wavegen.w1.func = tdwf.funcSine 
wavegen.w1.offs = 0.0
wavegen.w1.start()
scope = tdwf.Scope(ad2.hdwf)
scope.ch1.rng = 5
scope.ch2.rng = 5


# ==========================================================================
#   Esecuzione...
# [1] Preparazione dataset
Am = np.full((nf, 2*nspaz), np.nan)    # pre-allocazione dati (nan non sono plottati!) 
phim = np.full((nf, 2*nspaz), np.nan)  # pre-allocazione dati

# [2] Ciclo misura

for j in range(nspaz):  # Ciclo spazzate
    for ar in range(2 if flag_return else 1): # Ciclo A/R
        for ii in range(len(fv)):  # Ciclo frequenze
            # [3a] calcolo frequenza
            if ar==0: # caso A(ndata)
                findex = ii
            else:     # caso R(itorno)
                findex = len(fv)-ii-1
            # Frequenza attuale
            ff = fv[findex]
            # [3b] stima parametri di sampling
            #
            #  COSA VOGLIO: misurare nper con al massimo npt punti acquisizione
            #  DOMANDA: quale è la MASSIMA frequenza di sampling che posso usare?
            #
            #  NOTARE: solo 100MSa/s intero (qui df) è una frequenza ammessa.
            #
            #  SE voglio misurare nper periodi a ff devo misura per un tempo TT = nper/ff
            #  SE misuro ad un rate fs, mi servono npt = fs*nper/ff punti di acquisizione
            #
            #  voglio che fs*nper/ff sia il più alto possibile ma al massimo uguale a npt 
            #  (altrimenti no buco il buffer...), per ottenere questo sceglo un df intero 
            #  in modo che fs = 100MHz/df soddisfi la relazione sopra
            #
            #  => df = celing(100MHz*nper/(npt*ff)) 
            #
            df = math.ceil(100e6*nper/(npt*ff))
            scope.fs = 100e6/df
            scope.npt = int(scope.fs*nper/ff)
            #  Ribadiamo il trigger... 
            scope.trig(True, hist = 0.01)   #provare a disattivare il trigger per velocizzare.
            wavegen.w1.freq = ff 
            # [3c] Campionamento e analisi risultati
            scope.sample()
            fitfunc = lambda x,A,phi, offset: A * np.cos(2*np.pi*ff*x + phi) + offset
            pp1,cm1 = so.curve_fit(fitfunc, scope.time.vals, scope.ch1.vals, p0=[1,0,0])
            pp2,cm2 = so.curve_fit(fitfunc, scope.time.vals, scope.ch2.vals, p0=[1,0,0])
            # Volendo ci sono gli errori...
            #epp1 = np.sqrt(np.diagonal(cm1))
            #epp2 = np.sqrt(np.diagonal(cm2))
            # [3d] Aggiustamento della fase (se ampiezza negativa...)
            if pp1[0] < 0: 
                pp1[0] *= -1
                pp1[1] += np.pi    
            if pp2[0] < 0:
                pp2[0] *= -1
                pp2[1] += np.pi    
            # [3e] Aggiornamento dei dati
            Am[findex, ar+2*j] = pp2[0]/pp1[0]
            phim[findex, ar+2*j] = (pp2[1]-pp1[1] + np.pi) % (2*np.pi) - np.pi
        
    print('spazzata',j+1,'di',nspaz,'effetuata')

datas = np.full((nf,5),np.nan)

#print('Gain', Am)
#print('Dephasing',phim)

if flag_return:
    for ii in range(nf):
        datas[ii,0] = fv[ii]
        datas[ii,1] = np.mean(Am[ii])
        datas[ii,2] = np.std(Am[ii])
        datas[ii,3] = np.mean(phim[ii])
        datas[ii,4] = np.std(phim[ii])
else:
    for ii in range(nf):
        datas[ii,0] = fv[ii]
        datas[ii,1] = np.mean(Am[ii,0::2])
        datas[ii,2] = np.std(Am[ii,0::2])
        datas[ii,3] = np.mean(phim[ii,0::2])
        datas[ii,4] = np.std(phim[ii,0::2])

#print('datas',datas)

fig = plt.figure('Plot Gain-Dephasing', figsize=(10, 6), dpi=100)

ax1, ax2= fig.subplots(2, 1, sharex=True, gridspec_kw= dict(height_ratios=[1, 1], hspace=0.05))

ax1.errorbar(datas[:,0], datas[:,1], datas[:,2], fmt='.', label='Gain')
ax1.set_ylabel('G[a.u.]')
ax1.set_yscale('log')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

ax2.errorbar(datas[:,0], datas[:,3], datas[:,4], fmt='.', label='Dephasing')
ax2.set_xscale('log')
ax2.grid(ls='dashed', color='lightgray')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Dephasing [rad]')
ax2.legend()

fig.align_ylabels((ax1, ax2))
#plt.ylim(-0.001, 0.001)

plt.show()

Save = True

if Save:
    path = "/home/marco/Desktop/Uni_anno3/TD/Es_11/acquisizioni/"
    name = "spazzata_minimo_positivo.txt"
    np.savetxt(path+name, datas, delimiter=',', 
           header="#Freq[Hz], #Gain_mean[a.u.], #Gain_std[a.u.], #Phase_mean[rad], #Phase_std[rad]")
    print(f'Dati salvati in {path+name}')
    ad2.close()
    
else:
    ad2.close()