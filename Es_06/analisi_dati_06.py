import numpy as np
import matplotlib.pyplot as plt
import math as mt
from pathlib import Path
from scipy.optimize import curve_fit

plt.close('all')

filenames = np.loadtxt('/home/marco/Desktop/Uni_anno3/TD/Es_06/Spettri_LED/elencofile.txt', dtype=str)

mean_x = []
std_x = []
color = []

for i in range(len(filenames)):
    
    color.append(filenames[i].split('_')[2])

    #print(filenames[i])
    x, y = np.loadtxt('/home/marco/Desktop/Uni_anno3/TD/Es_06/Spettri_LED/'+filenames[i], unpack=True)

    y -= 1490  # sottraggo il valore di offset

    #negative_mask = y < 0  giusto per vedere se ci sono valori negativi
    #print(sum(negative_mask))

    #print(np.sum(y))   Per valutare la non nomralizzazione dello spettro
    
    fig = plt.figure(i, figsize=(10,6), dpi=100, layout='constrained')
    plt.plot(x, y, '.', label='spettro'+filenames[i])
    plt.title(filenames[i][:-4])
    plt.xlabel('$\lambda$ [nm]', fontsize='18')
    plt.ylabel('I [counts]', fontsize='18')
    plt.xticks(fontsize='16')
    plt.yticks(fontsize='16')
    plt.legend()
    plt.grid()
    
    mean_x = np.append(mean_x, np.sum(x * (y/np.sum(y))))
    #print('lunghezza onda media per', filenames[i], ' : ', mean_x[i])

    #larghezza a metà altezza
    half_max = np.max(y) / 2
    indices_above_half_max = np.where(y >= half_max)[0]
    if len(indices_above_half_max) > 0:
        hwhm = (x[indices_above_half_max[-1]] - x[indices_above_half_max[0]])/2
        std_x = np.append(std_x, hwhm)
        #print('FWHM per', filenames[i], ' : ', hwhm)
    #else:
        #print('Nessun valore sopra la metà massima per', filenames[i])


plt.show()

datas = np.full((len(filenames),3), np.nan, dtype=object)

datas[:,0] = color
datas[:,1] = mean_x
datas[:,2] = std_x


Save = False

if Save:
    path = "/home/marco/Desktop/Uni_anno3/TD/Es_06/acquisizioni/"
    name = "mean_std_wavelenght.txt"
    np.savetxt(path+name, datas, delimiter=',', fmt=['%s', '%.6f', '%.6f'],
           header="Colore_LED, mean, std")


V_soglia = []
s_V_soglia = []
color = []

characteristic_names = np.loadtxt('/home/marco/Desktop/Uni_anno3/TD/Es_06/acquisizioni/curve_caratteristiche_LED/elencofile.txt', dtype=str)

for j in range(len(characteristic_names)):

    plain_name = Path(characteristic_names[j]).stem
    color = np.append(color, plain_name.rsplit('_',1)[-1])
    #print(j, color)

    Ch2, s_Ch2, I, s_I = np.loadtxt('/home/marco/Desktop/Uni_anno3/TD/Es_06/acquisizioni/curve_caratteristiche_LED/'+characteristic_names[j], delimiter=',', unpack=True)

    '''
    fig2 = plt.figure(j+len(filenames), figsize=(10,6), dpi=100, layout='constrained')
    plt.errorbar(Ch2, I, yerr=s_I, xerr=s_Ch2, fmt='.', label='Caratteristica LED '+color)
    plt.title('Caratteristica LED '+color)
    plt.xlabel('V [V]')
    plt.ylabel('I [A]')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    '''
    # Per determinare il voltaggio di soglia

    target = 1e-4
    closest_index = min(range(len(I)), key=lambda i: abs(I[i] - target))
    
    V_soglia = np.append(V_soglia, Ch2[closest_index])
    s_V_soglia = np.append(s_V_soglia, s_Ch2[closest_index])

    #print('Per il LED', color, 'il voltaggio di soglia per I =', target, 'A è V =', Ch2[closest_index], 'V')

data_soglia = np.full((len(filenames),3), np.nan, dtype=object)

data_soglia[:,0] = color
data_soglia[:,1] = V_soglia
data_soglia[:,2] = s_V_soglia

Save1 = False

if Save1:
    path = "/home/marco/Desktop/Uni_anno3/TD/Es_06/acquisizioni/"
    name = "voltaggi_soglia.txt"
    np.savetxt(path+name, data_soglia, delimiter=',',  fmt=['%s', '%.6f', '%.6f'],
           header="Colore_LED, V_soglia, s_V_soglia")


#plot dei dati grezzi invertiti
'''
fig2 = plt.figure(j+len(filenames)+1, figsize=(10,6), dpi=100, layout='constrained')
plt.errorbar(V_soglia, 1/mean_x, yerr=(std_x/mean_x**2), xerr=s_V_soglia, fmt='.', label='Dati sperimentali')
plt.title('Plot 1/lambda in funzione di V_soglia')
plt.ylabel('$\\frac{1}{\lambda}$ [nm$^{-1}$]')
plt.xlabel('V [V]')
#plt.yscale('log')
plt.legend()
plt.grid()
'''
#plot dei dati grezzi
'''
plt.errorbar(1/mean_x, V_soglia, yerr=s_V_soglia, xerr=(std_x/mean_x**2), fmt='.', label='Dati sperimentali')
plt.title('Plot V_soglia in funzione di 1/lambda')
plt.xlabel('$\\frac{1}{\lambda}$ [nm$^{-1}$]')
plt.ylabel('V [V]')
#plt.yscale('log')
plt.legend()
plt.grid()
#plt.savefig('/home/marco/Desktop/Uni_anno3/TD/Es_06/logbook/plot_Vsoglia_vs_1suLambda.png', dpi=300)
'''

#Fit, necessarie incertezze efficaci, ma visto che il modello è lineare scambio x con y ==> plot di 1/lambda vs V_soglia
def linear_fit(x, a, b):
    return a*x + b

popt, pcov = curve_fit(linear_fit, V_soglia, 1/(mean_x*1e-9), sigma=std_x/(mean_x**2*1e-9), p0=(806554, 0.5*806554), absolute_sigma=True)

a1, b1 = popt
sa1, sb1 = np.sqrt(np.diag(pcov))

res = 1/(mean_x*1e-9) - linear_fit(V_soglia, *popt)
chi2 = np.sum((res / (std_x/(mean_x**2*1e-9))) ** 2)
dof = len(V_soglia) - len(popt)

c = 299792458   # speed of light in vacuum in m/s
e = 1.6021766341e-19    # elementary charge in Coulombs

h = e/(a1 * c)   # Planck constant in J*s
s_h = (e/(a1**2* c))*sa1

print(f'Fit results: a = {a1:.6f} ± {sa1:.6f}, b = {b1:.6f} ± {sb1:.6f}')
print(f'Chi-squared/dof: {chi2/dof:.6f}, Degrees of freedom: {np.sqrt(2/dof)}')
print(f'Estimated Planck constant: h = {h:.6e} ± {s_h:.6e} J·s','\n',s_h/h*100,'%')

fig3 = plt.figure(j+len(filenames)+2, figsize=(10,6), dpi=100, layout='constrained')

ax1, ax2 = fig3.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})

ax1.errorbar(V_soglia, 1/(mean_x*1e-9), yerr=std_x/(mean_x**2*1e-9), xerr=s_V_soglia, fmt='.', label='Dati sperimentali')
ax1.plot(V_soglia, linear_fit(V_soglia, *popt), 'r-', label='Fit lineare')
ax1.set_ylabel('$\\frac{1}{\lambda}$ [m$^{-1}$]')
ax1.set_title('Fit di $\\frac{1}{\lambda}$ in funzione di V_soglia')
ax1.legend()
ax1.grid(color='lightgray', ls='dashed')

ax2.errorbar(V_soglia, res, yerr=std_x/(mean_x**2*1e-9), xerr=s_V_soglia, fmt='.', label='Residui')
ax2.axhline(0, color='red', ls='dashed')
ax2.set_xlabel('V [V]')
ax2.set_ylabel('Residui [m$^{-1}$]')
ax2.legend()
ax2.grid(color='lightgray', ls='dashed')

fig3.align_ylabels((ax1, ax2))

savefit = False

if savefit:
    plt.savefig('/home/marco/Desktop/Uni_anno3/TD/Es_06/logbook/plot_fit_Vsoglia_vs_1suLambda.png', dpi=300)
    plt.savefig('/home/marco/Desktop/Uni_anno3/TD/Es_06/logbook/Planck.pdf', dpi=300)

    ax1.set_ylabel('$\\frac{1}{\lambda}$ [nm$^{-1}$]', fontsize='18')
    ax2.set_xlabel('V [V]', fontsize='18')
    ax2.set_ylabel('Residui [nm$^{-1}$]', fontsize='18')
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax1.legend(fontsize='16')
    ax2.legend(fontsize='16')

    plt.savefig('/home/marco/Desktop/Uni_anno3/TD/Es_06/logbook/plot_fit_Vsoglia_vs_1suLambda1.png', dpi=300)

plt.show()



  


