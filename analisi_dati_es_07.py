import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# forma nomi file: I_d_LED_[colore].txt

filenames = np.loadtxt('/home/marco/Desktop/Uni_anno3/TD/Es_07/acquisizioni/dati_LED/elencofile.txt', dtype=str)

c, s_c = np.loadtxt('/home/marco/Desktop/Uni_anno3/TD/Es_07/acquisizioni/concentrazioni.txt', unpack=True)

d = 1.2e-2  #m    #INSERIRE IL VALORE DEL DIAMETRO DELLA CUVETTA
s_d = 0.05e-2
color = []
co_abs = []
s_co_abs = []

 #fit lineare
def linear(x, m, q):
    return m*x + q

for i in range(len(filenames)):
    
    I, s_I = np.loadtxt('/home/marco/Desktop/Uni_anno3/TD/Es_07/acquisizioni/dati_LED/'+filenames[i], unpack=True)

    plain_name = Path(filenames[i]).stem
    color.append(plain_name.rsplit('_',1)[-1])

    # --- Extract I0 and errors ---
    I0 = I[0]
    s_I0 = s_I[0]

    # --- Compute absorbance and propagated error ---
    Id = I
    s_Id = s_I

    A = -np.log10(Id / I0)
    s_A = np.sqrt( (s_Id/Id)**2 + (s_I0/I0)**2 ) * (1/np.log(10))

    # ===========================================================
    #           INVERTED FIT:  c = M * A + Q
    # ===========================================================

    def linear_inv(A, M, Q):
        return M*A + Q

    popt, pcov = curve_fit(
        linear_inv,
        A,            # x variable (independent)
        c,            # y variable (dependent)
        sigma=s_c,    # errors on dependent variable (c)
        p0=(1, 0),
        absolute_sigma=True
    )

    M, Q = popt
    sM, sQ = np.sqrt(np.diag(pcov))

    # ===========================================================
    #       COMPUTE m AND q FOR THE ORIGINAL MODEL A = m c + q
    # ===========================================================
    m = 1/M
    q = -Q/M

    # Propagate uncertainties
    sm = sM / M**2
    sq = np.sqrt((sQ/M)**2 + (Q*sM/M**2)**2)

    #CONTROLLO SU INCERTEZZE
    musk = (s_c < M*s_A*1e-1)
    #print(f'Controllo su incertezze: {musk.sum()}')

    # ===========================================================
    #               RESIDUALS AND CHI-SQUARE IN INVERTED SPACE
    # ===========================================================

    res_c = c - linear_inv(A, M, Q)
    chi2 = np.sum((res_c / s_c) ** 2)
    dof = len(c) - len(popt)

    
    print(f'Fit results for {filenames[i]}:')
    print(f'M = {M:.6f} ± {sM:.6f}, Q = {Q:.6f} ± {sQ:.6f}')
    print(f'Original slope m = {m:.6f} ± {sm:.6f}')
    print(f'Chi²/dof = {chi2/dof:.6f} +/- {np.sqrt(2/dof)} \n')
    
    # ===========================================================
    #                       ABSORPTION COEFFICIENT
    # ===========================================================
    co_abs.append(m/d)
    s_co_abs.append(np.sqrt(sm**2 * d**2 + s_d**2 * m**2) / d**2)

    # ===========================================================
    #                   INVERTED PLOT (c vs A)
    # ===========================================================
    
    fig = plt.figure(figsize=(10,6), dpi=100, layout='constrained')
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})

    # Plot data (c vs A)
    ax1.errorbar(A, c, yerr=s_c, xerr=s_A, fmt='.', label='Dati sperimentali')

    # Fit line in inverted space
    A_plot = np.linspace(min(A), max(A), 200)
    ax1.plot(A_plot, linear_inv(A_plot, M, Q), 'r-', label='Fit')

    ax1.set_ylabel('concentrazione [a.u.]')
    ax1.set_title('Fit di c in funzione di A per LED ' + color[i])
    ax1.legend()
    ax1.grid(color='lightgray', ls='dashed')

    # Residuals in inverted space
    ax2.errorbar(A, res_c, yerr=s_c, xerr=s_A, fmt='.', label='Residui')
    ax2.axhline(0, color='red', ls='dashed')
    ax2.set_xlabel('Assorbanza [a.u.]')
    ax2.set_ylabel('Residui [a.u.]')
    ax2.legend()
    ax2.grid(color='lightgray', ls='dashed')

    fig.align_ylabels((ax1, ax2))


    savefit = False

    if savefit:
        plt.savefig('/home/marco/Desktop/Uni_anno3/TD/Es_07/logbook/plot_fit_conc_vs_A_'+color[i]+'.png', dpi=300)

        ax1.set_ylabel('concentrazione [a.u.]', fontsize='18')
        ax2.set_xlabel('Assorbanza [a.u.]', fontsize='18')
        ax2.set_ylabel('Residui [a.u]', fontsize='18')
        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', labelsize=16)
        ax2.tick_params(axis='x', labelsize=16)
        ax2.tick_params(axis='y', labelsize=16)
        ax1.legend(fontsize='16')
        ax2.legend(fontsize='16')

        plt.savefig('/home/marco/Desktop/Uni_anno3/TD/Es_07/presentazione/plot_fit_conc_pres_vs_A_'+color[i]+'.png', dpi=300)
    

co_abs = np.array(co_abs)
s_co_abs = np.array(s_co_abs)

L, s_L = np.loadtxt('/home/marco/Desktop/Uni_anno3/TD/Es_06/acquisizioni/mean_std_wavelenght.txt', delimiter=',', unpack=True, usecols=(1,2))

x, abs = np.loadtxt('/home/marco/Desktop/Uni_anno3/TD/Es_07/acquisizioni/spettro_beta-carotene_sample2.txt', delimiter=',', unpack=True)

Matrix = np.column_stack((L, s_L, co_abs, s_co_abs))

#print(s_L)

fig1, ax3 = plt.subplots(1,1, figsize=(10,6), dpi=100, layout='constrained')

ax3.errorbar(L, co_abs*1e-9, yerr = s_co_abs*1e-9, xerr=s_L, fmt='.', label='coeff abs vs $\lambda$')
ax3.plot(x, abs*1e-7, linestyle='-', label='Spettro E160a')
ax3.set_xlabel('$\lambda_{LED}$ [nm]')
ax3.set_ylabel('$c_0 \epsilon_{\lambda}$ [1/nm]')
ax3.legend()
ax3.grid(color='lightgray', ls='dashed')

savefig=False

if savefig:
    path1 = "/home/marco/Desktop/Uni_anno3/TD/Es_07/acquisizioni/"
    np.savetxt(path1+"co_abs_vs_wl.txt", Matrix, delimiter='\t',
               header="#L[nm]  #s_L[nm]  #co_abs[1/m]  #s_co_abs[1/m]")

    plt.savefig('/home/marco/Desktop/Uni_anno3/TD/Es_07/logbook/plot_coeffasb_vs_lambda.png', dpi=300)
    plt.savefig('/home/marco/Desktop/Uni_anno3/TD/Es_07/logbook/SpettroAssorbimento.pdf', dpi=300)

    ax3.set_xlabel('$\lambda_{LED}$ [nm]', fontsize='18')
    ax3.set_ylabel('$c_0 \epsilon_{\lambda}$ [1/nm]', fontsize='18')
    ax3.tick_params(axis='x', labelsize=16)
    ax3.tick_params(axis='y', labelsize=16)
    ax3.legend(fontsize='16')

    plt.savefig('/home/marco/Desktop/Uni_anno3/TD/Es_07/presentazione/plot_coeffasb_vs_lambda_1.png', dpi=300)



plt.show()


