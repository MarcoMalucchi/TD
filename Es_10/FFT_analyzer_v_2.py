import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

#==========
#AVERAGING.
#==========

# --- Configuration ---
path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/parte_alta_y/FFT/'
save_results = False  # Toggle for saving

# 1. Scan for the FFT files
fft_files = sorted([f for f in os.listdir(path) if f.endswith('_FFT.txt')])
num_files = len(fft_files)

if num_files == 0:
    print("No _FFT.txt files found.")
else:
    # 2. Peek at first file for dimensions
    first_data = np.loadtxt(os.path.join(path, fft_files[0]), delimiter=',', skiprows=1)
    num_rows, num_cols = first_data.shape

    # 3. Create the 3D Archive
    archive = np.zeros((num_rows, num_cols, num_files))

    # 4. Fill the Archive
    print(f"Archiving {num_files} files...")
    for i, fname in enumerate(fft_files):
        data = np.loadtxt(os.path.join(path, fname), delimiter=',', skiprows=1)
        if data.shape == (num_rows, num_cols):
            archive[:, :, i] = data

    # 5. Math: Average and Std Dev across the File Dimension (axis 2)
    avg_matrix = np.mean(archive, axis=2)
    std_matrix = np.std(archive, axis=2)

    # 6. Extract Components (Manual FFT indices: 0=Fx, 1=PSDx, 2=Fy, 3=PSDy)
    # Adjust these indices if your column order is different!
    fx = avg_matrix[:, 0]
    psd_x_avg = avg_matrix[:, 1]
    psd_x_std = std_matrix[:, 1]
    
    fy = avg_matrix[:, 2]
    psd_y_avg = avg_matrix[:, 3]
    psd_y_std = std_matrix[:, 3]

# --- GLOBAL PLOT STYLE FOR BEAMER ---
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2
})

# Extract parent folder name (parte_alta_y, etc.)
parent_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))

# --- 7. Visualization: X and Y Comparison ---
fig_avg = plt.figure(figsize=(12, 7))

plt.semilogy(fx, psd_x_avg,
             marker='o',
             linestyle='-',
             linewidth=2,
             markersize=4,
             alpha=0.8,
             label='PSD mediato X')

plt.semilogy(fy, psd_y_avg,
             marker='o',
             linestyle='-',
             linewidth=2,
             markersize=4,
             alpha=0.8,
             label='PSD mediato Y')

plt.xlabel("Frequenza [Hz]")
plt.ylabel(r"PSD [$\mathrm{\frac{g^2}{Hz}}$]")

plt.title(f"{parent_folder}: PSD mediato (N = {num_files})",
          fontweight='bold')

plt.grid(True, which="both", linewidth=0.8, alpha=0.3)

plt.legend()
plt.tight_layout()
plt.show()

# --- 8. Conditional Saving Block ---
if save_results:
    fig_path = os.path.join(path, "PSD_mediato.png")
    fig_avg.savefig(fig_path, dpi=300)
    print(f"Figura salvata in: {fig_path}")

    # Save CSV
    output_data = np.column_stack((fx, psd_x_avg, psd_x_std,
                                   psd_y_avg, psd_y_std))
    header = "Freq_Hz, Mean_PSD_X, Std_PSD_X, Mean_PSD_Y, Std_PSD_Y"
    output_path = os.path.join(path, "AVERAGED_XY_RESONANCE.csv")

    np.savetxt(output_path, output_data,
               delimiter=",",
               header=header,
               comments='')
    print(f"Dati salvati in: {output_path}")
