import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

#==========
#AVERAGING.
#==========

# --- Configuration ---
path = '/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/parte_bassa_y/FFT/'
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

# --- 7. Visualization: X and Y Comparison (Dots Only) ---
    plt.figure(figsize=(12, 7))
    
    # Plot X-Axis using markers ('o') without a connecting line
    # markersize=2 keeps the plot from looking too crowded
    plt.semilogy(fx, psd_x_avg, marker='o', linestyle='-', color='blue', markersize=2, alpha=0.7, label='Avg PSD X')
    
    # Plot Y-Axis using markers ('o') without a connecting line
    plt.semilogy(fy, psd_y_avg, marker='o', linestyle='-', color='green', markersize=2, alpha=0.7, label='Avg PSD Y')

    # Formatting the plot
    plt.xlabel(r"Frequency [Hz]")
    plt.ylabel(r"PSD [$\mathrm{\frac{g^2}{Hz}}$]")
    plt.title(f"Ensemble Average Comparison (N={num_files} files)")
    
    # Add a subtle grid for better readability of log values
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # --- 8. Conditional Saving Block ---
    if save_results:
        # Saving Frequency, Mean, and StdDev for both X and Y
        output_data = np.column_stack((fx, psd_x_avg, psd_x_std, psd_y_avg, psd_y_std))
        header = "Freq_Hz, Mean_PSD_X, Std_PSD_X, Mean_PSD_Y, Std_PSD_Y"
        output_path = os.path.join(path, "AVERAGED_XY_RESONANCE.csv")
        
        np.savetxt(output_path, output_data, delimiter=",", header=header, comments='')
        print(f"Results saved to: {output_path}")


#=========================
# MULTIPLE LORENTZIAN FIT.
#=========================

# --- 1. Define the Lorentzian Functions ---
def single_lorentzian(f, f0, A, gamma):
    return A / (1 + ((f - f0) / gamma)**2)

def multi_lorentzian(f, *params):
    """Sum of N Lorentzians based on the number of parameters provided"""
    result = np.zeros_like(f)
    # Each Lorentzian has 3 params: f0, A, gamma
    for i in range(0, len(params), 3):
        f0, A, gamma = params[i:i+3]
        result += single_lorentzian(f, f0, A, gamma)
    return result

# --- 2. Load and MASK the Data ---
data = np.loadtxt(os.path.join(path, "AVERAGED_XY_RESONANCE.csv"), delimiter=",", skiprows=1)

# Create a mask for frequencies between 0.5 and 20 Hz
# (Starting at 0.5 Hz avoids the DC/Low-freq artifacts)
mask = (data[:, 0] >= 0.5) & (data[:, 0] <= 20)

f_data = data[mask, 0]
psd_data = data[mask, 3]   # Change to 1 for X, 3 for Y
sigma_data = data[mask, 4] # Change to 2 for X, 4 for Y

# --- 3. Peak Finding (now only looks in the 0.5-20Hz range) ---
threshold = 5e-3
peaks, _ = find_peaks(psd_data, height=threshold, distance=10)
# Build initial guesses [f0, A, gamma] for each peak found
initial_guesses = []
for p in peaks:
    f0_guess = f_data[p]
    A_guess = psd_data[p]
    gamma_guess = 0.02*f0_guess #based on the hypothesis that we know the damping ratio, here assumed to be of 2%
    print(f"f0_guess:", f0_guess, "A_guess:", A_guess, "gamma_guess:", gamma_guess)
    #gamma_guess = 0.5 # Starting guess for width in Hz
    initial_guesses.extend([f0_guess, A_guess, gamma_guess])

# --- 4. Perform the Fit with Optimization ---
if len(initial_guesses) > 0:
    lower_bounds = []
    upper_bounds = []
    
    # We must iterate through the guesses we just made to ensure 
    # the boundaries match the parameters 1:1
    for i in range(len(peaks)):
        f0_guess = initial_guesses[i*3]
        A_guess = initial_guesses[i*3 + 1]
        g_guess = initial_guesses[i*3 + 2]

        # 1. Frequency Bounds: Keep the peak within 0.5Hz of where find_peaks saw it
        f_min = f0_guess - 0.5
        f_max = f0_guess + 0.5
        
        # 2. Amplitude Bounds: Must be positive, max 3x the detected height
        a_min = A_guess * 0.2  # Allow it to shrink if peaks overlap
        a_max = A_guess * 3.0
        
        # 3. Gamma (Width) Bounds: 
        # Crucial: HWHM must be > 0. A 5Hz width is very broad for structural resonance.
        g_min = 0.001 
        g_max = 5.0

        # Append to the global bounds lists
        lower_bounds.extend([f_min, a_min, g_min])
        upper_bounds.extend([f_max, a_max, g_max])
    
    try:
        # Convert lists to tuples as required by scipy
        popt, pcov = curve_fit(
            multi_lorentzian, f_data, psd_data, 
            p0=initial_guesses, 
            sigma=sigma_data, 
            absolute_sigma=True, 
            bounds=(lower_bounds, upper_bounds),
            maxfev=50000 
        )
        
        # ... rest of the plotting code ...
        
        fit_curve = multi_lorentzian(f_data, *popt)
        
        # Calculate one-standard-deviation errors from the covariance matrix
        perr = np.sqrt(np.diag(pcov))
        
        print(f"Successfully fitted {len(peaks)} peaks.")
        for i in range(0, len(popt), 3):
            f0, A, gamma = popt[i:i+3]
            f0_err = perr[i]
            # Damping ratio zeta = gamma / f0
            zeta = gamma / f0
            print(f"Peak {i//3 + 1}: f0 = {f0:.3f} ± {f0_err:.3f} Hz, Zeta = {zeta:.4f}")

    except RuntimeError as e:
        print(f"Fit failed to converge: {e}")
        fit_curve = None
else:
    print("No peaks found above the threshold!")
    fit_curve = None

# --- 5. Plot the Result ---
plt.figure(figsize=(12, 7))
plt.semilogy(f_data, psd_data, 'o', markersize=2, color='green', alpha=0.3, label='Data')
if fit_curve is not None:
    plt.semilogy(f_data, fit_curve, color='red', linewidth=2, label='Multi-Lorentzian Fit')
    # Mark the identified peaks
    plt.semilogy(f_data[peaks], psd_data[peaks], "x", color='black', label='Fitted Peaks')

plt.axhline(y=threshold, color='black', linestyle='--', alpha=0.5, label='Threshold')
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.legend()
plt.show()