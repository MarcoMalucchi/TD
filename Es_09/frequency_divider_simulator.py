import numpy as np
from scipy.signal import square
import matplotlib.pyplot as plt

# Parameters
f = 5                 # frequency in Hz
duration = 1.0        # seconds
sampling_rate = 1000  # Hz (samples per second)

# Time array
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Square wave
x = square(2 * np.pi * f * t)

# Plot
plt.plot(t, x)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("5 Hz Square Wave")
plt.show()
