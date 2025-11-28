import numpy as np
import matplotlib.pyplot as plt
import math as m

# Component values
R = 46.758               # ohms
L = 97.318e-3            # H
C = 94.209e-9            # F
w0 = 1 / m.sqrt(L * C)   # rad/s
q = (1 / R) * m.sqrt(L / C)

# Frequency sweep around resonance
w = np.logspace(-3, 3, 1000) * w0
H = 1 / (1 - (w / w0)**2 + 1j * (w / w0) / q)

# Magnitude and phase
mag = 20 * np.log10(np.abs(H))
phase = np.angle(H, deg=True)

# Normalized frequency in log10
x_as = np.log10(w / w0)

# --- Create vertical stacked plots ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex= False)  # sharex ensures x-axis aligned

# Magnitude plot
ax1.plot(x_as, mag, 'b')
ax1.set_ylabel("Magnitude (dB)")
ax1.set_xlabel("log10(ω / ω₀)")
ax1.set_title("Bode Diagram of RLC Low-Pass Filter")
ax1.grid(True, which="both")

# Phase plot
ax2.plot(x_as, phase, 'r')
ax2.set_xlabel("log10(ω / ω₀)")
ax2.set_ylabel("Phase (degrees)")
ax2.grid(True, which="both")

plt.xlim([-3, 3])  # optional
plt.tight_layout()
plt.show()
