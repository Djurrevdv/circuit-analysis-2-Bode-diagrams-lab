import numpy as np
import matplotlib.pyplot as plt

# Component values
R = 998.63               # ohms
C = 1.0214e-6            # farads
w0 = 1 / (R * C)         # rad/s
print(f"Corner frequency w0 = {w0:.2f} rad/s")

# Frequency sweep (logarithmic)
w = np.logspace(-3, 3, 1000) * w0  # sweep from 0.001*w0 to 1000*w0

# Transfer function H(jw) = 1 / (1 + j(w / w0))
H = 1 / (1 + 1j * (w / w0))

# Magnitude and phase
mag = 20 * np.log10(np.abs(H))       # dB
phase = np.angle(H, deg=True)        # degrees

# Normalized frequency in log10
x_as = np.log10(w / w0)

# --- Plot Bode magnitude ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex= False)

ax1.plot(x_as, mag, 'b')
ax1.set_ylabel("Magnitude (dB)")
ax1.set_xlabel("log10(ω / ω₀)")
ax1.set_title("Bode Diagram of RC Low-Pass Filter")
ax1.grid(True, which="both")

# Phase plot
ax2.plot(x_as, phase, 'r')
ax2.set_xlabel("log10(ω / ω₀)")
ax2.set_ylabel("Phase (degrees)")
ax2.grid(True, which="both")

plt.xlim([-3, 3])  # optional
plt.tight_layout()
plt.show()
