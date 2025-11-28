import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------
# RC Circuit Parameters
# --------------------------
R = 998.63               # ohms
C = 1.0244e-6            # farads
w0 = 1 / (R * C)         # rad/s
print(f"Corner frequency w0 = {w0:.2f} rad/s")

# Theoretical frequency sweep (Hz)
f_theo = np.logspace(0, 5, 1000)  # from 1 Hz to 100 kHz
w_theo = 2 * np.pi * f_theo

# Transfer function H(jw) = 1 / (1 + j(w / w0))
H = 1 / (1 + 1j * (w_theo / w0))

# Magnitude and phase
mag_theo = 20 * np.log10(np.abs(H))
phase_theo = np.angle(H, deg=True)
x_theo = np.log10(w_theo / w0)

# --------------------------
# Load Measured Data
# --------------------------
csv_file = 'measurementsRC.csv'  # replace with your CSV file
data = pd.read_csv(csv_file)

f_meas = data.iloc[:,0].values
mag_meas = data.iloc[:,1].values
phase_meas = data.iloc[:,2].values
w_meas = 2 * np.pi * f_meas
x_meas = np.log10(w_meas / w0)

# --------------------------
# Closest theoretical points
# --------------------------
mag_theo_closest = []
phase_theo_closest = []

for w in w_meas:
    idx = np.argmin(np.abs(w_theo - w))
    mag_theo_closest.append(mag_theo[idx])
    phase_theo_closest.append(phase_theo[idx])

mag_theo_closest = np.array(mag_theo_closest)
phase_theo_closest = np.array(phase_theo_closest)

# --------------------------
# Absolute Error
# --------------------------
mag_error = np.abs(mag_meas - mag_theo_closest)       # in dB
phase_error = np.abs(phase_meas - phase_theo_closest) # in degrees

avg_mag_error = np.mean(mag_error)
avg_phase_error = np.mean(phase_error)

print(f"Average Magnitude Error: {avg_mag_error:.2f} dB")
print(f"Average Phase Error: {avg_phase_error:.2f} °")

# --------------------------
# All-in-one Figure with 3 subplots
# --------------------------
fig, axs = plt.subplots(3, 1, figsize=(5, 6), sharex=True)

# Magnitude comparison
axs[0].plot(x_theo, mag_theo, label='Theoretical', color='blue', linewidth=2)
axs[0].plot(x_meas, mag_meas, label='Measured', color='red', linestyle='-', linewidth=1)
axs[0].set_ylabel("Magnitude (dB)")
axs[0].set_title("Bode Diagram Measurements RC Circuit")
axs[0].grid(True, which='both')
axs[0].legend()

# Phase comparison
axs[1].plot(x_theo, phase_theo, label='Theoretical', color='blue', linewidth=2)
axs[1].plot(x_meas, phase_meas, label='Measured', color='red', linestyle='-', linewidth=1)
axs[1].set_ylabel("Phase (°)")
axs[1].grid(True, which='both')
axs[1].legend()

# Error plot
axs[2].plot(x_meas, mag_error, label='Magnitude Error (dB)', linestyle='-', color='red')
axs[2].plot(x_meas, phase_error, label='Phase Error (°)', linestyle='-', color='blue')
axs[2].set_xlabel("log10(ω / ω₀)")
axs[2].set_ylabel("Absolute Error")
axs[2].grid(True, which='both')
axs[2].legend()

plt.tight_layout()
plt.show()

