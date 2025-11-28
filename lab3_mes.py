import numpy as np
import matplotlib.pyplot as plt
from pydwf import DwfLibrary
import time

# --------------------------------------------
# Initialize WaveForms / AD3
# --------------------------------------------

dwf = DwfLibrary()
devices = dwf.devices()   # List available devices
if len(devices) == 0:
    raise RuntimeError("No Analog Discovery device found!")

hdwf = devices[0].open()


print("Opening device...")
hdwf = dwf.device.open(-1)
if hdwf is None:
    raise RuntimeError("No Analog Discovery device found!")

# --------------------------------------------
# Sweep settings
# --------------------------------------------
f_start = 10             # Hz
f_stop  = 100000         # Hz
Npoints = 60             # Number of frequency points
amplitude = 1.0          # AWG amplitude in volts

freqs = np.logspace(np.log10(f_start), np.log10(f_stop), Npoints)

mag_response = []
phase_response = []

# --------------------------------------------
# Configure AnalogIn (Oscilloscope)
# --------------------------------------------
sample_rate = 100_000_000  # 100 MS/s
dwf.analogIn.frequencySet(hdwf, sample_rate)

# Enable channels 1 and 2
dwf.analogIn.channelEnableSet(hdwf, 0, True)
dwf.analogIn.channelRangeSet(hdwf, 0, 5.0)
dwf.analogIn.channelEnableSet(hdwf, 1, True)
dwf.analogIn.channelRangeSet(hdwf, 1, 5.0)

# --------------------------------------------
# Main sweep loop
# --------------------------------------------
for f in freqs:

    # Set AWG
    dwf.analogOut.nodeEnableSet(hdwf, 0, 0, True)
    dwf.analogOut.nodeFunctionSet(hdwf, 0, 0, dwf.funcSine)
    dwf.analogOut.nodeAmplitudeSet(hdwf, 0, 0, amplitude)
    dwf.analogOut.nodeFrequencySet(hdwf, 0, 0, f)
    dwf.analogOut.configure(hdwf, 0, True)

    time.sleep(0.05)  # settling

    # Set scope record length
    cycles = 8
    record_length = cycles / f
    dwf.analogIn.recordLengthSet(hdwf, record_length)

    # Start acquisition
    dwf.analogIn.configure(hdwf, False, True)

    # Wait for done
    while True:
        status = dwf.analogIn.status(hdwf, True)
        if status == dwf.DwfState.Done:
            break

    # Read channels
    vin  = np.array(dwf.analogIn.statusData(hdwf, 0))
    vout = np.array(dwf.analogIn.statusData(hdwf, 1))

    # Time vector
    t = np.arange(len(vin)) / sample_rate

    # Complex amplitude using projection
    w = 2 * np.pi * f
    exp_sig = np.exp(-1j * w * t)

    H_in  = np.sum(vin  * exp_sig)
    H_out = np.sum(vout * exp_sig)

    A_in  = 2 * abs(H_in)  / len(vin)
    A_out = 2 * abs(H_out) / len(vout)

    mag_response.append(A_out / A_in)

    phase = np.angle(H_out) - np.angle(H_in)
    phase_response.append(np.degrees(phase))

# --------------------------------------------
# Finished
# --------------------------------------------
dwf.device.close(hdwf)

# --------------------------------------------
# Plotting Bode diagram
# --------------------------------------------
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.semilogx(freqs, 20*np.log10(mag_response), "b")
plt.grid(True)
plt.ylabel("Magnitude (dB)")
plt.title("Analog Discovery 3 - Bode Plot (Measured)")

plt.subplot(2,1,2)
plt.semilogx(freqs, phase_response, "r")
plt.grid(True)
plt.ylabel("Phase (deg)")
plt.xlabel("Frequency (Hz)")

plt.tight_layout()
plt.show()
