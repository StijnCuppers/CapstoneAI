import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt



# Generate a sample signal (sum of two sine waves)
fs = 20833.3333  # Sampling frequency in kHz
T = 0.240048004  # Duration in milliseconds
t = np.linspace(0, T, int(fs*T), endpoint=True)  # Time vector

print(len(t))
f1, f2, f3 = 1009000, 2490088, 3700000  # Frequencies of the sine waves
signal = np.sin(2 * np.pi * f1/1000 * t) + 0.5 * np.sin(2 * np.pi * f2/1000 * t) +0.25 * np.sin(2* np.pi *f3/1000 * t)


fs = 20833.3  # Sampling frequency in kHz
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(fft_result), d=1/fs)




def plot_domains(t, signal, frequencies, fft_result):
    # Plot the results
    plt.figure(figsize=(10, 6))

    # Time-domain signal
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title("Time-Domain Signal")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")

    # Frequency-domain signal
    plt.subplot(2, 1, 2)
    plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(frequencies)//2])
    plt.title("Frequency-Domain Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()

plot_domains(t, signal, frequencies, fft_result)
