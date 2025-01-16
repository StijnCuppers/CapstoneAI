import numpy as np
import matplotlib.pyplot as plt
import torch
from preprocessing import read_seperate_csv_from_zip, valid_velo_data_cropped



# Generate a sample signal (sum of two sine waves)
fs = 20833.3333  # Sampling frequency in kHz
T = 0.0960000002 # Duration in milliseconds

t = torch.linspace(0, T, int(fs*T))  # Time vector
print(len(t))

#f1, f2, f3 = 100900, 249008, 370000  # Frequencies of the sine waves
#signal = torch.sin(2 * np.pi * f1/1000 * t) + 0.5 * np.sin(2 * np.pi * f2/1000 * t) +0.25 * np.sin(2* np.pi *f3/1000 * t)

df = read_seperate_csv_from_zip("all_bubbles.zip")

#print(df.head(10))
#signal = df['voltage_exit'][4]


def load_and_transform(df):
    signal = df['voltage_exit'][4]
    fs = 20833.3333  # Sampling frequency in kHz
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_result), d=1/fs)
    return frequencies, fft_result



def transform_signal(signal, fs, mode='abs'):
    fft_result = torch.fft.fft(signal)
    frequencies = torch.fft.fftfreq(len(fft_result), d=1/fs)
    if mode == 'abs':
        return frequencies, torch.abs(fft_result)
    elif mode == 'complex':
        return frequencies, fft_result
    else:
        return 'Invalid mode'

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
    plt.plot(frequencies[1:len(frequencies)//2], np.abs(fft_result)[1:len(frequencies)//2])
    plt.title("Frequency-Domain Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()

def bandpass_filter(signal, fs, f_low, f_high):
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_result), d=1/fs)
    fft_result[(frequencies < f_low)] = 0
    fft_result[(frequencies > f_high)] = 0
    return np.fft.ifft(fft_result)


class FourierModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None

    def compile(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, X_train, y_train, epochs, batch_size):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(X_train), batch_size):
                inputs = X_train[i:i + batch_size]
                targets = y_train[i:i + batch_size]
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(X_train)}')

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X)


length = 2000

X, y = valid_velo_data_cropped(df, 'out', length=length)
X = torch.Tensor(X)
y = torch.Tensor(y)

frequencies, X_fourier = transform_signal(X, fs)


X_fourier_train = X_fourier[:int(0.7*len(X))]
X_fourier_test = X_fourier[int(0.7*len(X)):]
y_train = y[:int(0.7*len(y))]
y_test = y[int(0.7*len(y)):]



#signal = bandpass_filter(signal, fs, 0, 500)
frequencies, fft_result = transform_signal(X[12], fs)

model = FourierModel(input_size=length, hidden_size=100, output_size=1)
model.compile(learning_rate=0.001)

model.train(X_fourier_train, y_train, epochs=10, batch_size=10)

predicitons = model.predict(X_fourier_test)

print(np.abs(y_test - predicitons.squeeze()))
print(np.abs(y_test - predicitons.squeeze()).shape)
print(np.abs(y_test - predicitons.squeeze()).dtype)
print(predicitons.dtype)
print(y_test.dtype)
print(predicitons.squeeze().shape)
print(y_test.shape)




mae = torch.mean(np.abs(y_test - predicitons.squeeze()))

print(mae)

#plot_domains(t, X[12], frequencies, fft_result)


