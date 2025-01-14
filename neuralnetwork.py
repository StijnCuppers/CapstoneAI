# Imports 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Dummy dataset
def load_dummy_data():
    X_train = torch.tensor([[[i] for i in range(10)] for _ in range(100)], dtype=torch.float32)
    X_test = torch.tensor([[[i] for i in range(10)] for _ in range(100)], dtype=torch.float32)
    X_val = torch.rand(100, 10)
    y_train = torch.tensor([sum(seq[-1]) for seq in X_train], dtype=torch.float32)
    y_test = torch.tensor([sum(seq[-1]) for seq in X_test], dtype=torch.float32)
    y_val = torch.randint(0, 2, (100,))
    return X_train, X_test, X_val, y_train, y_test, y_val
    
X_train, X_test, X_val, y_train, y_test, y_val = load_dummy_data()

print(X_train.shape)

# Models
def model_CNN(data=None, epochs=4, batch_size=10, learning_rate=0.001, loss_fn=torch.nn.MSELoss()):
    if data:
        # Implement once data preprocessing is done
        pass
    else:
        X_train, X_test, X_val, y_train, y_test, y_val = load_dummy_data()

    X_train = X_train.permute(0, 2, 1)
    X_test = X_test.permute(0, 2, 1)
    X_val = X_val.permute(0, 2, 1)

    CNN = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  
        nn.ReLU(), 
        nn.MaxPool1d(kernel_size=2, stride=2), 
        nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Flatten(),  
        nn.Linear(32 * (10 // 4), 64), 
        nn.ReLU(),
        nn.Linear(64, 1)  
    )
    optimizer = torch.optim.Adam(CNN.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0
        CNN.train()
        for i in range(0, len(X_train), batch_size):
            input = X_train[i:i+batch_size]
            targets = y_train[i:i+batch_size]

            predictions = CNN(input).squeeze()

            loss = loss_fn(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (len(X_train) // batch_size):.4f}")

    return CNN


def model_fourier(data=None, epochs=4, batch_size=10, learning_rate=0.001, loss_fn=torch.nn.MSELoss()): # Jan-Paul, implement Fourier model here
    if data:
        pass # Implement once preprocess is done
    else:
        X_train, X_test, X_val, y_train, y_test, y_val = load_dummy_data()


    fs = 20833.3  # Sampling frequency in kHz
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_result), d=1/fs)

    input_size = len(frequencies)


    model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
    


    return None

def model_self(data = None, epochs = 2, batch_size = 10, learning_rate = 0.001, loss_fn = torch.nn.MSELoss()):
    if data:
        pass # Implement once preprocess is done
    else:
        X_train, X_test, X_val, y_train, y_test, y_val = load_dummy_data()
    
    lstm = torch.nn.LSTM(1, 50, 2, batch_first = True)
    fc = torch.nn.Linear(50, 1)
    optimizer = torch.optim.Adam(list(lstm.parameters()) + list(fc.parameters()), lr=learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            input = X_train[i:i+batch_size]
            targets = y_train[i:i+batch_size]
            outputs, (hidden, _) = lstm(input)
            predictions = fc(hidden[-1])

            loss = loss_fn(predictions.squeeze(), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (len(X_train)) // batch_size:.4f}")
    return lstm, fc

# Evaluation block
def plot_results(model = 'self', data=None):
    if data:
        # implement when datapreprocess is done
        pass
    else:
        X_train, X_test, X_val, y_train, y_test, y_val = load_dummy_data()
    if model == 'self':
        lstm, fc = model_self()
        with torch.no_grad():  
            outputs, (hidden, _) = lstm(X_test)
            y_pred = fc(hidden[-1]).squeeze()
    elif model == 'CNN':
        CNN = model_CNN(input_channels=1, output_size=1, timesteps=10) 
        CNN.eval()          
        with torch.no_grad():  
            y_pred = CNN(X_test).squeeze()
    else:
        model = model_fourier() # Jan-Paul implement here
        y_pred = None
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.numpy(), label='Actual', color='blue')
    plt.plot(y_pred.numpy(), label='Predictions', color='red', linestyle='dashed')
    plt.legend()
    plt.title(f'Predictions vs Actual, MSE={MSE}')
    plt.xlabel('Test Data Index')
    plt.ylabel('Output Value')
    plt.show()

# Main
#plot_results()