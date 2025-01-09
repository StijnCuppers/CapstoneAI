# Imports 
import torch
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

# Models
def model_pretrained(): # Max, implement pretrained model here
    return None

def model_fourier(): # Jan-Paul, implement Fourier model here
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
        with torch.no_grad():  # Disable gradient computation for evaluation
            outputs, (hidden, _) = lstm(X_test)
            y_pred = fc(hidden[-1]).squeeze()
    elif model == 'pretrained':
        model = model_pretrained() # Max implement 
        y_pred = None
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
plot_results()