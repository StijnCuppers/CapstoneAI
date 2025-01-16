# imports
from preprocessing import read_seperate_csv_from_zip
from preprocessing import valid_velo_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch

# standard values
seed = 0
train_split = 0.7

# format data
def format_data_lstm(filename, focus='out'):
    '''
    Format the data for use in LSTM model.

    Args:
        filename (str): Name of the ZIP file containing the data
        focus (str): 'in', 'out' or 'both' to specify which signal to use

    Returns:
        X_train (np.array): Scaled input data for training
        X_test (np.array): Scaled input data for testing
        y_train (np.array): Scaled target data for training
        y_test (np.array): Scaled target data for testing
    '''

    data = read_seperate_csv_from_zip(filename)
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    if focus == 'in':
        X, y = valid_velo_data(data, 'in')

    elif focus == 'out':
        X, y = valid_velo_data(data, 'out')
        
    else:
        data = None # Implement when both signals are used
    X = X[..., np.newaxis]

    X_scaled = feature_scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, train_size=train_split, random_state=seed)
    return X_train, X_test, y_train, y_test, target_scaler


def train_lstm(X_train, y_train, epochs=5, learning_rate=0.01, hidden_size=50, loss_fn=torch.nn.MSELoss()):
    '''
    Train the LSTM model, all model parameters are hardcoded but I will change them later to be easier changed

    Args:
        X_train (np.array): Scaled input data for training
        y_train (np.array): Scaled target data for training
        epochs (int): Number of epochs to train the model
        learning_rate (float): Learning rate for the optimizer
        hidden_size (int): Number of hidden units in the LSTM layer
        loss_fn (torch.nn.Module): Loss function to use
    
    Returns:
        model (torch.nn.Module): Trained LSTM model
        fc (torch.nn.Module): Fully connected layer for the output
    '''
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    input_size = X_train.shape[-1]
    lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
    fc = torch.nn.Linear(hidden_size, 1)
    optimizer = torch.optim.Adam(list(lstm.parameters()) + list(fc.parameters()), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs, _ = lstm(X_train)
        predictions = fc(outputs[:, -1, :])

        loss = loss_fn(predictions.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    return lstm, fc


def train_lstm_grid_search(
    X_train, y_train, X_val, y_val, 
    epochs=5, 
    param_grid=None, 
    loss_fn=torch.nn.MSELoss()
):
    '''
    Train the LSTM model using grid search for hyperparameter tuning.

    Args:
        X_train (np.array): Scaled input data for training
        y_train (np.array): Scaled target data for training
        X_val (np.array): Scaled input data for validation
        y_val (np.array): Scaled target data for validation
        epochs (int): Number of epochs to train the model
        param_grid (dict): Dictionary containing hyperparameters to search
        loss_fn (torch.nn.Module): Loss function to use

    Returns:
        best_model (torch.nn.Module): Best LSTM model
        best_fc (torch.nn.Module): Best fully connected layer
        best_params (dict): Best hyperparameters
    '''
    if param_grid is None:
        param_grid = {
            "learning_rate": [0.1, 0.01, 0.001],
            "hidden_size": [5, 10, 20, 30],
        }

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    best_loss = float("inf")
    best_model = None
    best_fc = None
    best_params = None

    for lr in param_grid["learning_rate"]:
        for hidden_size in param_grid["hidden_size"]:
            print(f"Training with learning_rate={lr}, hidden_size={hidden_size}")
            input_size = X_train.shape[-1]
            lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
            fc = torch.nn.Linear(hidden_size, 1)
            optimizer = torch.optim.Adam(list(lstm.parameters()) + list(fc.parameters()), lr=lr)

            for epoch in range(epochs):
                # Training
                lstm.train()
                optimizer.zero_grad()
                outputs, _ = lstm(X_train)
                predictions = fc(outputs[:, -1, :])
                train_loss = loss_fn(predictions.squeeze(), y_train)
                train_loss.backward()
                optimizer.step()

            # Validation
            lstm.eval()
            with torch.no_grad():
                val_outputs, _ = lstm(X_val)
                val_predictions = fc(val_outputs[:, -1, :])
                val_loss = loss_fn(val_predictions.squeeze(), y_val).item()

            print(f"Validation Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = lstm
                best_fc = fc
                best_params = {"learning_rate": lr, "hidden_size": hidden_size}

    print(f"Best Parameters: {best_params}, Best Validation Loss: {best_loss:.4f}")
    return best_model, best_fc, best_params


def test_lstm(X_test, y_test, model, fc, target_scaler ):
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    with torch.no_grad():
        outputs, _ = model(X_test)
        predictions = fc(outputs[:, -1, :]).squeeze()

    mae = torch.mean(torch.abs(predictions - y_test))
    print(f"Scaled Test Set Mean Absolute Error: {mae.item():.4f}")
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_original = target_scaler.inverse_transform(predictions.numpy().reshape(-1, 1)).flatten()

    # Calculate MAE on the original scale
    mae_original = np.mean(np.abs(predictions_original - y_test_original))
    print(f"Test Set MAE (Original Scale): {mae_original:.4f}")
    print(f'Test set mean (Original Scale): {np.mean(y_test_original)}')


X_train, X_test_val, y_train, y_test_val, target_scaler = format_data_lstm('all_bubbles.zip', focus='out')
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, train_size=0.5, random_state=seed)
model, fc, params = train_lstm_grid_search(X_train, y_train, X_val, y_val, epochs=5)
test_lstm(X_test, y_test, model, fc, target_scaler)

