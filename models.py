import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import dataloading
import preprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUModel(torch.nn.Module):
    # input should be [batch_size, number of timesteps, number of features]
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h_0)
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(torch.nn.Module):
    # input should be [batch_size, number of timesteps, number of features]
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_size = hidden_size  # Ensure hidden_size is correctly defined
        self.fc = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  # Selecting the last time step's output
        out = self.fc(out)
        
        return out


# Maybe change the function so it automatically finds the feature scaler and model files?
def get_results(input_folder_path, model_folder_path, jump=900, 
                plot_hist=False, n_bins=20):
    """Plots the final results as an average of three models.
    
    Args:
        input_folder_path: path to folder containing the .bin file
        model_folder_path: path that leads to the model folder where the three models (.h5 files) and scalers (.pkl files) are stored.
        jump: parameter for the preprocessing. Not recommended to change it, unless you get indexing errors (then lower the value).
        plot_hist: Will plot a histogram if set to True
        n_bins: Amount of bins in the histogram

    Output:
        dataframe with predictions of 3 models, final prediction, true prediction and standard deviation
    """
    if jump != 900:
        print(f"Note: jump is usually set at 900 because it leads to less noisy data and therefore better predictions.")
        print(f"However, jump is now set at {jump}. Only change the value of jump to a lower value if you get indexing errors!")
    # TO DO:
    # Function that finds path to files in the folder
    # Segment bubbles from .bin file
    # X_test = preprocessing.frame_waves(X_test, "out", length=150, jump=jump) # TO DO
    
    # Loading the models

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_1 = GRUModel(1, 20, num_layers=2)
    model_1.load_state_dict(torch.load(model_folder_path+'\GRU_lr0.01_H20_norm2_nlayer2.h5', map_location=torch.device('cpu')))

    model_2 = torch.load(model_folder_path+"\full_LSTM_lr0.008_H30_norm3_nlayer2.h5", map_location=torch.device('cpu'))

    model_3 = GRUModel(1, 20, num_layers=2)
    model_3.load_state_dict(torch.load(model_folder_path+"\GRU_lr0.02_H20_norm3_nlayer2.h5", map_location=torch.device('cpu')))

    # Import StandardScalers with scaling parameters of the training set
    with open(model_folder_path+"\feature_scaler.pkl", 'rb') as file: # maybe adjust path
        loaded_feature_scaler = pickle.load(file)
        
    with open(model_folder_path+'\target_scaler.pkl', 'rb') as file: # maybe adjust path
        loaded_target_scaler = pickle.load(file)

    # Making model predictions
    model_1.eval()
    model_2.eval()
    model_3.eval()
    X_test = loaded_feature_scaler.transform(X_test)
    X_test = torch.tensor(X_test[...,np.newaxis], dtype=torch.float32)
    model_1.to(device)
    model_2.to(device)
    model_3.to(device)

    with torch.no_grad():
        predictions_1 = model_1(X_test)
        predictions_2 = model_2(X_test)
        predictions_3 = model_3(X_test)
        
    # Returning to original scale
    predictions_1 = loaded_target_scaler.inverse_transform(predictions_1.reshape(-1,1)).flatten()
    predictions_2 = loaded_target_scaler.inverse_transform(predictions_2.reshape(-1,1)).flatten()
    predictions_3 = loaded_target_scaler.inverse_transform(predictions_3.reshape(-1,1)).flatten()

    # Making the dataframe with the mean
    combined_df = {"predictions model 1": predictions_1, "predictions model 2": predictions_2, 
                   "predictions model 3": predictions_3}
    combined_df["final prediction"] = combined_df[["predictions model 1", 
                                                   "predictions model 2", 
                                                   "predictions model 3"]].mean(axis=1)
    combined_df["standard deviation"] = combined_df[["predictions model 1", 
                                                   "predictions model 2", 
                                                   "predictions model 3"]].std(axis=1)

    if plot_hist:
        # Plotting a histogram
        predictions = combined_df["final prediction"].tolist()
        plt.hist(predictions, bins=n_bins, alpha=0.5, edgecolor="black", label="prediction")
        plt.xlabel("velocity")
        plt.legend()
        plt.show()
    
    return combined_df
