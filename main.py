### Start user input ###
path_to_zips = [r'C:\Users\slcup\Documents\Aerospace Engineering\Minor\Capstone\Capstone data\Data']
### End user input ###

# Libary imports
import pandas as pd
import torch
import numpy as np

# Function imports
from advanced_dataloading import process_folder
from advanced_preprocessing import frame_waves, valid_velo_data
from models import load_scalers, load_models
from models import LSTMModel

# Load the models and scalers
gru1, gru2, lstm = load_models()
gru1.eval(), gru2.eval(), lstm.eval()
feature_scaler, target_scaler, feature_scaler2, target_scaler2 = load_scalers()

# Loop over all zips
for file_path in path_to_zips:
    # Load and preprocess the input
    df = process_folder(file_path, plot=True, labels=True)
    X_gru1 = frame_waves(df['VoltageOut'], length=150, jump=900)[0]
    X_gru1_scaled = torch.tensor(feature_scaler.transform(X_gru1)[...,np.newaxis], dtype=torch.float32)   
    X_gru2 = frame_waves(df['VoltageOut'], length=250, jump=0)[0]
    X_gru2_scaled = torch.tensor(feature_scaler2.transform(X_gru2)[...,np.newaxis], dtype=torch.float32)
    X_lstm = frame_waves(df['VoltageOut'], length=150, jump=900)[0]
    X_lstm_scaled = torch.tensor(feature_scaler.transform(X_lstm)[...,np.newaxis], dtype=torch.float32)

    # Make predictions
    with torch.no_grad():  
        y_gru1_scaled = gru1(X_gru1_scaled)
        y_gru2_scaled = gru2(X_gru2_scaled)
        y_lstm_scaled = lstm(X_lstm_scaled)
    y_gru1 = target_scaler.inverse_transform(y_gru1_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_gru2 = target_scaler2.inverse_transform(y_gru2_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_lstm = target_scaler.inverse_transform(y_lstm_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()

    y_pred = ((y_lstm+y_gru1+y_gru2)/3).flatten()
    outcome_df = pd.DataFrame({"predictions model 1": y_gru1, "predictions model 2": y_gru2, "predictions model 3": y_lstm, "model prediction": y_pred})
    outcome_df['Standard deviation'] = outcome_df[["predictions model 1", "predictions model 2", "predictions model 3"]].std(axis=1)
    outcome_df['Standard deviation %'] = outcome_df['Standard deviation'] / outcome_df['model prediction'] * 100
 
print(outcome_df.head(10))

# Evaluation metrics
valid_bubbles = len(outcome_df[outcome_df['Standard deviation %'] < 10])/len(outcome_df) * 100
print(f'Percentage of valid bubbles: {valid_bubbles}%')

