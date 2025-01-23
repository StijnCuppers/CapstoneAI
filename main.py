### Start user input ###
path_to_data = [r'C:\Users\slcup\Documents\Aerospace Engineering\Minor\Capstone\Capstone data\Data']
path_to_output = r'C:\Users\slcup\Documents\Aerospace Engineering\Minor\Capstone\Capstone data\Data'
### End user input ###

# Libary imports
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
device = 'cpu'
from sklearn.model_selection import train_test_split
import time

# Function imports
from advanced_dataloading import process_folder
from advanced_preprocessing import frame_waves, valid_velo_data
from models import load_scalers, load_models, LSTMModel, GRUModel

# Load the models and scalers
start_time = time.time()
gru1, gru2, lstm = load_models()
gru1.eval(), gru2.eval(), lstm.eval()
feature_scaler, target_scaler, feature_scaler2, target_scaler2 = load_scalers()

# Loop over all zips
for file_path in path_to_data:
    # Load and preprocess the input
    df = process_folder(file_path, plot=True, labels=False)
    X_gru1 = frame_waves(df['VoltageOut'], length=150, jump=0)[0]
    X_gru1_scaled = torch.tensor(feature_scaler.transform(X_gru1)[...,np.newaxis], dtype=torch.float32)   
    X_gru2 = frame_waves(df['VoltageOut'], length=150, jump=0)[0]
    X_gru2_scaled = torch.tensor(feature_scaler.transform(X_gru2)[...,np.newaxis], dtype=torch.float32)
    X_lstm = frame_waves(df['VoltageOut'], length=150, jump=0)[0]
    X_lstm_scaled = torch.tensor(feature_scaler.transform(X_lstm)[...,np.newaxis], dtype=torch.float32)

    # Make predictions
    with torch.no_grad():  
        y_gru1_scaled = gru1(X_gru1_scaled)
        y_gru2_scaled = gru2(X_gru2_scaled)
        y_lstm_scaled = lstm(X_lstm_scaled)
    y_gru1 = target_scaler.inverse_transform(y_gru1_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_gru2 = target_scaler.inverse_transform(y_gru2_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_lstm = target_scaler.inverse_transform(y_lstm_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()

    y_pred = ((y_lstm+y_gru1+y_gru2)/3).flatten()
    outcome_df = pd.DataFrame({"predictions model 1": y_gru1, "predictions model 2": y_gru2, "predictions model 3": y_lstm, "final prediction": y_pred})
    outcome_df['Standard deviation'] = outcome_df[["predictions model 1", "predictions model 2", "predictions model 3"]].std(axis=1)
    outcome_df['Standard deviation %'] = outcome_df['Standard deviation'] / outcome_df['final prediction'] * 100
 
print(outcome_df.head(10))
outcome_df.to_csv(f'{path_to_output}/velocity_predictions.csv', index=False)
end_time = time.time()


# Evaluation metrics (remove '''...''' if interested)
'''
valid_bubbles_ai = len(outcome_df[outcome_df['Standard deviation %'] < 10])/len(outcome_df) * 100
valid_bubbles_boring_software = len(valid_velo_data(df)[0])/len(df) * 100

X_velo, y_velo = valid_velo_data(df)
X_velo = frame_waves(X_velo, length=150, jump=0)[0]
X_velo_scaled = torch.tensor(feature_scaler.transform(X_velo)[...,np.newaxis], dtype=torch.float32)
with torch.no_grad():  
        y_gru1_scaled_velo = gru1(X_velo_scaled)
        y_gru2_scaled_velo = gru2(X_velo_scaled)
        y_lstm_scaled_velo = lstm(X_velo_scaled)
y_gru1_velo = target_scaler.inverse_transform(y_gru1_scaled_velo.detach().cpu().numpy().reshape(-1, 1)).flatten()
y_gru2_velo = target_scaler.inverse_transform(y_gru2_scaled_velo.detach().cpu().numpy().reshape(-1, 1)).flatten()
y_lstm_velo = target_scaler.inverse_transform(y_lstm_scaled_velo.detach().cpu().numpy().reshape(-1, 1)).flatten()
y_pred_velo = ((y_lstm_velo+y_gru1_velo+y_gru2_velo)/3).flatten()
outcome_df_valid = pd.DataFrame({"predictions model 1": y_gru1_velo, "predictions model 2": y_gru2_velo, "predictions model 3": y_lstm_velo, "final prediction": y_pred_velo})
outcome_df_valid['Standard deviation'] = outcome_df_valid[["predictions model 1", "predictions model 2", "predictions model 3"]].std(axis=1)
outcome_df_valid['Standard deviation %'] = outcome_df_valid['Standard deviation'] / outcome_df_valid['final prediction'] * 100
valid_test_results = outcome_df_valid[(outcome_df_valid["Standard deviation"]/outcome_df_valid["final prediction"]) <= 0.1]

filtered_outcome_df = outcome_df[outcome_df['Standard deviation %'] < 10]
average_percentage_std = filtered_outcome_df['Standard deviation %'].mean()

print(f"Percentage found valid bubbles (uncertainty < 10%) with speed difference <10% from truth:  {len(valid_test_results) / (len(outcome_df_valid)) * 100:.4f} %")
print(f'Percentage AI found valid bubbles (uncertainty < 10%): {valid_bubbles_ai:.4f} % vs M2 analyzer: {valid_bubbles_boring_software:.4f} %, improvement: {((valid_bubbles_ai - valid_bubbles_boring_software)/valid_bubbles_boring_software)*100:.4f} %')
print(f'Model uncertainty (average uncertainty of valid bubbles): {average_percentage_std:.4f} % with {len(filtered_outcome_df) / len(outcome_df_valid) * 100} % of the labled samples')
'''
print(f"Time to execute: {end_time - start_time} seconds")
