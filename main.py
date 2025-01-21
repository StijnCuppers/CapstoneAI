### Start user input ###
path_to_zips = []
### End user input ###

# Libary imports
import pandas as pd

# Function imports
from advanced_dataloading import process_folder
from advanced_preprocessing import frame_waves
from models import load_scalers, load_models
from models import LSTMModel

# Load the models and scalers
gru1, gru2, lstm = load_models()
feature_scaler, target_scaler = load_scalers()

# Loop over all zips
for file_path in path_to_zips:
    # Load and preprocess the input
    df = process_folder(file_path, plot=True, labels=False)
    #X_gru1 = frame_waves(df['voltage_exit'], 'out', length=150, jump=900)
    #X_gru2 = frame_waves(df['voltage_exit'], 'out', length=250, jump=0)
    #X_lstm = frame_waves(df['voltage_exit'], 'out', length=150, jump=900)

#print(df)