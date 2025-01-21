### Start user input ###
path_to_zips = ['all_bubbles.zip']
### End user input ###

# Libary imports
import pandas as pd

# Function imports
from advanced_dataloading import process_folder
from advanced_preprocessing import frame_waves

# Load the model and scalers
gru1 = None
gru2 = None
lstm = None
feature_scaler = None
target_scaler = None

# Loop over all zips
for file_path in path_to_zips:
    # Load and preprocess the input
    big_bubbles_df = process_folder(r'C:\Users\slcup\Documents\Aerospace Engineering\Minor\Capstone\Capstone data\Data', plot=True, labels=False)
    #X_gru1 = frame_waves(df['voltage_exit'], 'out', length=150, jump=900)
    #X_gru2 = frame_waves(df['voltage_exit'], 'out', length=250, jump=0)
    #X_lstm = frame_waves(df['voltage_exit'], 'out', length=150, jump=900)

#print(df)