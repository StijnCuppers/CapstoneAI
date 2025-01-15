# imports
from preprocessing import read_seperate_csv_from_zip
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# format data
def format_data_lstm(filename, focus='out'):
    '''
    Format the data for use in LSTM model.

    Args:
        filename (str): Name of the ZIP file containing the data
        focus (str): 'in', 'out' or 'both' to specify which signal to use

    Returns:
        
    
    '''
    data = read_seperate_csv_from_zip(filename)
    scaler = StandardScaler()
    if focus == 'in':
        data = data['voltage_entry']

    elif focus == 'out':
        data = data[['voltage_exit', 'VeloOut']]
        X = np.stack(data["voltage_exit"].values)
        X = X[..., np.newaxis]
        y = data["VeloOut"].values


    else:
        data = None # Implement when both signals are used
    print(f'X shape: {X.shape}, y shape: {y.shape}')
    print(X[:5])
    print(y[:5])
format_data_lstm('all_bubbles.zip', focus='out')