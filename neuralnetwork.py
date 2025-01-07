# Imports 
import torch

# Dummy dataset(replace when preprocessing is done)

# Models
def model_self(): # Stijn, implement LSTM model here
    return None

def model_pretrained(): # Max, implement pretrained model here
    return None

def model_fourier(): # Jan-Paul, implement Fourier model here
    return None

def train_model(model = 'Self'):
    if model == 'Self':
        model = model_self()
    elif model == 'Pretrained':
        model = model_pretrained()
    else:
        model = model_fourier()
    return None

# evaluation block
