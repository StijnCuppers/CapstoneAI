import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import dataloading 


"---------------------------------------------------------------------------------------"
"Functions to load the bubble signals from the zipped .csv files"

# change to your data path
data_path = R"C:\Users\Silke\Documents\GitHub\CapstoneAI\Data\all_bubbles.zip"

def read_whole_csv_from_zip(zip_filename):
    """
    Read CSV files ending with '_whole.csv' from a ZIP file.

    Args:
        zip_filename (str): Inputh path to the ZIP file

    Returns:
        list of pd.DataFrame: List of DataFrames containing the data from the CSV files
    """
    dataframes = []
    
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        for file in zipf.namelist():
            if file.endswith('_whole.csv'):
                with zipf.open(file) as csvfile:
                    df = pd.read_csv(csvfile)
                    dataframes.append(df)
                    print(f"Read {file} from {zip_filename}")

    return dataframes

def read_seperate_csv_from_zip(zip_filename):
    """
    Read CSV files ending with '_seperate.csv' from a ZIP file.

    Args:
        zip_filename (str): Name of the ZIP file

    Returns:
        list of pd.DataFrame: List of DataFrames containing the data from the CSV files
    """
    dataframes = []
    
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        for file in zipf.namelist():
            if file.endswith('_seperate.csv'):
                with zipf.open(file) as csvfile:
                    df = pd.read_csv(csvfile)
                    dataframes.append(df)
                    print(f"Read {file} from {zip_filename}")

    return dataframes


"-----------------------------------------------------------------------------------"
"Actual preprocessing functions"


def scale_time(data, length):
    """
    Scales all data to <length> timesteps; adds padding to start and end of the signal.

    Args: 
        data: Numpy array with dimension [samples, datapoints]
        length: length all samples will be changed into. length should be longer than the longest data signal.

    Returns:
        Numpy array with dimension [samples, datapoints] with <length> datapoints.
    """

    return




