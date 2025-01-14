import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import zipfile
import ast

import dataloading


#########################################################
# REMOVE FUNCTIONS IN THIS BLOCK FOR FINAL MODEL
# THESE ARE ONLY FOR PROCESSING THE ZIPPED .CSV FILES
#########################################################

"---------------------------------------------------------------------------------------"
"Functions to load the bubble signals from the zipped .csv files"

# This function now only handles the zip if it contains only one _whole.csv. Edit if more data is added!
def read_whole_csv_from_zip(zip_filename):
    """
    Read CSV files ending with '_whole.csv' from a ZIP file.

    Args:
        zip_filename (str): Inputh path to the ZIP file

    Returns:
        list of pd.DataFrame: List of DataFrames containing the data from the CSV files
    """
    
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        print(f"Opened ZIP file: {zip_filename}")
        for file in zipf.namelist():
            print(f"Found file in ZIP: {file}")
            if file.endswith('_whole.csv'):
                print(f"Reading file: {file}")
                with zipf.open(file) as csvfile:
                    df = pd.read_csv(csvfile, header=0)
                    print(f"Read {file} from {zip_filename}")
    
    # converting the voltage_full with datapoints (now seen as str) to np.array
    df["voltage_full"] = df["voltage_full"].apply(ast.literal_eval)

    return df


# This function now only handles the zip if it contains only one _seperate.csv. Edit if more data is added!
def read_seperate_csv_from_zip(zip_filename):
    """
    Read CSV files ending with '_seperate.csv' from a ZIP file.

    Args:
        zip_filename (str): Name of the ZIP file

    Returns:
        list of pd.DataFrame: List of DataFrames containing the data from the CSV files
    """
    
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        print(f"Opened ZIP file: {zip_filename}")
        for file in zipf.namelist():
            print(f"Found file in ZIP: {file}")
            if file.endswith('seperate.csv'):
                print(f"Reading file: {file}")
                with zipf.open(file) as csvfile:
                    df = pd.read_csv(csvfile, header=0)
                    print(f"Read {file} from {zip_filename}")
    
    # converting the col with voltages (now seen as str) to lists
    for col in ['voltage_entry', 'voltage_exit']:
        df[col] = df[col].apply(ast.literal_eval)
        df[col] = df[col].apply(lambda x: np.array(x))
    
    return df


##################################################
# END OF BLOCK
##################################################

"-----------------------------------------------------------------------------------"
"Actual preprocessing functions"


def scale_time(data, length=None):
    """
    Scales all data to <length> timesteps; adds padding to start and end of the signal.

    Args: 
        data: Numpy array with dimension [samples, datapoints]
        length: length all samples will be changed into. length should be longer than the longest data signal.
                if not fixed, it is 10 timesteps longer than the longest data signal.

    Returns:
        Numpy array with dimension [samples, datapoints] with <length> datapoints.
    """
    # if type(data) != type(np.zeros(3)):
    #     data = np.array(data)
    
    if length is None:
        length = max([len(sample) for sample in data]) + 10
        print(f"No sample length was given, so length was automatically fixed at {length}")

    # if data.ndim != 2:
    #     raise ValueError("Data should be a 2D numpy array with dimensions [samples, datapoints]. (Hint: try to add .tolist() to input)")
    
    scaled_data = []
    
    for sample in data:
        if len(sample) > length:
            raise ValueError("There is a sample longer than the specified length")
        
        # Calculate padding
        total_padding = length - len(sample)
        pad_before = total_padding // 2
        pad_after = total_padding - pad_before
        
        # Pad the sample
        sample = np.array(sample)
        padded_sample = np.pad(sample, (pad_before, pad_after), mode='edge')
        scaled_data.append(padded_sample)
    
    return np.array(scaled_data)



"----------------------------------------------------------------------------------"
"Loading the data from dataloading.py (from meta and bin files etc.)"

# Commented because the whole bubble is too big to store in a numpy array, but partial storing should work

# bin_file, metadata_file, evt_file, run_name = dataloading.find_files(R"C:\Users\Silke\Documents\GitHub\CapstoneAI\Data")

# metadata = dataloading.get_metadata(metadata_file)
# coef1 = metadata["channelCoef1"]
# coef2 = metadata["channelCoef2"]

# voltage_data, bubbles = dataloading.get_bubbles(bin_file, coef1, coef2, w=2500)

# bubble_data = []
# for idx, (tA0, tA, tA1, tE0, tE, tE1) in enumerate(bubbles):
#     voltage_bubble = voltage_data[tA0:tE1 + 1]
#     bubble_data.append(voltage_bubble)

# voltage_whole_2d = bubble_data
# whole_scaled_bubbles = scale_time(voltage_whole_2d, length=None)
# plt.plot(np.arange(len(whole_scaled_bubbles[0])), whole_scaled_bubbles[0])
# plt.show()

"----------------------------------------------------------------------------------"
"Calling the functions using the .csv files"

# seperate_bubbles = read_seperate_csv_from_zip("all_bubbles.zip")
# print("\n")
# print(seperate_bubbles.head())

# voltage_exit_2d = np.array(seperate_bubbles["voltage_exit"].tolist())
# scaled_bubbles = scale_time(voltage_exit_2d, length=None)
# #lengths = [len(sample) for sample in scaled_bubbles]

# plt.plot(np.arange(len(scaled_bubbles[0])), scaled_bubbles[0])
# plt.show()


## Whole bubbles: takes too long to load from CSV file. Loading from dataloading.py is more efficient.

# whole_bubbles = read_whole_csv_from_zip("all_bubbles.zip")
# voltage_whole_2d = np.array(whole_bubbles["voltage_full"].tolist())
# lengths = [len(sample) for sample in voltage_whole_2d]
# lengths = set(lengths)
# print(lengths)

# whole_scaled_bubbles = scale_time(voltage_whole_2d, length=None)
# plt.plot(np.arange(len(whole_scaled_bubbles[0])), whole_scaled_bubbles[0])
# plt.show()