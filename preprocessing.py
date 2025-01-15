import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import zipfile
import ast

import dataloading

########################################################
# TABLE OF CONTENTS

# read_whole_csv_from_zip (not working at the moment, files too large)
# read_seperate_csv_from_zip
# scale_time: makes sure all bubble arrays are equal size (! only needed for whole bubbles)
# frame_waves: crops data so it zooms in on the waves
# valid_velo_data: returns data and labels with only valid velocities

########################################################


#########################################################
# REMOVE FUNCTIONS IN THIS BLOCK BELOW FOR FINAL MODEL
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
                    df = pd.read_csv(csvfile, header=0, delimiter=";")
                    print(f"Read {file} from {zip_filename}")
    
    # converting the col with voltages (now seen as str) to lists
    for col in ["voltage_entry", "voltage_exit"]:
        df[col] = df[col].apply(ast.literal_eval)
    
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


def frame_waves(data, mode, length=500):
    """
    Function that crops to the waves. Only works for data that is only v_out or v_in signals (mode).

    Args:
        data: numpy array or list with the voltage data of the bubbles
        mode: two options; "in" for data of bubble entry or "out" for data of bubble exit. 
        length: amount of timesteps of the cropped part. Standard value is set at 500.

    Output:
        Numpy array with the cropped voltages, dimension [#samples, length]
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # if mode is in, grabs the last <length> datapoints
    if mode == "in":
        cropped_data = np.array([sample[-length:] for sample in data])

    # if mode is out, grabs the first <length
    if mode == "out":
        cropped_data = np.array([sample[:length] for sample in data])
    
    return cropped_data


def valid_velo_data(data, mode):
    """
    Extracts the data with valid velocities. Only works for pandas DataFrames right now.

    Args:
        data: enter the dataframe with bubble voltages and labels
        mode: "in" gets all bubbles with a valid VeloIn, "out" with a valid VeloOut, 
                "or" with either valid VeloIn or VeloOut

    Output:
        x: Numpy array with voltage data of all valid bubbles. Either only in or out signal.
        y: Numpy array with labels of all valid bubbles
    """
    if mode not in ["in", "out", "or"]:
        print("Error: choose valid mode. Choose from ['in', 'out', 'or']")
        return
    
    if mode == "in":
        valid = data[data["VeloIn"] != -1]
        x = valid["voltage_entry"].tolist()
        y = valid["VeloIn"].astype(float).tolist()

    if mode == "out":
        valid = data[data["VeloOut"] != -1]
        x = valid["voltage_exit"].tolist()
        y = valid["VeloOut"].astype(float).tolist()

    if mode == "or":
        x = []
        y = []

        # first seperating everything that has either/or both VeloIn and VeloOut 
        valid = data[(data["VeloOut"] != -1) | (data["VeloIn"]!= -1)]
        
        # In case there's two velocities, returns the minimum velocity and data
        for _, row in valid.iterrows():
            if row["VeloIn"] != -1 and row["VeloOut"] != -1:
                if row["VeloIn"] < row["VeloOut"]:
                    x.append(row["voltage_entry"])
                    y.append(row["VeloIn"])
                else:
                    x.append(row["voltage_exit"])
                    y.append(row["VeloOut"])
            elif row["VeloIn"] != -1:
                x.append(row["voltage_entry"])
                y.append(row["VeloIn"])
            else:
                x.append(row["voltage_exit"])
                y.append(row["VeloOut"])
        

    
    return np.array(x), np.array(y)


def valid_velo_data_cropped(data, mode, length=500):
    """
    Extracts the data with valid velocities and combines it with the frame_waves function. 
    Only works for pandas DataFrames right now.

    Args:
        data: enter the dataframe with bubble voltages and labels
        mode: "in" gets all bubbles with a valid VeloIn, "out" with a valid VeloOut, 
                "or" with either valid VeloIn or VeloOut
        length: The amount of timesteps the output will be. Standard value is set at 500.

    Output:
        x: Numpy array with cropped voltage data of all valid bubbles. Either only in or out signal.
        y: Numpy array with labels of all valid bubbles
    """
    if mode not in ["in", "out", "or"]:
        print("Error: choose valid mode. Choose from ['in', 'out', 'or']")
        return
    
    if mode == "in":
        valid = data[data["VeloIn"] != -1]
        x = frame_waves(valid["voltage_entry"].tolist(), mode, length=length)
        y = valid["VeloIn"].astype(float).tolist()

    if mode == "out":
        valid = data[data["VeloOut"] != -1]
        x = frame_waves(valid["voltage_exit"].tolist(), mode, length=length)
        y = valid["VeloOut"].astype(float).tolist()
    
    if mode == "or":
        x = []
        y = []
        # first seperating everything in 
        valid = data[(data["VeloOut"] != -1) | (data["VeloIn"]!= -1)]
        
        for _, row in valid.iterrows():
            if row["VeloIn"] != -1 and row["VeloOut"] != -1:
                if row["VeloIn"] < row["VeloOut"]:
                    x.append(row["voltage_entry"][-length:])
                    y.append(row["VeloIn"])
                else:
                    x.append(row["voltage_exit"][:length])
                    y.append(row["VeloOut"])
            elif row["VeloIn"] != -1:
                x.append(row["voltage_entry"][-length:])
                y.append(row["VeloIn"])
            else:
                x.append(row["voltage_exit"][:length])
                y.append(row["VeloOut"])
        

    return np.array(x), np.array(y)

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
# # #lengths = [len(sample) for sample in scaled_bubbles]

# plt.plot(np.arange(len(scaled_bubbles[0])), scaled_bubbles[0])
# print(scaled_bubbles[0])
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