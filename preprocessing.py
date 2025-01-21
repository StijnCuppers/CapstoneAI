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
# valid_velo_data_cropped: combines frame_waves and valid_velo_data
# random_flip: duplicates and randomly flips some data
# random_noise: duplicates and randomly adds noise to some data
# bin_data: bins all y labels as data (does not regard X data) > part of flatten_data_distribution
# calculate_duplication_factors: calculates how to scale data in bins > part of flatten_data_distribution
# duplicate_and_augment_data: duplicates and augments data based on bin frequency > part of flatten_data_distribution
# flatten_data_distribution > flattens the data distribution according to bin sizes, by augmenting and duplicating data



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


def frame_waves(data, mode, length=500, labels=None, n_crops=1, jump=0):
    """
    Function that crops to the waves. Only works for data that is only v_out or v_in signals (mode).

    Args:
        data: numpy array or list with the voltage data of the bubbles
        labels: labels of the data. If n_crops=2, you must put in labels!
        mode: two options; "in" for data of bubble entry or "out" for data of bubble exit. 
        length: amount of timesteps of the cropped part. Standard value is set at 500.
        n_crops: can be 1 or 2. For 1, makes one zoomed-in sample per bubble.
                For 2, picks two parts of the wave signal (so final output doubles in size)
        jump: Recommended value is an integer between 0-500 (for small lengths). 
                Gives the amount of steps away from the frame edge 
                (to obtain clearer waves, at the cost of possibly overshooting the bubble time frame)

    Output:
        cropped_data: Numpy array with the cropped voltages, dimension [#samples, length]
        labels: labels put in and duplicated to match the cropped_data. If n_crop=1,
        do not save the labels! Instead call the function using: cropped_data, _ = frame_waves(...)
        
    """
    if n_crops not in [1, 2]:
        raise ValueError("n_crops should be either 1 or 2")
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # if mode is in, grabs the last <length> datapoints
    if mode == "in":
        if n_crops == 1:
            cropped_data = np.array([sample[-length:-jump] for sample in data])
        if n_crops == 2:
            if labels is None:
                raise ValueError("labels must be given in frame_waves when n_crops=2!")
            cropped_data = np.array([sample[-(jump+length):-jump] for sample in data])
            cropped_data2 = np.array([sample[-(jump+length+(length//2)):-(jump+length//2)] for sample in data])
            cropped_data = np.concatenate([cropped_data, cropped_data2])
            labels = np.concatenate([labels, labels])

    # if mode is out, grabs the first <length>
    if mode == "out":
        if n_crops == 1:
            cropped_data = np.array([sample[jump:(jump+length)] for sample in data])
        if n_crops == 2:
            cropped_data = np.array([sample[jump:(jump+length)] for sample in data])
            cropped_data2 = np.array([sample[(jump+length//2):(jump+length + length//2)] for sample in data])
            cropped_data = np.concatenate([cropped_data, cropped_data2])
            labels = np.concatenate([labels, labels])
    
    return cropped_data, labels


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


def valid_velo_data_cropped(data, mode, length=500, jump=0):
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

    if mode == "or" and jump != 0:
        print("Note: the jump argument does not work yet for the 'or' mode. Now jump is automatically set at 0.")
    
    if mode == "in":
        valid = data[data["VeloIn"] != -1]
        x = frame_waves(valid["voltage_entry"].tolist(), mode, length=length, jump=jump)
        y = valid["VeloIn"].astype(float).tolist()

    if mode == "out":
        valid = data[data["VeloOut"] != -1]
        x = frame_waves(valid["voltage_exit"].tolist(), mode, length=length, jump=jump)
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


def random_flip(data, labels, chance, random_seed=None):
    """
    Randomly duplicates some samples and performs a horizontal flip on them.

    Args:
        data: 2D Numpy array (or list) with features of the samples
        labels: 1D Numpy array (or list) with labels per sample
        chance: fraction of data (between 0 and 1) that will get flipped and duplicated
        random_seed: optional random seed (integer)

    Output:
        x: Numpy array with the duplicated/transformed samples appended
        y: Numpy array with the duplicated labels appended
    """
    if not (0. < chance < 1.):
        raise ValueError("Chance should be between 0 and 1 (inclusive)")
    if len(data) != len(labels):
        raise ValueError("data and labels should be the same length")

    if isinstance(data, list):
        data = np.array(data)
    if isinstance(labels, list):
        labels = np.array(labels)

    # set random seed if applicable
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # amount of images to be flipped:
    n = int(chance*len(data))

    # creating an array of length n with random numbers 
    random_list = np.random.randint(0, len(data), n)

    x_new = data
    y_new = labels
    for rand in random_list:
        duplicate = data[rand][::-1]
        duplicate_label = labels[rand]
        x_new = np.concatenate([x_new, [duplicate]])
        y_new = np.append(y_new, duplicate_label)

    return x_new, y_new


def random_noise(data, labels, chance, noise_level=0.005, random_seed=None):
    """
    Duplicates some samples and adds noise to them.
    
    Args:
        data: 2D Numpy array (or list) with features of the samples
        labels: 1D Numpy array (or list) with labels per sample
        chance: fraction of data (between 0 and 1) that will get augmented and duplicated
        noise_level: standard deviation in the noise. Must be non-negative. 

    Output:
        x: Numpy array with the duplicated/transformed samples appended
        y: Numpy array with the duplicated labels appended
    """

    if not (0. < chance < 1.):
        raise ValueError("Chance should be between 0 and 1 (inclusive)")
    if len(data) != len(labels):
        raise ValueError("data and labels should be the same length")

    if isinstance(data, list):
        data = np.array(data)
    if isinstance(labels, list):
        labels = np.array(labels)

    # set random seed if applicable
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # amount of images to be flipped:
    n = int(chance*len(data))

    # creating an array of length n with random numbers 
    random_list = np.random.randint(0, len(data), n)

    x_new = data
    y_new = labels
    for rand in random_list:
        duplicate = data[rand] + np.random.normal(0, noise_level, data[rand].shape)
        duplicate_label = labels[rand]
        x_new = np.concatenate([x_new, [duplicate]])
        y_new = np.append(y_new, duplicate_label)

    return x_new, y_new


def bin_data(y, bins):
    """
    Makes bins based on y data (velocities)

    Args: 
        y: numpy array with labels
        bins: number of bins

    Output:
        hist: array that contains counts of datapoints in each bin.
        bin_indices: array that indicates which bin each data point in y belongs to.
    
    """
    hist, bin_edges = np.histogram(y, bins=bins)
    bin_indices = np.digitize(y, bin_edges[:-1])
    return hist, bin_indices


def calculate_duplication_factors(hist, scale_factor=0.5):
    """
    Calculates the factors that each bin should be duplicated with. Less frequent bins will get duplicated more.

    Args:
        hist: array with samples per bin
        scale_factor: determines how much the distribution will be flattened. 
                        1=almost completely flat, 0=no flattening

    Output:
        array with the factors per bin
        
    """
    max_freq = np.max(hist)
    factors = np.zeros_like(hist, dtype=float)
    # preventing division by 0
    non_zero_indices = hist > 0

    # scaling factor so less frequent data does not get fully duplicated 10 times.
    factors[non_zero_indices] = (max_freq / hist[non_zero_indices]) * scale_factor
    # No duplication for the most frequent bin
    factors[hist == max_freq] = 1 

    return factors


def duplicate_and_augment_data(X, y, bin_indices, factors, noise=0.005):
    """
    Duplicates/augments the data per bin, scaled with the size of each bin 
    (smaller bin -> more duplication)

    Args:
        X: X data
        y: y data
        bin_indices: array with which datapoints correspond to which bins (from bin_data)

    Output:
        augmented_X: lengthened and partly augmented X data
        augmented_y: lenghthened y data of the augmented_X data
    
    """
    augmented_X = X.copy()
    augmented_y = y.copy()
    for i, (x_value, y_value) in enumerate(zip(X, y)):
        bin_idx = bin_indices[i] - 1
        factor = factors[bin_idx]
        for _ in range(int(factor) - 1):
            if np.random.rand() < 0.5:
                x_new, y_new = random_noise([x_value], [y_value], chance=1, noise_level=noise)
            else:
                x_new, y_new = random_flip([x_value], [y_value], chance=1)
            augmented_X = np.concatenate([augmented_X, x_new])
            augmented_y = np.concatenate([augmented_y, y_new])
    return augmented_X, augmented_y


def flatten_data_distribution(X, y, bins, scaling_factor=0.5, noise=0.005):
    """
    Combines the functions to flatten the distribution (by augmenting data).
    
    Args:
        X: X data
        y: y data
        bins: amount of bins
        scaling_factor: factor that prevents data from becoming all bins becoming the most frequent

    Output:
        augmented_X: lengthened and partly augmented X data
        augmented_y: lenghthened y data of the augmented_X data
    """
    hist, bin_indices = bin_data(y, bins)
    factors = calculate_duplication_factors(hist, scale_factor=scaling_factor)
    augmented_X, augmented_y = duplicate_and_augment_data(X, y, bin_indices, factors, noise=noise)
    return augmented_X, augmented_y

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