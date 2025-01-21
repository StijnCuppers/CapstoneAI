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

main_folder = R"C:\Users\TUDelft\Desktop\NEW_DATA"

def read_seperate_csv_from_zip(main_folder):
    """
    Read all CSV files from the ZIP file located in the main folder.

    Args:
        main_folder (str): Path to the main folder containing the ZIP file.

    Returns:
        pd.DataFrame: A single concatenated DataFrame containing data from all CSV files in the ZIP.
    """
    zip_filepath = os.path.join(main_folder, "All_bubbles.zip")  

    with zipfile.ZipFile(zip_filepath, 'r') as zipf:
        print(f"Opened ZIP file: {zip_filepath}")
        for file in zipf.namelist():
            if file.endswith('Combined_bubbles.csv'): 
                print(f"Reading file: {file}")
                with zipf.open(file) as csvfile:
                    df = pd.read_csv(csvfile, header=0, delimiter=";")

                # Convert the VoltageOut column to lists
                if "VoltageOut" in df.columns:
                    df["VoltageOut"] = df["VoltageOut"].apply(ast.literal_eval)
                    
    return df

df = read_seperate_csv_from_zip(main_folder)
print(df.head())

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
    if length is None:
        length = max([len(sample) for sample in data]) + 10
        print(f"No sample length was given, so length was automatically fixed at {length}")
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


def frame_waves(data, length=500):
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
        cropped_data = np.array([sample[:length] for sample in data])
    
    return cropped_data


def valid_velo_data(data):
    """
    Extracts only the data with valid velocities (VeloOut != -1).

    Args:
        data (pd.DataFrame): DataFrame containing VoltageOut and VeloOut columns.

    Returns:
        x (np.ndarray): Array of VoltageOut data with valid velocities.
        y (np.ndarray): Array of VeloOut labels with valid velocities.
    """
    valid = data[data["VeloOut"] != -1]
    x = valid["VoltageOut"].apply(ast.literal_eval).tolist()
    y = valid["VeloOut"].astype(float).tolist()
    return np.array(x), np.array(y)


def valid_velo_data_cropped(data, length=500):
    """
    Extracts the data with valid velocities and crops VoltageOut signals to the specified length.

    Args:
        data (pd.DataFrame): DataFrame containing VoltageOut and VeloOut columns.
        length (int): Target length for cropping signals.

    Returns:
        x (np.ndarray): Array of cropped VoltageOut data with valid velocities.
        y (np.ndarray): Array of VeloOut labels with valid velocities.
    """
    valid = data[data["VeloOut"] != -1]
    x = np.array([np.array(ast.literal_eval(sample)[:length]) for sample in valid["VoltageOut"]])
    y = valid["VeloOut"].astype(float).tolist()
    return x, np.array(y)


def random_flip(data, labels, chance, random_seed=None):
    """
    Randomly duplicates some samples and performs a horizontal flip on them.

    Args:
        data (np.ndarray): Array of voltage signals.
        labels (np.ndarray): Array of corresponding labels.
        chance (float): Fraction of data (between 0 and 1) to flip and duplicate.
        random_seed (int): Optional random seed for reproducibility.

    Returns:
        np.ndarray: Augmented voltage signals.
        np.ndarray: Augmented labels.
    """
    if not (0. < chance < 1.):
        raise ValueError("Chance should be between 0 and 1 (inclusive)")
    if len(data) != len(labels):
        raise ValueError("data and labels should be the same length")

    if random_seed is not None:
        np.random.seed(random_seed)

    n = int(chance * len(data))
    random_indices = np.random.randint(0, len(data), n)
    flipped_data = np.array([data[i][::-1] for i in random_indices])
    flipped_labels = labels[random_indices]
    return np.concatenate([data, flipped_data]), np.concatenate([labels, flipped_labels])


def random_noise(data, labels, chance, noise_level=0.005, random_seed=None):
    """
    Duplicates some samples and adds noise to them.

    Args:
        data (np.ndarray): Array of voltage signals.
        labels (np.ndarray): Array of corresponding labels.
        chance (float): Fraction of data (between 0 and 1) to augment with noise.
        noise_level (float): Standard deviation of noise.

    Returns:
        np.ndarray: Augmented voltage signals.
        np.ndarray: Augmented labels.
    """
    if not (0. < chance < 1.):
        raise ValueError("Chance should be between 0 and 1 (inclusive)")
    if len(data) != len(labels):
        raise ValueError("data and labels should be the same length")

    if random_seed is not None:
        np.random.seed(random_seed)

    n = int(chance * len(data))
    random_indices = np.random.randint(0, len(data), n)
    noisy_data = np.array([data[i] + np.random.normal(0, noise_level, len(data[i])) for i in random_indices])
    noisy_labels = labels[random_indices]
    return np.concatenate([data, noisy_data]), np.concatenate([labels, noisy_labels])


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
    Flattens the data distribution by augmenting and duplicating data in less frequent bins.

    Args:
        X (np.ndarray): Voltage signals.
        y (np.ndarray): Velocity labels.
        bins (int): Number of bins for the velocity labels.
        scaling_factor (float): Scaling factor to control augmentation intensity.

    Returns:
        np.ndarray: Augmented voltage signals.
        np.ndarray: Augmented velocity labels.
    """
    hist, bin_edges = np.histogram(y, bins=bins)
    bin_indices = np.digitize(y, bin_edges[:-1])
    max_freq = np.max(hist)
    factors = (max_freq / hist) * scaling_factor
    factors[hist == max_freq] = 1
    augmented_X = X.copy()
    augmented_y = y.copy()
    for i, (x_val, y_val) in enumerate(zip(X, y)):
        bin_idx = bin_indices[i] - 1
        for _ in range(int(factors[bin_idx]) - 1):
            if np.random.rand() < 0.5:
                x_new, y_new = random_noise([x_val], [y_val], chance=1, noise_level=noise)
            else:
                x_new, y_new = random_flip([x_val], [y_val], chance=1)
            augmented_X = np.concatenate([augmented_X, x_new])
            augmented_y = np.concatenate([augmented_y, y_new])
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