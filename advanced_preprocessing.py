from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import random
from sklearn.model_selection import KFold

import os
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import ast


def random_noise(data, labels, chance, noise_level=0.005, random_seed=None):
    if not (0. <= chance):
        raise ValueError("Chance should be larger than 0")
    if len(data) != len(labels):
        raise ValueError("data and labels should be the same length")

    if isinstance(data, list):
        data = np.array(data)
    if isinstance(labels, list):
        labels = np.array(labels)

    # set random seed if applicable
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # amount of images to be augmented:
    n = int(chance * len(data))

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


def random_flip(data, labels, chance, random_seed=None):
    if not (0. <= chance):
        raise ValueError("Chance should be larger than 0")
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
    n = int(chance * len(data))

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


def read_seperate_csv_from_zip(zip_filename):
    """
    Read CSV files ending with '_seperate.csv' from a ZIP file.

    Args:
        zip_filename (str): Name of the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(zip_filename, header=0, delimiter=';')
    # converting the column with voltages (now seen as str) to lists
    df["VoltageOut"] = df["VoltageOut"].apply(ast.literal_eval)
    return df


def valid_velo_data(data):
    """
    Extracts the data with valid velocities. Only works for pandas DataFrames right now.

    Args:
        data: DataFrame with bubble voltages and labels.

    Returns:
        x: Numpy array with voltage data of all valid bubbles.
        y: Numpy array with labels of all valid bubbles.
    """
    valid = data[data["VeloOut"] != -1]
    x = valid["VoltageOut"].tolist()
    y = valid["VeloOut"].astype(float).tolist()
    return np.array(x), np.array(y)


def frame_waves(data, length=500, labels=None, n_crops=1):
    """
    Function that crops to the waves.

    Args:
        data: Numpy array or list with the voltage data of the bubbles.
        labels: Labels of the data. If n_crops=2, you must pass labels!
        length: Number of timesteps of the cropped part (default: 500).
        n_crops: Number of crops (1 or 2). If 1, makes one zoomed-in sample per bubble. If 2, creates two crops.

    Output:
        cropped_data: Numpy array with the cropped voltages, dimension [#samples, length].
        labels: Labels duplicated to match the cropped_data. If n_crops=1, the labels are returned as is.
    """
    if n_crops not in [1, 2]:
        raise ValueError("n_crops should be either 1 or 2")
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if n_crops == 1:
        cropped_data = np.array([sample[:length] for sample in data])
    if n_crops == 2:
        if labels is None:
            raise ValueError("Labels must be provided when n_crops=2.")
        cropped_data = np.array([sample[:length] for sample in data])
        cropped_data2 = np.array([sample[length//2:length + length//2] for sample in data])
        cropped_data = np.concatenate([cropped_data, cropped_data2])
        labels = np.concatenate([labels, labels])
    
    return cropped_data, labels


def valid_velo_data_cropped(data, length=500, n_crops=1):
    """
    Extracts the data with valid velocities and crops the voltage signals.

    Args:
        data: DataFrame with bubble voltages and labels.
        length: Number of timesteps for the cropped data.
        n_crops: Number of crops (1 or 2). Default is 1.

    Returns:
        x: Numpy array with cropped voltage data of all valid bubbles.
        y: Numpy array with labels of all valid bubbles.
    """
    valid = data[data["VeloOut"] != -1]
    x, y = frame_waves(valid["VoltageOut"].tolist(), length=length, labels=valid["VeloOut"], n_crops=n_crops)
    if n_crops == 1:
        y = valid["VeloOut"].astype(float).tolist()
    return np.array(x), np.array(y)


filename = '/kaggle/input/bubbles-1/2024-11-12T145426_seperate.csv'
data = read_seperate_csv_from_zip(filename)
print("Data loaded")
