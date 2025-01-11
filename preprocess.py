import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import torch

#Folder containing .bin, .binlog, (.evt) of one run
input_folder = R"C:\Users\TUDelft\Desktop\bubble_data"


def find_files(folder_path):
    """
    Find .bin and .bin.log and (if avalible) .evtlog files in given folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        tuple: Paths to the .bin_file, .binlog_file and .evt_file (None if not found).
    """
    bin_file = None
    metadata_file = None
    evt_file = None
    for file in os.listdir(folder_path):
        if file.endswith(".bin") and "_stream" not in file:
            bin_file = os.path.join(folder_path, file)
        elif file.endswith(".binlog"):
            metadata_file = os.path.join(folder_path, file)
        elif file.endswith(".evt"):
            evt_file = os.path.join(folder_path, file)
    return bin_file, metadata_file, evt_file


def get_metadata(metadata_file):
    """
    Extracts metadata from .binlog.

    Args:
        metadata_file (str): Path to metadata file (.binlog)

    Returns:
        dict: Metadata including channelCoef1, channelCoef2, acquisitionFrequency and acquisitionComment.
    """
    tree = ET.parse(metadata_file)
    root = tree.getroot()

    metadata = {
        "channelCoef1": float(root.find(".//channel").attrib['channelCoef1']),
        "channelCoef2": float(root.find(".//channel").attrib['channelCoef2']),
        "acquisitionFrequency": float(root.attrib['acquisitionFrequency']),
        "acquisitionComment": (root.attrib['acquisitionComment']),
        "bin_file": root.find(".//channel").attrib['channelOutputFile']    
    }
    return metadata


def get_bubbles(bin_file, coef1, coef2):

    trans_data = np.fromfile(bin_file, dtype = ">i2")
    voltage_data = trans_data * coef2 + coef1
    
    # Threasholds for bubble detection
    lower_threashold = coef1
    upper_threashold = 0.40 + coef1 

    #Initializing bubble detection
    bubbles_detect = []
    in_bubble = False
    tA = None
    tE = None

    # Detect bubbles 
    for i, voltage in enumerate(voltage_data):
        # Bubble entry
        if not in_bubble:
            if voltage > upper_threashold:
                tA = i
                in_bubble = True
        # Bubble exit
        else:
            if voltage < lower_threashold:
                tE = i
                bubbles_detect.append((len(bubbles_detect), tA, tE))
                in_bubble = False
                tA = None
                tE = None
    print(f"Bubbles extracted: {len(bubbles_detect)}")
    return bubbles_detect



def bubble_labels(evt_file):
    """
    Extracts bubble speeds from the evt_file 

    Args:
        evt_file (str): Path to eventlog file (.evt)

    Returns:

    """


if __name__ == "__main__":
    folder_path = input_folder

    bin_file, metadata_file, evt_file = find_files(folder_path)
    if not bin_file or not metadata_file:
        print(".bin or .binlog file not found. Exiting script.")
        sys.exit(1)
    print(f"Folder path: {folder_path}")
    print(f"Binary file: {bin_file}")
    print(f"Metadata file: {metadata_file}")
    print(f"Event log file: {evt_file}")

    metadata = get_metadata(metadata_file)
    print(f"Extracted metadata:\n {metadata}")
    coef1 = metadata["channelCoef1"]
    coef2 = metadata["channelCoef2"]

    bubbles_detect = get_bubbles(bin_file, coef1, coef2)

    print(f"Total voltage values: {len(voltage_data)}")
    print(f"Total bubbles extracted: {len(bubbles)}")

