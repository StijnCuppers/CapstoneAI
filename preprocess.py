import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import torch
import Evtx.Evtx as evtx

#Folder containing .bin, .binlog, (.evt, .evtlog) of one run
input_folder = R"INPUT_PATH"


def find_files(folder_path):
    """
    Find .bin and .bin.log and (if avalible) .evtlog files in given folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        tuple: Paths to the .bin_file, .binlog_file and .evtlog_file (None if not found).
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

    print(f"Folder path: {folder_path}")
    print(f"Binary file: {bin_file}")
    print(f"Metadata file: {metadata_file}")
    print(f"Event log file: {evt_file}")

    return bin_file, metadata_file, evt_file

"-------------------------------------------------------------------------------------------------------"


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

    print(f"Extracted metadata:\n {metadata}")

    return metadata


"-------------------------------------------------------------------------------------------------------"

def get_bubbles(bin_file, coef1, coef2, w):
    """
    Extracts bubble entries and exists implementing dual-threasholding strategy.
    
    Args:
        bin_file (str): Path to the binary file (.bin).
        coef1 (float): Channel coefficient 1 (offset).
        coef2 (float): Channel coefficient 2 (scaling factor).
        w (int): Bubble window for bubble detection
        
    Returns:
        tuple: (voltage_data, bubbles) where:
            -voltage_data (array): Array of voltage values.
            -bubbles (list): Tuple list of bubble data (tA0, tA, tA1, tE0, tE, tE1).
    """    
    # Read binary data and apply conversion to voltage
    trans_data = np.memmap(bin_file, dtype=">i2", mode="r")
    voltage_data = trans_data * coef2 + coef1
    
    # Threasholds for bubble detection
    lower_threashold = coef1
    upper_threashold = 0.20 + coef1 
    bubbles = []
    in_bubble = False
    last_lower, tA, tE = None, None, None

    # Detecting bubbles
    for i, voltage in enumerate(voltage_data):
        if not in_bubble:
            if voltage < lower_threashold:
                last_lower = i
            # Valid entry
            if last_lower is not None and voltage > upper_threashold:
                tA = last_lower
                tA0 = max(0, tA - w)
                tA1 = max(0, tA + w)
                in_bubble = True
        else:
            if voltage < upper_threashold:
                tE = i
                # Valid exit
                if voltage < lower_threashold:
                    tE = i - 1
                    tE0 = min(len(voltage_data) - 1, tE - w)
                    tE1 = min(len(voltage_data) - 1, tE + w)
                    bubbles.append((tA0, tA, tA1, tE0, tE, tE1))
                    in_bubble = False
                    last_lower = None

    print(f"Bubbles detected: {len(bubbles)}")
    return voltage_data, bubbles
    
"-------------------------------------------------------------------------------------------------------"


def bubble_labels(evt_file):
    """
    Extracts bubble speeds from the evt_file 

    Args:
        evt_file (str): Path to eventlog file (.evt)

    Returns:
        list with arrival time (t_a), exit time (t_e), arrival velocity (v_a), 
        exit velocity (v_e) for every bubble in chronological order.
        [[t_a, t_e, v_a, v_e], [...], ...]
    """
    return -1 # temporary value while I work on the code

"-------------------------------------------------------------------------------------------------------"


if __name__ == "__main__":
    folder_path = input_folder

    bin_file, metadata_file, evt_file = find_files(folder_path)
    if not bin_file or not metadata_file:
        print(".bin or .binlog file not found. Exiting script.")
        sys.exit(1)

    metadata = get_metadata(metadata_file)
    coef1 = metadata["channelCoef1"]
    coef2 = metadata["channelCoef2"]

    voltage_data, bubbles_detect = get_bubbles(bin_file, coef1, coef2, w=10000)