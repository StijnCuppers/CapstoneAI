import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import torch

# Folder containing .bin, .binlog, (.evt, .evtlog) of one run
# Replace with your own input path for now: R"input path"
input_folder = R"C:\Users\Silke\Documents\GitHub\CapstoneAI\Data"


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
        elif file.endswith(".evt") and "_stream" not in file:
            evt_file = os.path.join(folder_path, file)
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
    return metadata


"-------------------------------------------------------------------------------------------------------"


def get_bubbles(bin_file, coef1, coef2):

    trans_data = np.fromfile(bin_file, dtype = ">i2")
    voltage_data = trans_data * coef2 + coef1
    print(f"Total voltage values: {len(voltage_data)}")
    
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
    return voltage_data, bubbles_detect


"-------------------------------------------------------------------------------------------------------"


def bubble_labels(evt_file):
    """
    Extracts bubble speeds from the evt_file for bubbles

    Args:
        evt_file (str): Path to eventlog file (.evt)

    Returns:
        pandas DataFrame with entry time (t_a), exit time (t_e), velocity in (v_a), 
        velocity out (v_e) for every bubble in chronological order.
    """
    def evt_dataframe(evt_file):
        "Function that puts the .evt file into a dataframe (note: NOT the _stream.evt, but the other one)"
        with open(evt_file, 'rb') as file:
            content = file.read()

        # Converts content to string
        content_str = content.decode('latin1')  # Adjust encoding if necessary

        # Splits the content into lines
        lines = content_str.splitlines()

        # Splits each line into columns based on tab and makes a DataFrame
        data = [line.split('\t') for line in lines]
        df = pd.DataFrame(data[1:], columns=data[0])
        
        return df
    
    # Replace commas with periods (as decimals are now seperated with commas)
    evt_df = evt_dataframe(evt_file)
    evt_df = evt_df.replace(",", ".", regex=True)
    evt_df = evt_df.astype(float)
  
    return evt_df


"-------------------------------------------------------------------------------------------------------"


def v_total_labels(evt_file):
    """
    Extracts bubble labels (entry and exit velocities) for bubbles that have both v_in and v_out

    Args:
        evt_file (str): Path to eventlog file (.evt)

    Returns:
        numpy array containing lists with [t_a, t_e, v_in, v_out] for each 
        t_a = arrival time (Entry),  t_e = exit time (Exit),
        v_in = arrival velocity (VeloIn),  v_out = exit velocity (VeloOut)

    """ 
    evt_df = bubble_labels(evt_file)

    # Only sampling bubbles with VeloIn and VeloOut
    valid_bubbles = evt_df[(evt_df["VeloOut"]!=-1) & (evt_df["VeloIn"]!=-1)]
    labels = np.array([valid_bubbles["Entry"], valid_bubbles["Exit"],
                       valid_bubbles["VeloIn"], valid_bubbles["VeloOut"]])
    labels = np.transpose(labels)

    return labels


def v_out_labels(evt_file):
    """
    Extracts bubble labels (exit velocities) for bubbles that have a v_out

    Args:
        evt_file (str): Path to eventlog file (.evt)

    Returns:
        numpy array containing lists with [t_a, t_e, v_out] for each 
        t_a = arrival time (Entry),  t_e = exit time (Exit),
        v_out = exit velocity (VeloOut)

    """ 
    evt_df = bubble_labels(evt_file)

    # Only sampling bubbles with VeloOut
    valid_bubbles = evt_df[(evt_df["VeloOut"]!=-1)]
    labels = np.array([valid_bubbles["Entry"], valid_bubbles["Exit"], valid_bubbles["VeloOut"]])
    labels = np.transpose(labels)

    return labels


def v_in_labels(evt_file):
    """
    Extracts bubble labels (exit velocities) for bubbles that have both v_in

    Args:
        evt_file (str): Path to eventlog file (.evt)

    Returns:
        numpy array containing lists with [t_a, t_e, v_in] for each 
        t_a = arrival time (Entry),  t_e = exit time (Exit),
        v_in = arrival velocity (VeloIn)

    """ 
    evt_df = bubble_labels(evt_file)

    # Only sampling bubbles with VeloIn
    valid_bubbles = evt_df[(evt_df["VeloIn"]!=-1)]
    labels = np.array([valid_bubbles["Entry"], valid_bubbles["Exit"],
                       valid_bubbles["VeloIn"]])
    labels = np.transpose(labels)

    return labels

"--------------------------------------------------------------------------------------------------------"


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

    # TEMPORARILY COMMENT THIS BLOCK TO SAVE TIME WHEN TESTING OTHER FUNCTIONS; IT TAKES A LONG TIME
    metadata = get_metadata(metadata_file)
    print(f"Extracted metadata:\n {metadata}")
    coef1 = metadata["channelCoef1"]
    coef2 = metadata["channelCoef2"]
    voltage_data, bubbles_detect = get_bubbles(bin_file, coef1, coef2)

    print("v_total labels:")
    v_total_labels = v_total_labels(evt_file)
    print(v_total_labels)
    print(f"amount of bubbles with v_in and v_out: {len(v_total_labels)}")

    print("\nv_in labels:")
    v_in_labels = v_in_labels(evt_file)
    print(v_in_labels)
    print(f"amount of bubbles with v_in: {len(v_in_labels)}")

    print("\nv_out labels:")
    v_out_labels = v_out_labels(evt_file)
    print(v_out_labels)
    print(f"amount of bubbles with v_out: {len(v_out_labels)}")

