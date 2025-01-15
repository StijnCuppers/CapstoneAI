import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import zipfile
import sys



# Folder containing .bin, .binlog, (.evt, .evtlog) of one run
input_folder = R"C:\Users\TUDelft\Desktop\bubble_data"


def find_files(folder_path):
    """
    Find .bin, .bin.log, run name and (if needed) .evtlog in given folder.

    Args:
        folder_path (str): Path to the input folder.

    Returns:
        tuple: Paths to the .bin_file, .binlog_file run name and .evtlog_file (None if not found)
    """
    bin_file = metadata_file = run_name = evt_file = None

    for file in os.listdir(folder_path):
        if file.endswith(".bin") and "_stream" not in file:
            bin_file = os.path.join(folder_path, file)
            run_name = os.path.splitext(file)[0]
        elif file.endswith(".binlog"):
            metadata_file = os.path.join(folder_path, file)
        elif file.endswith(".evt") and "_stream" not in file:
            evt_file = os.path.join(folder_path, file)
        
    return bin_file, metadata_file, run_name, evt_file 


def get_metadata(metadata_file):
    """
    Extracts metadata from .binlog.

    Args:
        metadata_file (str): Path to metadata file (.binlog).

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


def get_bubbles(bin_file, coef1, coef2, w):
    """
    Extracts bubble entries and exists implementing dual-threasholding strategy.
    
    Args:
        bin_file (str): Path to the binary file (.bin).
        coef1 (float): Channel coefficient 1 (offset).
        coef2 (float): Channel coefficient 2 (scaling factor).
        w (int): Bubble window for bubble detection.
        
    Returns:
        voltage_data (np.array): Array of voltage values.
        bubbles (lst): Tuple list of bubble data (tA0, tA, tA1, tE0, tE, tE1).
    """    
    trans_data = np.memmap(bin_file, dtype=">i2", mode="r")
    voltage_data = trans_data * coef2 + coef1
    
    lower_threashold = coef1
    upper_threashold = 0.20 + coef1 
    bubbles = []
    in_bubble, last_lower, tA, tE = False, None, None, None

    for i, voltage in enumerate(voltage_data):
        if not in_bubble:
            if voltage < lower_threashold:
                last_lower = i
            if last_lower is not None and voltage > upper_threashold:
                tA = last_lower
                tA0 = max(0, tA - w)
                tA1 = max(0, tA + w)
                in_bubble = True
        else:
            if voltage < upper_threashold:
                tE = i
                if voltage < lower_threashold:
                    tE = i - 1
                    tE0 = min(len(voltage_data) - 1, tE - w)
                    tE1 = min(len(voltage_data) - 1, tE + w)
                    bubbles.append((int(tA0), int(tA), int(tA1), int(tE0), int(tE), int(tE1)))
                    in_bubble, last_lower = False, None

    return voltage_data, bubbles


def get_labels(evt_file):
    """
    Extracts bubbles labels and label summary from the evt_file.

    Args:
        evt_file (str): Path to eventlog file (.evt).

    Returns:
        bubble_labels (lst): List of tuples containing (Entry, Exit, Veloc, VeloIn, VeloOut).
    """
    with open(evt_file, 'rb') as file:
        content = file.read()

    content_str = content.decode('latin1') 
    lines = content_str.splitlines()
    data = [line.split('\t') for line in lines]

    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=headers)
    df = df.replace(",", ".", regex=True) 

    df["Entry"] = df["Entry"].astype(int)
    df["Exit"] = df["Exit"].astype(int)
    df["Veloc"] = df["Veloc"].astype(float)
    df["VeloIn"] = df["VeloIn"].astype(float)
    df["VeloOut"] = df["VeloOut"].astype(float)
    
    valid_bubbles_df = df[df["Veloc"] != -1]
    bubble_labels = list(valid_bubbles_df[["Entry", "Exit", "VeloIn", "VeloOut"]].itertuples(index=False, name=None))

    return bubble_labels


def save_bubbles(voltage_data, bubbles, mode="seperate", run_name=None, bubble_labels=None):
    """
    Save bubbles to a Dataframe and CSV file.

    Args:
        voltage_data (array): Array of voltage data
        bubbles (lst): Tuple list of bubble data (tA0, tA, tA1, tE0, tE, tE1)
        mode (str): Determines what voltage data to include:
            - "whole": Voltage from tA0 to tE1 (whole bubble)
            - "seperate" Voltage from tA0 to tA1 (entry) and tE0 to tE1 (exit)
        run_name (str): Base name of the run (used for saving files).
        bubble_labels (list of tuples): Ground truth labels as (Entry, Exit, Veloc, VeloIn, VeloOut) for labelin.

    Returns:
        bubble_df (pd.DataFrame): containing bubble details and voltage segments
    """
    bubble_data = []

    for idx, (tA0, tA, tA1, tE0, tE, tE1) in enumerate(bubbles):
        # Bubble index
        bubble_info = {"bubble": idx + 1}

        # Velocity labels 
        if bubble_labels:
            velo_in = velo_out = -1
            for Entry, Exit, VeloIn, VeloOut in bubble_labels:
                if Entry <= tE and Exit >= tA:  
                    velo_in = VeloIn
                    velo_out = VeloOut
                    break   
            bubble_info["VeloIn"] = velo_in
            bubble_info["VeloOut"] = velo_out

        # Voltage data
        if mode == "whole":
            bubble_info["voltage_full"] = voltage_data[tA0:tE1 + 1].tolist()
        elif mode == "seperate":
            bubble_info["voltage_entry"] = voltage_data[tA0:tA1 + 1].tolist()
            bubble_info["voltage_exit"] = voltage_data[tE0:tE1 + 1].tolist()
        
        bubble_data.append(bubble_info)

    bubble_df = pd.DataFrame(bubble_data)
    output_file = f"{run_name}_{mode}.csv"
    bubble_df.to_csv(output_file, index=False, sep=";")
    return bubble_df


def zip_all_csv_files(zip_filename):
    """
    Zip all CSV files in the current directory into a single ZIP file.

    Args:
        zip_filename (str): Name of the output ZIP file
    """
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir():
            if file.endswith('.csv'):
                zipf.write(file)
                print(f"Added {file} to {zip_filename}")

    print(f"All CSV files zipped as {zip_filename}")

def dataloading():
    
    return None


if __name__ == "__main__":
    folder_path = input_folder

    bin_file, metadata_file, run_name, evt_file = find_files(folder_path)
    if not bin_file or not metadata_file:
        print(".bin or .binlog file not found. Exiting script.")
        sys.exit(1)

    metadata = get_metadata(metadata_file)
    coef1 = metadata["channelCoef1"]
    coef2 = metadata["channelCoef2"]

    bubble_labels = get_labels(evt_file)
    voltage_data, bubbles = get_bubbles(bin_file, coef1, coef2, w=2000)
    bubbles_df= save_bubbles(voltage_data, bubbles, mode="seperate", run_name=run_name, bubble_labels=bubble_labels)
    zip_all_csv_files('all_bubbles.zip')