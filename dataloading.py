import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import zipfile

# Folder containing .bin, .binlog, (.evt, .evtlog) of one run
input_folder = R"C:\Users\Silke\Documents\GitHub\CapstoneAI\Data"


def find_files(folder_path):
    """
    Find .bin and .bin.log and (if available) .evtlog files in given folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        tuple: Paths to the .bin_file, .binlog_file .evtlog_file (None if not found) and run_name.
    """
    bin_file = None
    metadata_file = None
    evt_file = None
    run_name = None

    for file in os.listdir(folder_path):
        if file.endswith(".bin") and "_stream" not in file:
            bin_file = os.path.join(folder_path, file)
            run_name = os.path.splitext(file)[0]
        elif file.endswith(".binlog"):
            metadata_file = os.path.join(folder_path, file)
        elif file.endswith(".evt") and "_stream" not in file:
            evt_file = os.path.join(folder_path, file)
        
    return bin_file, metadata_file, evt_file, run_name


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

   
def save_bubbles(voltage_data, bubbles, mode="whole", run_name=None):
    """
    Save bubbles to a Dataframe and CSV file.

    Args:
        voltage_data (array): Array of voltage data
        bubbles (list): Tuple list of bubble data (tA0, tA, tA1, tE0, tE, tE1)
        mode (str): Determines what voltage data to include:
            - "whole": Voltage from tA0 to tE1 (whole bubble)
            - "seperate" Voltage from tA0 to tA1 (entry) and tE0 to tE1 (exit)
            - "entry": Voltage from tA0 to tA1 (entry)
            - "exit": Voltage from tE0 to tE1 (exit)
        run_name (str): Base name of the run

    Returns:
        pd.DataFrame: containing bubble details and voltage segments
    """
    bubble_data = []

    for idx, (tA0, tA, tA1, tE0, tE, tE1) in enumerate(bubbles):
        voltage_bubble = voltage_data[tA0:tE1 + 1]
        voltage_entry = voltage_data[tA0:tA1 + 1]
        voltage_exit = voltage_data[tE0:tE1 + 1]

        bubble_info = {
            "bubble": idx + 1,
            "tA0" : tA0,
            "tA": tA,
            "tA1": tA1,
            "tE0": tE0, 
            "tE": tE,
            "tE1": tE1   
        }

        if mode == "whole":
            bubble_info["voltage_full"] = voltage_bubble.tolist()
        elif mode == "seperate":
            bubble_info["voltage_entry"] = voltage_entry.tolist()
            bubble_info["voltage_exit"] = voltage_exit.tolist()
        elif mode == "entry":
            bubble_info["voltage_entry"] = voltage_exit.tolist()
        elif mode == "exit":
            bubble_info["voltage_exit"] = voltage_exit.tolist()
        bubble_data.append(bubble_info)

    bubble_df = pd.DataFrame(bubble_data)

    output_file = f"{run_name}_{mode}.csv"
    bubble_df.to_csv(output_file, index=False)
    print(f"Bubble data saves to {output_file}")

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

    bin_file, metadata_file, evt_file, run_name = find_files(folder_path)
    if not bin_file or not metadata_file:
        print(".bin or .binlog file not found. Exiting script.")
        sys.exit(1)

    metadata = get_metadata(metadata_file)
    coef1 = metadata["channelCoef1"]
    coef2 = metadata["channelCoef2"]

    # # !!! IMPORTANT !!! These lines are commented because they read the voltage file and save them as zipped .csv files
    # # If you do not have the zipped file, run this only once and then comment again. It takes some time and a lot of storage.
    
    # voltage_data, bubbles = get_bubbles(bin_file, coef1, coef2, w=2500)
    # bubbles_whole_df = save_bubbles(voltage_data, bubbles, mode = "whole", run_name=run_name)
    # print(bubbles_whole_df.head())
    # bubbles_seperate_df = save_bubbles(voltage_data, bubbles, mode = "seperate", run_name=run_name)
    # print(bubbles_seperate_df.head())
    # zip_all_csv_files('all_bubbles.zip')

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

