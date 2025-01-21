import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.signal import find_peaks
import zipfile

########################################################
# TABLE OF CONTENTS

# find_files: Locate and extract file paths for `.bin`, `.binlog`, and `.evt` files from a folder.
# get_binlogdata: Parse `.binlog` files to extract metadata such as coefficients, frequency, and comments.
# get_labels: Extract bubble labels from `.evt` files with valid velocities (`VeloOut != -1`).
# get_bubbles_advanced: Extract bubble entries and exits using a dual-thresholding strategy. 
#                       Includes downsampling, smoothing, gradient computation, and peak detection. 
#                       Optionally plots results and saves the plot.
# plot_bubble_detection: Visualize voltage data, detected peaks, and entry/exit points. 
#                        Saves the plot in the specified folder.
# save_bubbles: Save extracted bubble data into a CSV file. Match bubble data with labels and identify missing labels.
# process_folder: Process a single folder containing bubble run data and generate a CSV.
# process_main_folder: Process all subfolders in a main folder. Combine data from all folders into a single CSV file.
# zip_all_csv_files: Zip all CSV files in the main folder and its subfolders into a single ZIP file.
# Main Execution: Process the main folder with all subfolders, generate combined CSV, and ZIP file.
########################################################


def find_files(folder_path):
    """
    Find relevant files paths and the run name in given folder

    Args:
        folder_path (str): Path to the input folder.

    Returns:
        tuple: Paths to .bin_file, .binlog_file .evtlog_file and the run_name
    """
    
    bin_file = binlog_file = run_name = evt_file = None

    for file in os.listdir(folder_path):
        if file.endswith(".bin") and "_stream" not in file:
            bin_file = os.path.join(folder_path, file)
            run_name = os.path.splitext(file)[0]
        elif file.endswith(".binlog"):
            binlog_file = os.path.join(folder_path, file)
        elif file.endswith(".evt") and "_stream" not in file:
            evt_file = os.path.join(folder_path, file)
        
    return bin_file, binlog_file, evt_file, run_name 


def get_binlogdata(binlog_file):
    """
    Extracts binlogdata from .binlog.

    Args:
        binlog_file (str): Path to binlog file (.binlog).

    Returns:
        dict: Metadata including channelCoef1, channelCoef2, flowRate and acquisitionComment.
    """
    tree = ET.parse(binlog_file)
    root = tree.getroot()

    acquisition_comment = root.attrib.get('acquisitionComment', '')
    flow_rate_match = re.search(r'(\d+)\s*[lL][/-]?[mM]in', acquisition_comment)
    if flow_rate_match:
        flow_rate = int(flow_rate_match.group(1)) 
    else:
        flow_rate = -1 

    binlogdata = {
        "channelCoef1": float(root.find(".//channel").attrib['channelCoef1']),
        "channelCoef2": float(root.find(".//channel").attrib['channelCoef2']),
        "acquisitionFrequency": float(root.attrib['acquisitionFrequency']),
        "flowRate": flow_rate, 
        "bin_file": root.find(".//channel").attrib['channelOutputFile']    
    }

    print("Binlog data extracted")
    return binlogdata


def get_labels(evt_file):
    """
    Extracts bubble labels with VeloOut != -1 from the evt_file.

    Args:
        evt_file (str): Path to eventlog file (.evt).

    Returns:
        extracted_bubbles (list): List of tuples containing (L_idx, Exit, VeloOut) where VeloOut != -1.
    """
    with open(evt_file, 'rb') as file:
        content = file.read()

    lines = content.decode('latin1').splitlines()
    data = [line.split('\t') for line in lines]

    headers, rows = data[0], data[1:]
    exit_idx = headers.index("Exit")
    veloout_idx = headers.index("VeloOut")

    extracted_bubbles = []
    valid_idx = 0  

    for row in rows:
        # Extract and process Exit and VeloOut fields
        exit_value = int(row[exit_idx])
        veloout_value = float(row[veloout_idx].replace(",", "."))
        
        # Include only labels where VeloOut != -1
        if veloout_value != -1:
            extracted_bubbles.append(["L" + str(valid_idx), exit_value, veloout_value])
            valid_idx += 1 

    print(f"LABELS: {len(extracted_bubbles)} bubble labels with VeloOut != -1 extracted.")
    return extracted_bubbles


def get_bubbles_advanced(bin_file, coef1, coef2, plot=False, folder_path=None, run_name=None):
    """
    Extracts bubble entries and exits implementing dual-thresholding strategy.

    Args:
        bin_file (str): Path to the binary file (.bin).
        coef1 (float): Channel coefficient 1 (offset).
        coef2 (float): Channel coefficient 2 (scaling factor).
        plot (bool): Whether to plot the results. Defaults to False.
        folder_path (str, optional): Path to the folder where the plot will be saved. Required if plot=True.
        run_name (str, optional): Name of the run for naming the plot file. Required if plot=True.
    
    Returns:
        list: Extracted bubble data.
    """
    trans_data = np.memmap(bin_file, dtype=">i2", mode="r")
    voltage_data = (trans_data.astype(np.float32) * coef2 + coef1)
    print(f"{len(voltage_data)} datapoints extracted")

    downsample_factor = 5
    voltage_data_downsampled = voltage_data[::downsample_factor]

    # Apply moving average for additional smoothing
    window_size = 100 
    kernel = np.ones(window_size) / window_size
    smoothed_voltage_data = np.convolve(voltage_data_downsampled, kernel, mode='valid')
    smoothed_voltage_data = np.concatenate((np.full(window_size - 1, smoothed_voltage_data[0]), smoothed_voltage_data))

    # Compute the gradient of the smoothed and averaged data
    gradient = np.gradient(smoothed_voltage_data)

    # Detect peaks in the negative gradient
    peaks, _ = find_peaks(-gradient, prominence=0.005, distance=1000) 

    tE = peaks * downsample_factor
    tE1 = tE - 1000
    tE1 = tE1[tE1 >= 0] 

    tE0 = tE1 - 4001
    tE0 = tE0[tE0 >= 0] 

    bubbles = []
    for idx, (start, end, peak) in enumerate(zip(tE0, tE1, tE)):
        if start >= 0 and end < len(voltage_data):
            voltage_out = voltage_data[start:end].tolist() 
            bubbles.append(["E"+str(idx), peak, voltage_out])

    # Plot if requested
    if plot:
        if folder_path is None or run_name is None:
            raise ValueError("Both `folder_path` and `run_name` must be provided when plot=True.")
        plot_bubble_detection(voltage_data, tE, tE1, tE0, n=5000000, folder_path=folder_path, run_name=run_name)

    return bubbles


def plot_bubble_detection(voltage_data, tE, tE1, tE0, n, folder_path, run_name):
    """
    Plots the results of the voltage data and all detected peaks, saves the plot, and allows code execution to continue.

    Args:
        voltage_data (ndarray): Original voltage data.
        tE (ndarray): Detected peaks in original indices.
        tE1 (ndarray): Entry indices.
        tE0 (ndarray): Exit indices.
        n (int): Number of points to plot from the original voltage data.
        folder_path (str): Path to the folder where the plot should be saved.
        run_name (str): Name of the current run to use for naming the plot file.
    """
    # Create the plot
    plt.figure(figsize=(15, 7))
    plt.plot(np.arange(len(voltage_data[:n])), voltage_data[:n], label="Original Voltage Data", color="blue", alpha=0.3)

    # Plot tE, tE1, and tE0 within the first `n` points
    valid_tE = tE[tE < n]
    plt.scatter(valid_tE, voltage_data[valid_tE], color="red", label="Detected Peaks (tE)", marker="x", s=50)

    valid_tE1 = tE1[tE1 < n]
    plt.scatter(valid_tE1, voltage_data[valid_tE1], color="purple", label="Exit (tE1)", marker="o", s=50)

    valid_tE0 = tE0[tE0 < n]
    plt.scatter(valid_tE0, voltage_data[valid_tE0], color="pink", label="Entry (tE0)", marker="o", s=50)

    # Labels and title
    plt.title("Voltage Data with Detected Peaks and Shifts")
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage")
    plt.legend()

    # Save the plot to the folder
    plot_file_name = f"{run_name}_bubbles_plot.png"
    plot_file_path = os.path.join(folder_path, plot_file_name)
    plt.savefig(plot_file_path, dpi=300)
    print(f"Plot saved to {plot_file_path}")

    # Close the plot to free memory and allow the code to continue
    plt.close()


def save_bubbles(extracted_bubbles, run_name, folder_path, bubble_labels, flow_rate, frequency):
    """
    Saves extracted bubble data to a Pandas DataFrame and identifies missing labels.

    Args:
        extracted_bubbles (list): A list of bubbles, where each bubble is [Bidx, tE, VoltageOut].
        run_name (str, optional): Name of the run for file naming. Defaults to None.
        bubble_labels (list, optional): List of labels where each label is [Lidx, ExitIdx, VeloOut]. Defaults to None.
        flow_rate (int): Flow rate of measurement in L/min.
        frequency (float): Frequency of the measurement.

    Returns:
        pd.DataFrame: A DataFrame containing [bubble_idx, B_idx, L_idx, VeloOut, VoltageOut, flowRate, Frequency].
    """
    rows = []
    if bubble_labels:
        missing_labels = [] 

    # Iterate through each bubble in the extracted bubbles
    for bubble_idx, (E_idx, tE, VoltageOut) in enumerate(extracted_bubbles):
        if bubble_labels:
            # Check if any label's ExitIdx is within the bubble's range
            matched_label = None
            for label in bubble_labels:
                L_idx, Exit_idx, VeloOut = label
                if tE - 1000 <= Exit_idx <= tE + 1000:
                    matched_label = (L_idx, VeloOut)
                    break

            # If a matching label is found, use its values
            if matched_label:
                L_idx, VeloOut = matched_label
            else:
                L_idx, VeloOut = -1, -1 
        else:
            # No labels provided
            L_idx, VeloOut = -1, -1

        # Append the bubble information to the rows
        rows.append({
            "bubble_idx": str(bubble_idx)+"_"+run_name,
            "E_idx": E_idx,
            "L_idx": L_idx,
            "VeloOut": VeloOut,
            "VoltageOut": VoltageOut,
            "FlowRate": flow_rate,  
            "Frequency": frequency
        })

    # Identify missing labels
    if bubble_labels:
        for label in bubble_labels:
            L_idx, Exit_idx, VeloOut = label
            found = False
            for _, tE, _ in extracted_bubbles:
                if tE - 1000 <= Exit_idx <= tE + 1000:
                    found = True
                    break
            if not found:
                missing_labels.append(label)

    # Create a DataFrame
    saved_bubbles = pd.DataFrame(rows)

    # Create a DataFrame
    saved_bubbles = pd.DataFrame(rows)

    # Save the DataFrame to a file in the specified folder
    if run_name:
        file_name = os.path.join(folder_path, f"{flow_rate}_{run_name}_bubbles.csv")
    else:
        file_name = os.path.join(folder_path, f"{flow_rate}_bubbles.csv")
    
    saved_bubbles.to_csv(file_name, index=False, sep=";")
    print(f"Saved bubbles to {file_name}")

    if bubble_labels: 
        # Print missing labels
        if missing_labels:
            print("\nMissing Labels:")
            for label in missing_labels:
                print(f"L_idx: {label[0]}, ExitIdx: {label[1]}, VeloOut: {label[2]}")
        else:
            print("No missing labels.")

    # Count and print bubbles with VeloOut != -1
    valid_bubbles = saved_bubbles[saved_bubbles["VeloOut"] != -1]
    print(f"EXTRACTED: {len(valid_bubbles)} bubbles have VeloOut != -1 out of {len(saved_bubbles)} total bubbles.")
    print(saved_bubbles.head())

    return saved_bubbles


def process_folder(folder_path, plot, labels):
    """
    Processes a single folder containing bubble run data.

    Args:
        folder_path (str): Path to the folder containing the data files.
        plot (bool): Whether to generate plots during processing.
        labels (bool): Whether to process labels.

    Returns:
        pd.DataFrame: A DataFrame containing the processed bubble data.
    """
    bin_file, binlog_file, evt_file, run_name = find_files(folder_path)

    binlogdata = get_binlogdata(binlog_file)
    coef1 = binlogdata["channelCoef1"]
    coef2 = binlogdata["channelCoef2"]
    flowRate = binlogdata["flowRate"]
    acquisitionFrequency = ["acquisitionFrequency"]

    print(binlogdata)

    extracted_bubbles = get_bubbles_advanced(bin_file, coef1, coef2, plot, folder_path, run_name)

    if labels:
        bubble_labels = get_labels(evt_file)
    else:
       bubble_labels = None 

    save_bubbles_df = save_bubbles(extracted_bubbles, run_name, folder_path, bubble_labels, flowRate, acquisitionFrequency)
    zip_all_csv_files(folder_path)
    return save_bubbles_df


def process_main_folder(main_folder_path, plot=False, labels=False):
    """
    Processes all subfolders in a main folder, saves individual CSVs, and combines all data.

    Args:
        main_folder_path (str): Path to the main folder containing subfolders with data.
        plot (bool): Whether to generate plots during processing.
        labels (bool): Whether to process labels.

    Returns:
        pd.DataFrame: A combined DataFrame containing data from all subfolders.
    """
    # Initialize a list to hold DataFrames from all subfolders
    combined_data = []

    # Loop through all subfolders in the main folder
    for subfolder in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder)
        if os.path.isdir(subfolder_path):  # Process only directories
            print(f"Processing folder: {subfolder_path}")
            try:
                # Process the subfolder and save its DataFrame
                df = process_folder(subfolder_path, plot=plot, labels=labels)
                combined_data.append(df)
            except Exception as e:
                print(f"Error processing folder {subfolder_path}: {e}")

    # Combine all DataFrames into one
    if combined_data:
        big_bubbles_data = pd.concat(combined_data, ignore_index=True)

        # Save the combined DataFrame to the main folder
        output_file = os.path.join(main_folder_path, "Combined_bubbles.csv")
        big_bubbles_data.to_csv(output_file, index=False, sep=";")
        print(f"Combined data saved to {output_file}")

        return big_bubbles_data
    else:
        print("No valid data found to combine.")
        return pd.DataFrame()


def zip_all_csv_files(main_folder):
    """
    Zip all CSV files in the main folder and its subfolders into a single ZIP file,
    but include them all as if in a single flat directory.

    Args:
        main_folder (str): Path to the main folder containing subfolders with CSV files.
    """
    zip_path = os.path.join(main_folder, "All_bubbles.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(main_folder):  
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(root, file)  
                    arcname = os.path.basename(file)  
                    zipf.write(full_path, arcname=arcname) 
                    print(f"Added {full_path} to {zip_path} as {arcname}")

    print(f"All CSV files in {main_folder} and its subfolders zipped as {zip_path}")


if __name__ == "__main__":
    main_folder_path = R"C:\Users\TUDelft\Desktop\new"
    big_bubbles_df = process_folder(main_folder_path, plot=True, labels=True)
    print("Processing complete.")




