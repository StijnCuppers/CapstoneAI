# Capstone project
This repository is a submission to the TI3150TU Capstone AI project of the TU Delft for the project: Enhanced Bubble Characterization from Optical Fiber Probe Data. The aim of this project is to utilize AI framworks for the analysis of bubble flow in bubble column reactors, utilizing a fiber probe. Mainly our models are used to handle the given bubble data to seperate the bubbles and, if the .evtlog file is given, add the corresponding labels. After the preprocessing, 3 trained models will give the predictions of the speeds of the bubbles. The predictions of the 3 models will be averaged and the average speed of the 3 will be the final prediction.  


## Installation


The following libraries are required for this project:

These libraries are included with Python by default:
- `os` — File and directory management.
- `re` — Regular expressions for string matching.
- `sys` — System-specific parameters and functions.
- `pickle` — Object serialization.
- `zipfile` — Handling ZIP archives.
- `random` — Generating random numbers.
- `xml.etree.ElementTree` — Parsing and creating XML data.

---

These libraries are used for handling, analyzing, and manipulating data:
- `numpy` — Numerical computations and array processing.
- `pandas` — Data manipulation and analysis.

---

- `scikit-learn` — Provides tools for data preprocessing, model selection, and evaluation:
  - `train_test_split`
  - `KFold`
  - `StandardScaler`

---

- `torch` — PyTorch library for building and training neural networks.
- `torch.nn` — For creating neural network layers.
- `torch.optim` — Optimization algorithms.

---
- `scipy.signal` — Tools for signal processing, specifically:
  - `find_peaks` — For identifying peaks in signal data.

---

- `matplotlib` — Plotting and visualizing data.

## Usage
To use our code you should clone the repository.
Go to main.py and add the path of the folder with the probedata you want to analyze in the "path_to_zips =" list.
Add the desired output location to the "path_to_output" variable.
Make sure you put "r" before the path if '\' is used in the paths.
The folder should contain a .bin and a .binlog file and if wanted contain a .evtlog file.
If the .evtlog is available and labels = True it will check the predictions. 
Otherwise set labels = False and it will not use the .evtlog.
then run the code and it will analyze your bubbles. 
The output will be saved in the form of a csv in your specified output path.

The code will provide you a short summary of the dataframe and the amount of bubbles which have no label. After this it will give the predictions of the 3 models. The outcomes will be averaged to obtain a final prediction. This prediction is possible to check when the labels are available it will then give the standard deviation and the percentage. 
it will also give the amount of bubbles it can predict with an uncertainty lower than 10%. 
In the given folder it will add the all_bubbles.zip, a CSV-file with the data, and a .PNG with an example of the bubbles how they are extracted.

## Main functions

### **1. DataLoading**  
The purpose of data loading is to segment bubble exits from the raw input data. This process involves the following steps:  

#### **Data Extraction and Conversion**  
- Raw binary data (`.bin`) is extracted and transformed using channel coefficients (`Coef1`, `Coef2`), which convert binary voltage signals (`Vtrans`) into decimal voltage values (`Vdata`):  
- Due to the large dataset size (1,482,760,632 data points), voltage values are converted to `float32` format for optimal memory usage.

#### **Segmentation Algorithm**  
- **Downsampling**: The dataset is downsampled by a factor of 5 to reduce computational time.  
- **Smoothing**: A moving average filter (window size = 100) is applied to smooth voltage values, highlighting long-term trends.  
- **Gradient Analysis**: The smoothed signal's gradient is computed to detect significant voltage rises and drops.  
- **Peak Detection**: Peaks corresponding to bubble exits are identified by sudden drops in the signal gradient. These peaks are mapped back to their original data points.  
- **Region of Interest (ROI)**: For each detected bubble peak (`tE`), the ROI is defined as 1,000 points before (`tE0`) and 3,000 points after (`tE1`). These values, along with `Vdata`, are plotted.


#### **Dataset Composition**  
- The final dataset includes:
- Segmented voltage values.
- Exit velocities (if available).
- Acquisition frequency and flow rate for each bubble.
- Validation: The pipeline detects 3,437 bubbles across datasets with varying flow rates, compared to 3,239 events detected by the current bubble analyzer.

---

### **2. Preprocessing**  
Three neural network models (GRU 1, GRU 2, and LSTM) rely on preprocessing pipelines.  

#### **Steps in Preprocessing**  
1. **Frame Waves**:  
 - Voltage values are cropped and zoomed for each bubble.  
 - Two models (GRU 1 and LSTM) use:
   - Signal length = `150`
   - Jump = `0` (focusing on signal ends to reduce noise).  
 - GRU 2 uses:
   - Signal length = `250`
   - Jump = `500`.
2. **Scaling**:  
 - Data is normalized using `sklearn`’s `StandardScaler`, with training data and targets scaled to a mean of 0 and standard deviation of 1.  
 - Scaling parameters:  
   - `feature_scaler_1`: For models with `length=150, jump=0`.  
   - `feature_scaler_2`: For the model with `length=250, jump=500`.  
   - `target_scaler_1`: Common across all models.

---

### **3. Neural Network Models**  
Three models were trained on the segmented and preprocessed data, using a 50-25-25 train-validation-test split for optimization and evaluation.  

#### **GRU 1**  
- **Input**: Sequential data `[batch_size, number of timesteps, number of features]`.  
- **Architecture**: GRU layer → Fully connected layer.  
- **Training**: 1,500 epochs.  
- **Best Hyperparameters**:
- Learning rate: `0.01`  
- Hidden size: `20`  
- Norm: `2`  
- Layers: `2`

#### **GRU 2**  
- **Input**: Similar to GRU 1 but processes different input data.  
- **Architecture**: GRU layer → Fully connected layer.  
- **Training**: 700 epochs.  
- **Best Hyperparameters**:
- Learning rate: `0.02`  
- Hidden size: `20`  
- Norm: `3`  
- Layers: `2`

#### **LSTM**  
- **Input**: Similar to GRU models.  
- **Architecture**: LSTM layer → Fully connected layer.  
- **Training**: 1,500 epochs.  
- **Best Hyperparameters**:
- Learning rate: `0.008`  
- Hidden size: `30`  
- Norm: `3`  
- Layers: `2`

## Used neural networks
To analyze the data we use two GRU models both trained on different preprocessed data. This resulted in a GRU1 and GRU2 model. 
The GRU1 model is trained on sequential data with a length of 150 and 1500 epochs and resulted in a learning rate of 0.01, a hidden size of 20 a norm of 2  and a number of 2 layers.
The GRU2 model is trained on sequential data with a length of 250 and 1500 epochs and resulted in a learning rate of 0.02, a hidden size of 20 a norm of 3  and a number of 2 layers.
The LSTM model was also trained on the sequential data with a lengt of 150 and 1500 epochs and resulted in  a learning rate of 0.008, a hidden size of 30 a norm of 3 and a number of 2 layers.


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

