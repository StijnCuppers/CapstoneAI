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
Go to main.py and add the the path of the folder with the probedata you want to analyze in the "path_to_zips =" list.
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

