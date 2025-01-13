import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import dataloading 


def scale_time(data, length):
    """
    Scales all data to <length> timesteps; adds padding to start and end of the signal.

    Args: 
        data: Numpy array with dimension [samples, datapoints]
        length: length all samples will be changed into. length should be longer than the longest data signal.

    Returns:
        Numpy array with dimension [samples, datapoints] with <length> datapoints.
    """

    return




